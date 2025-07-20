import multiprocessing
import os
import json
import logging
import shutil
import time
import tempfile
from typing import Any, Dict, List, Optional
import cloudpickle
import base64
from io import BytesIO
import sys
import inspect
import copy
import asyncio
import uuid
import numpy as np
from PIL import Image
import signal
try:
    import psutil
except ImportError:
    psutil = None

from multiprocessing import shared_memory

class WorkflowContext:
    """
    Manages ephemeral data I/O for a single, stateful execution run.

    This class is designed to be instantiated for each run and is best used as a
    context manager (`with` statement) to ensure its resources are automatically
    cleaned up. It fully encapsulates the lifecycle of its storage backends.

    Key Principles:
    - **Run-Specific Lifecycle & Context Manager**: Each instance creates and manages
      its own temporary storage. Using it in a `with` block guarantees that the
      `shutdown()` method is called to release all resources.
    - **Unique ID as Key**: All data is stored and retrieved using a simple
      string `unique_id`, abstracting away internal details.
    - **Internal Key Closet**: Maintains an internal registry (`_key_closet`)
      that maps each `unique_id` to its storage location and type.
    - **Ephemeral by Default**: All stored data is temporary. Persistence is an
      explicit action handled by the `export_data()` method.
    - **Resilient Storage**: Features an automatic fallback from RAM to disk if the
      in-memory store runs out of space or allocation fails.
    """

    SERIALIZATION_EXT = ".pkl"

    def __init__(self, config_path: Optional[str] = None, max_ram_gb: Optional[float] = None, run_id: Optional[str] = None, create: bool = True):
        """
        Initializes the data handler for a new run.

        Args:
            config_path (str, optional): Path to a custom JSON config file.
                                         If None, uses the default.
            max_ram_gb (float, optional): A direct override for the max RAM usage in gigabytes. This has the
                                          highest priority.
            run_id (str, optional): Unique run ID. If None and create=True, generates a new one.
            create (bool): If True, creates a new context; if False, loads an existing one using run_id.
        """
        self.logger = self._get_logger()
        self._key_closet: Dict[str, Dict] = {}

        if create and run_id is None:
            run_id = uuid.uuid4().hex
        elif not create and run_id is None:
            raise ValueError("run_id must be provided when create=False")
        self.run_id = run_id
        self._creator = create
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'workflow_context.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self._log_message(f"Loaded configuration from {config_path}", "debug")
        except FileNotFoundError:
            self._log_message(f"Configuration file not found. Using default settings.", "warning")

        # Set up directories
        self._contexts_dir = os.path.join(tempfile.gettempdir(), config.get('CONTEXTS_DIR_NAME', "contexts"))
        os.makedirs(self._contexts_dir, exist_ok=True)
        self._run_dir = os.path.join(self._contexts_dir, self.run_id)
        self.receipt_path = os.path.join(self._run_dir, "receipt.json")

        if create:
            os.makedirs(self._run_dir, exist_ok=False)
            self._save_receipt()  # Save empty receipt
            self._log_message(f"Created new context for run_id: {self.run_id}", "info")
        else:
            if not os.path.exists(self.receipt_path):
                raise FileNotFoundError(f"No receipt found for run_id: {self.run_id}")
            self._load_receipt()
            self._log_message(f"Loaded existing context for run_id: {self.run_id}", "info")

        # Configure storage settings
        self._configure_storage(config, max_ram_gb)

        # Register signal handlers if creator
        if self._creator:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

        self._log_message("WorkflowContext is ready for this run.", "info")

    def __enter__(self):
        """Allows the class to be used as a context manager, returning itself on entry."""
        self._log_message("Entering context manager.", "debug")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures shutdown is called automatically when exiting a 'with' block."""
        self._log_message("Exiting context manager, shutting down...", "debug")
        self.shutdown()

    def _signal_handler(self, sig, frame):
        """Handles signals for graceful shutdown."""
        self._log_message(f"Received signal {sig}, shutting down...", "info")
        self.shutdown()
        sys.exit(1)

    def _configure_storage(self, config: Dict, max_ram_gb_override: Optional[float]):
        """
        Loads configuration and determines the final max RAM usage based on a
        priority system: Direct Override > Percentage Config > Fixed Config > Default.
        """

        source = ""
        final_size_gb = None

        # Priority 1: Direct override from the constructor
        if max_ram_gb_override is not None:
            final_size_gb = max_ram_gb_override
            source = "direct override"

        # Priority 2: Percentage of total memory from config
        elif 'MAX_RAM_PERCENT' in config:
            if psutil is None:
                self._log_message("`psutil` is not installed. Cannot use percentage-based allocation. Install with `pip install psutil`.", "error")
            else:
                try:
                    percent = float(config['MAX_RAM_PERCENT'])
                    if not (0 < percent <= 100):
                        raise ValueError("Percentage must be between 1 and 100.")
                    total_bytes = psutil.virtual_memory().total
                    self._shared_max_bytes = int(total_bytes * (percent / 100.0))
                    source = f"{percent}% of total RAM"
                except Exception as e:
                    self._log_message(f"Could not calculate percentage-based memory: {e}. Falling back to fixed size.", "warning")

        # Priority 3 & 4: Fixed GB from config or a hardcoded default
        if not hasattr(self, '_shared_max_bytes'):
            final_size_gb = config.get('MAX_RAM_GB', 2)  # Default to 2 GB
            source = "config file" if 'MAX_RAM_GB' in config else "hardcoded default"

        # Convert GB to bytes if necessary
        if final_size_gb is not None:
            self._shared_max_bytes = int(final_size_gb * 1_000_000_000)

        self._log_message(f"Max RAM usage set to {self._shared_max_bytes / 1_000_000_000:.2f} GB (source: {source})", "info")

    def _get_logger(self):
        """
        Sets up and returns a dedicated logger for this class to ensure that
        log messages are namespaced and formatted consistently.
        """
        logger = logging.getLogger(f"WorkflowContext_{id(self)}")
        logger.propagate = False
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - [WorkflowContext] %(levelname)s - %(message)s')
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def _log_message(self, message: str, level: str = "info"):
        """Centralized logging method for the handler."""
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(message)

    def _load_receipt(self):
        """Loads the key closet from the receipt JSON."""
        with open(self.receipt_path, 'r') as f:
            self._key_closet = json.load(f)

    def _save_receipt(self):
        """Saves the key closet to the receipt JSON atomically."""
        temp_path = self.receipt_path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(self._key_closet, f)
        os.replace(temp_path, self.receipt_path)

    def save(self, unique_id: str, data: Any, use_ram: bool = True):
        """
        Saves data ephemerally to RAM or disk, using its unique_id as the key.
        This method is resilient, with auto-fallback to disk and safe overwrite handling.
        """
        old_storage_info = self._key_closet.get(unique_id)
        if old_storage_info:
            self._log_message(f"Overwriting existing data for unique_id: '{unique_id}'", "warning")

        try:
            # Serialize data first to get size
            serialized = cloudpickle.dumps(data)
            size = len(serialized)

            if use_ram:
                try:
                    self._save_to_ram(unique_id, serialized, size)
                except (MemoryError, OSError):
                    self._log_message(f"Failed to allocate shared memory for '{unique_id}'. Falling back to disk.", "warning")
                    self._save_to_disk(unique_id, serialized, size)
            else:
                self._save_to_disk(unique_id, serialized, size)

        except Exception as e:
            self._log_message(f"Failed to save data for '{unique_id}'. Original data (if any) is preserved. Error: {e}", "error")
            if old_storage_info:
                self._key_closet[unique_id] = old_storage_info
            raise

        # After successful save, clean up old data if it existed
        if old_storage_info:
            self._cleanup_old_storage(old_storage_info)

        # Save updated receipt
        self._save_receipt()

    def _check_ram_capacity(self, size: int) -> bool:
        """Checks if there's enough capacity in RAM for the new data."""
        current_usage = sum(d['size'] for d in self._key_closet.values() if d['storage_type'] == 'ram')
        return current_usage + size <= self._shared_max_bytes

    def _save_to_ram(self, unique_id: str, serialized: bytes, size: int):
        """Handles saving serialized data to shared memory."""
        if not self._check_ram_capacity(size):
            raise MemoryError("Insufficient RAM capacity")

        shm_name = f"discomfort_shm_{self.run_id}_{uuid.uuid4().hex}"
        shm = shared_memory.SharedMemory(create=True, name=shm_name, size=size)
        shm.buf[0:size] = serialized
        shm.close()

        self._key_closet[unique_id] = {
            "storage_type": "ram",
            "shm_name": shm_name,
            "size": size
        }
        self._log_message(f"Saved '{unique_id}' to RAM (shm_name: {shm_name})", "debug")

    def _save_to_disk(self, unique_id: str, serialized: bytes, size: int):
        """Handles saving serialized data to disk."""
        filepath = os.path.join(self._run_dir, f"{unique_id}{self.SERIALIZATION_EXT}")
        with open(filepath, 'wb') as f:
            f.write(serialized)

        self._key_closet[unique_id] = {
            "storage_type": "disk",
            "path": filepath,
            "size": size
        }
        self._log_message(f"Saved '{unique_id}' to temporary disk file: {filepath}", "debug")

    def _cleanup_old_storage(self, storage_info: Dict):
        """Cleans up old storage for overwritten data."""
        storage_type = storage_info['storage_type']
        if storage_type == "ram":
            shm_name = storage_info['shm_name']
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                shm.close()
                shm.unlink()
                self._log_message(f"Deleted old RAM storage for shm_name: {shm_name}", "debug")
            except FileNotFoundError:
                pass  # Already cleaned
        elif storage_type == "disk":
            path = storage_info['path']
            if os.path.exists(path):
                os.remove(path)
                self._log_message(f"Deleted old disk file: {path}", "debug")

    def load(self, unique_id: str) -> Any:
        """
        Loads data from storage using its unique_id. It automatically determines
        whether to load from RAM or disk based on the key closet's records.
        """
        if unique_id not in self._key_closet:
            raise KeyError(f"No data found for unique_id: '{unique_id}'")
        storage_info = self._key_closet[unique_id]
        storage_type = storage_info["storage_type"]
        self._log_message(f"Loading '{unique_id}' from {storage_type}...", "debug")
        if storage_type == "ram":
            return self._load_from_ram(storage_info)
        else:
            return self._load_from_disk(storage_info)

    def _load_from_ram(self, storage_info: Dict) -> Any:
        """Loads data from shared memory."""
        shm_name = storage_info['shm_name']
        size = storage_info['size']
        shm = shared_memory.SharedMemory(name=shm_name)
        data_bytes = bytes(shm.buf[0:size])
        shm.close()
        return cloudpickle.loads(data_bytes)

    def _load_from_disk(self, storage_info: Dict) -> Any:
        """Loads data from disk."""
        path = storage_info['path']
        with open(path, 'rb') as f:
            return cloudpickle.load(f)

    def export_data(self, unique_id: str, destination_path: str, overwrite: bool = False):
        """
        Makes ephemeral data permanent by moving or copying it to a specified path.
        """
        if unique_id not in self._key_closet:
            raise KeyError(f"Cannot export. No data found for unique_id: '{unique_id}'")
        if os.path.exists(destination_path) and not overwrite:
            raise FileExistsError(f"Destination file exists and overwrite is False: {destination_path}")

        storage_info = self._key_closet[unique_id]
        storage_type = storage_info['storage_type']
        self._log_message(f"Exporting '{unique_id}' from {storage_type} to {destination_path}", "info")
        
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        data = self.load(unique_id)
        with open(destination_path, 'wb') as f:
            cloudpickle.dump(data, f)
        
        del self._key_closet[unique_id]
        self._save_receipt()
        self._log_message(f"Successfully exported '{unique_id}'. It is now persistent.", "info")

    def list_keys(self) -> List[str]:
        """Returns a list of all unique_ids currently stored."""
        return list(self._key_closet.keys())

    def get_usage(self) -> Dict:
        """Reports current RAM and temporary disk usage."""
        ram_usage_bytes = sum(d['size'] for d in self._key_closet.values() if d['storage_type'] == 'ram')
        disk_usage_bytes = sum(d['size'] for d in self._key_closet.values() if d['storage_type'] == 'disk')

        return {
            "ram_usage_mb": round(ram_usage_bytes / 1024**2, 2),
            "ram_capacity_mb": round(self._shared_max_bytes / 1024**2, 2),
            "temp_disk_usage_mb": round(disk_usage_bytes / 1024**2, 2),
            "stored_keys_count": len(self._key_closet)
        }

    def shutdown(self):
        """
        Shuts down all services and cleans up all ephemeral resources for the run.
        """
        self._log_message("Shutting down WorkflowContext and cleaning up ephemeral resources...", "info")
        for unique_id, storage_info in list(self._key_closet.items()):
            storage_type = storage_info['storage_type']
            if storage_type == 'ram':
                shm_name = storage_info['shm_name']
                try:
                    shm = shared_memory.SharedMemory(name=shm_name)
                    shm.close()
                    if self._creator:
                        shm.unlink()
                except FileNotFoundError:
                    pass  # Already unlinked
            elif storage_type == 'disk':
                path = storage_info['path']
                if os.path.exists(path):
                    os.remove(path)
        self._key_closet.clear()
        if os.path.exists(self._run_dir):
            shutil.rmtree(self._run_dir)
        self._log_message("Shutdown complete.", "info")