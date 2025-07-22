import os
import json
import shutil
import tempfile
from typing import Any, Dict, List, Optional
import cloudpickle
import sys
import uuid
import signal
import threading
try:
    import psutil
except ImportError:
    psutil = None
from multiprocessing import shared_memory, resource_tracker

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
      that maps each `unique_id` to its storage location and type. The on-disk
      `receipt.json` is treated as the single source of truth to sync state
      across different process instances.
    - **Ephemeral by Default**: All stored data is temporary. Persistence is an
      explicit action handled by the `export_data()` method.
    - **Resilient Storage**: Features an automatic fallback from RAM to disk if the
      in-memory store runs out of space or allocation fails.
    
    *How this works:*
    At any point in time, there should be two WorkflowContext instances running:
    - The orchestrator instance, which holds the context for *all* runs.
    - The run instance, which holds the context for a single workflow run.

    Workflow contexts are shared between the orchestrator and the run instance via the 
    receipt.json file using the _load_receipt and _save_receipt methods. This is a simple 
    way to ensure that all processes have access to the same context, and it is also a way 
    to ensure that the context is properly cleaned up when the processes exit.
    
    Resource Tracker Note:
    Python's multiprocessing.resource_tracker automatically tracks SharedMemory objects
    to prevent leaks. However, since we manage lifecycle ourselves and allow multiple
    WorkflowContext instances to access the same shared memory, we unregister from
    resource_tracker at every SharedMemory creation point to prevent double-cleanup warnings,
    and we also run cleanup at shutdown to ensure that all resources are released.
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
        # Note: _tracked_shm_names is instance-specific and won't persist across instances
        # We handle resource_tracker cleanup at every SharedMemory access point instead
        self._tracked_shm_names = set()  # Track which SHMs we've created in this instance

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
            config = {} # Ensure config is a dict

        # Set up directories
        self._contexts_dir = os.path.join(tempfile.gettempdir(), config.get('CONTEXTS_DIR_NAME', "contexts"))
        os.makedirs(self._contexts_dir, exist_ok=True)
        self._run_dir = os.path.join(self._contexts_dir, self.run_id)
        self.receipt_path = os.path.join(self._run_dir, "receipt.json")

        if create:
            if os.path.exists(self._run_dir):
                self._log_message(f"Run directory {self._run_dir} already exists. Re-creating.", "warning")
                shutil.rmtree(self._run_dir)
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

        # Register signal handlers if creator AND in the main thread.
        if self._creator and threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
                self._log_message("Signal handlers for graceful shutdown registered.", "debug")
            except Exception as e:
                self._log_message(f"Could not register signal handlers: {e}. This is expected if not in the main thread.", "warning")

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
        log messages are namespaced and formatted consistently, including the method name that calls _log_message.
        """
        import logging
        import inspect

        class CustomFormatter(logging.Formatter):
            def format(self, record):
                frame = inspect.currentframe().f_back  # Start from caller of format (emit)
                while frame and frame.f_code.co_name != '_log_message':
                    frame = frame.f_back
                if frame:
                    # One more step back to get the caller of _log_message
                    caller_frame = frame.f_back
                    record.caller_funcName = caller_frame.f_code.co_name if caller_frame else "<unknown>"
                else:
                    record.caller_funcName = "<unknown>"
                return super().format(record)

        logger = logging.getLogger(f"WorkflowContext_{id(self)}")
        logger.propagate = False
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)
            formatter = CustomFormatter('%(asctime)s - [WorkflowContext] (%(caller_funcName)s) %(levelname)s - %(message)s')
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
        """Loads the key closet from the receipt JSON, ensuring state is fresh."""
        try:
            with open(self.receipt_path, 'r') as f:
                self._key_closet = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self._log_message(f"Could not load receipt file: {e}. Key closet may be out of sync.", "error")
            self._key_closet = {} # Reset to avoid operating on stale data

    def _save_receipt(self):
        """Saves the key closet to the receipt JSON atomically."""
        temp_path = self.receipt_path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(self._key_closet, f, indent=4)
        os.replace(temp_path, self.receipt_path)

    def save(self, unique_id: str, data: Any, use_ram: bool = True):
        """
        Saves data ephemerally, ensuring state is synchronized before the write.
        """
        self._load_receipt() # This ensures we have the latest state before modifying it.
        
        old_storage_info = self._key_closet.get(unique_id)
        if old_storage_info:
            self._log_message(f"Overwriting existing data for unique_id: '{unique_id}'", "warning")
        try:
            self._log_message(f"Serializing data for unique_id: '{unique_id}'", "debug")
            serialized = cloudpickle.dumps(data)
            self._log_message(f"Serialized data for unique_id: '{unique_id}'", "debug")
            size = len(serialized)
            if use_ram:
                try:
                    self._save_to_ram(unique_id, serialized, size)
                except (MemoryError, OSError, ValueError) as e:
                    self._log_message(f"RAM save failed for '{unique_id}': {e}. Falling back to disk.", "warning")
                    self._save_to_disk(unique_id, serialized, size)
            else:
                self._save_to_disk(unique_id, serialized, size)
        except Exception as e:
            self._log_message(f"Failed to save data for '{unique_id}'. Original data (if any) is preserved. Error: {e}", "error", exc_info=True)
            if old_storage_info:
                self._key_closet[unique_id] = old_storage_info # Restore old info on failure
            else:
                self._key_closet.pop(unique_id, None)
            raise
        if old_storage_info:
            self._cleanup_old_storage(old_storage_info)
        self._save_receipt()

    def _check_ram_capacity(self, size: int) -> bool:
        """Checks if there's enough capacity in RAM for the new data."""
        current_usage = sum(d['size'] for d in self._key_closet.values() if d['storage_type'] == 'ram')
        return current_usage + size <= self._shared_max_bytes

    def _save_to_ram(self, unique_id: str, serialized: bytes, size: int):
        """Saves data to RAM, ensuring capacity is sufficient."""
        if not self._check_ram_capacity(size):
            raise MemoryError(f"Insufficient RAM capacity to store '{unique_id}' ({size} bytes).")
        self._log_message(f"Saving data for unique_id: '{unique_id}'", "debug")
        shm_name = f"discomfort_shm_{self.run_id}_{uuid.uuid4().hex}"
        shm = None
        try:
            shm = shared_memory.SharedMemory(create=True, name=shm_name, size=size)
            try: # Attempt to unregister from resource_tracker before unlinking
                resource_tracker.unregister(shm._name, "shared_memory")
            except:
                self._log_message(f"Failed to unregister from resource_tracker for {shm_name} -- probably already unregistered", "warning")
                pass
            self._tracked_shm_names.add(shm_name)
            shm.buf[0:size] = serialized
            shm.close()
            self._key_closet[unique_id] = {"storage_type": "ram", "shm_name": shm_name, "size": size}
            self._log_message(f"Saved '{unique_id}' to RAM (shm_name: {shm_name}, size: {size} bytes)", "debug")
        except Exception as e:
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except:
                    pass
            raise ValueError(f"Shared memory allocation failed for {shm_name}: {e}")

    def _save_to_disk(self, unique_id: str, serialized: bytes, size: int):
        filepath = os.path.join(self._run_dir, f"{unique_id}{self.SERIALIZATION_EXT}")
        self._log_message(f"Saving data for unique_id: '{unique_id}' to disk", "debug")
        with open(filepath, 'wb') as f:
            f.write(serialized)
        self._key_closet[unique_id] = {"storage_type": "disk", "path": filepath, "size": size}
        self._log_message(f"Saved '{unique_id}' to temporary disk file: {filepath}", "debug")

    def _cleanup_old_storage(self, storage_info: Dict):
        """Cleans up old storage for overwritten data."""
        storage_type = storage_info.get('storage_type')
        if storage_type == "ram":
            shm_name = storage_info.get('shm_name')
            if not shm_name:
                return
            try:
                # Attempt to connect to the shared memory segment
                shm = shared_memory.SharedMemory(name=shm_name)
                try:
                    resource_tracker.unregister(shm._name, "shared_memory")
                except:
                    self._log_message(f"Failed to unregister from resource_tracker for {shm_name} -- probably already unregistered", "warning")
                    pass
                try:
                    shm.close()
                    self._log_message(f"Closed old RAM storage (shm_name: {shm_name})", "debug")
                    shm.unlink() # This deletes the shared memory segment
                    self._tracked_shm_names.discard(shm_name)
                    self._log_message(f"Unlinked old RAM storage (shm_name: {shm_name})", "debug")
                except Exception as e:
                    # This is likely already unlinked, so we don't want to raise an error
                    try:
                        shm.close()  # Extra safety to ensure we close even on error
                        self._log_message(f"Closed old RAM storage (shm_name: {shm_name})", "debug")
                    except:
                        self._log_message(f"Failed to close old RAM storage (shm_name: {shm_name}) -- probably already unlinked", "warning")
            except FileNotFoundError:
                # This is not an error; it means the memory was already unlinked
                self._log_message(f"RAM storage (shm_name: {shm_name}) was already unlinked.", "debug")
            except Exception as e:
                self._log_message(f"Error during RAM storage cleanup for {shm_name}: {e}", "warning")
        elif storage_type == "disk":
            path = storage_info.get('path')
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    self._log_message(f"Deleted old ephemeral disk file: {path}", "debug")
                except Exception as e:
                    self._log_message(f"Error deleting ephemeral disk file {path}: {e}", "warning")

    def _load_from_ram(self, storage_info: Dict) -> Any:
        """ Loads data from RAM."""
        shm_name = storage_info['shm_name']
        size = storage_info['size']
        shm = shared_memory.SharedMemory(name=shm_name)
        try:
            try: # Attempt to unregister from resource_tracker before unlinking
                resource_tracker.unregister(shm._name, "shared_memory")
            except:
                self._log_message(f"Failed to unregister from resource_tracker for {shm_name} -- probably already unregistered", "warning")
                pass
            data_bytes = bytes(shm.buf[0:size])
            return cloudpickle.loads(data_bytes)
        finally:
            shm.close()

    def _load_from_disk(self, storage_info: Dict) -> Any:
        path = storage_info['path']
        with open(path, 'rb') as f:
            return cloudpickle.load(f)

    def load(self, unique_id: str) -> Any:
        """
        Loads data from storage, ensuring state is synchronized before the read.
        """
        self._load_receipt() # This ensures we have the latest state before reading.
        
        storage_info = self._key_closet.get(unique_id)
        if not storage_info:
            raise KeyError(f"No data found for unique_id: '{unique_id}' in run '{self.run_id}'")
        storage_type = storage_info["storage_type"]
        self._log_message(f"Loading '{unique_id}' from {storage_type}...", "debug")
        if storage_type == "ram":
            return self._load_from_ram(storage_info)
        else:
            return self._load_from_disk(storage_info)

    def export_data(self, unique_id: str, destination_path: str, overwrite: bool = False):
        """Makes ephemeral data permanent, ensuring state is synchronized first."""
        self._load_receipt() # **FIX**: Ensure we have the latest state.
        
        if unique_id not in self._key_closet:
            raise KeyError(f"Cannot export. No data found for unique_id: '{unique_id}'")
        if os.path.exists(destination_path) and not overwrite:
            raise FileExistsError(f"Destination file exists and overwrite is False: {destination_path}")
        storage_info = self._key_closet[unique_id]
        self._log_message(f"Exporting '{unique_id}' from {storage_info['storage_type']} to {destination_path}", "info")
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        data = self.load(unique_id) # Use the synchronized load method
        self._log_message(f"Saving data for unique_id: '{unique_id}' to disk", "debug")
        with open(destination_path, 'wb') as f:
            cloudpickle.dump(data, f)
        # Remove from ephemeral tracking
        self._cleanup_old_storage(self._key_closet[unique_id])
        del self._key_closet[unique_id]
        self._save_receipt()
        self._log_message(f"Successfully exported '{unique_id}'. It is now persistent.", "info")

    def list_keys(self) -> List[str]:
        """Returns a list of all unique_ids, ensuring state is synchronized."""
        self._load_receipt() # This ensures we have the latest state before listing.
        return list(self._key_closet.keys())

    def get_usage(self) -> Dict:
        """Reports current usage, ensuring state is synchronized."""
        self._load_receipt() # This ensures we have the latest state before reporting.
        
        ram_usage_bytes = sum(d['size'] for d in self._key_closet.values() if d['storage_type'] == 'ram')
        disk_usage_bytes = sum(d['size'] for d in self._key_closet.values() if d['storage_type'] == 'disk')
        return {
            "ram_usage_bytes": ram_usage_bytes, "ram_capacity_bytes": self._shared_max_bytes,
            "ram_usage_gb": round(ram_usage_bytes / 10**9, 3), "ram_capacity_gb": round(self._shared_max_bytes / 10**9, 3),
            "temp_disk_usage_bytes": disk_usage_bytes, "temp_disk_usage_mb": round(disk_usage_bytes / 1024**2, 2),
            "stored_keys_count": len(self._key_closet)
        }

    def shutdown(self):
        """Shuts down all services and cleans up all ephemeral resources for the run."""
        self._log_message("Shutting down WorkflowContext and cleaning up ephemeral resources...", "info")
        # Iterate over a copy of the items, as we may modify the dictionary
        for unique_id, storage_info in list(self._key_closet.items()):
            self._cleanup_old_storage(storage_info)
        self._key_closet.clear()
        # The creator of the context is responsible for deleting the entire run directory
        if self._creator and os.path.exists(self._run_dir):
            try:
                shutil.rmtree(self._run_dir)
                self._log_message(f"Removed temporary run directory: {self._run_dir}", "debug")
            except Exception as e:
                self._log_message(f"Failed to remove run directory {self._run_dir}: {e}", "error")
        self._log_message("Shutdown complete.", "info")