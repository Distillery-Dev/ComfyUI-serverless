# discomfort/data_handler.py

import json
import logging
import os
import subprocess
import shutil
import time
import tempfile
from typing import Any, Dict, List, Optional
import cloudpickle
import pyarrow
import pyarrow.plasma

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
      in-memory store runs out of space.
    """
    SERIALIZATION_EXT = ".pkl"

    def __init__(self, config_path: Optional[str] = None, plasma_store_size_gb: Optional[float] = None):
        """
        Initializes the data handler for a new run.

        Args:
            config_path (str, optional): Path to a custom JSON config file.
                                         If None, uses the default.
            plasma_store_size_gb (float, optional): A direct override for the Plasma
                                                    store size in gigabytes. This has the
                                                    highest priority.
        """
        self.logger = self._get_logger()
        self._key_closet: Dict[str, Dict] = {}

        # --- Configure Storage Settings (including dynamic memory) ---
        self._configure_storage(config_path, plasma_store_size_gb)

        # --- Initialize Ephemeral Storage ---
        self._temp_dir = tempfile.mkdtemp(prefix="discomfort_run_")
        self._log_message(f"Created ephemeral disk storage at: {self._temp_dir}", "info")

        self._plasma_process = None
        self._plasma_client = None
        self._initialize_plasma_store()

        self._log_message("WorkflowContext is ready for this run.", "info")

    def __enter__(self):
        """Allows the class to be used as a context manager, returning itself on entry."""
        self._log_message("Entering context manager.", "debug")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures shutdown is called automatically when exiting a 'with' block."""
        self._log_message("Exiting context manager, shutting down...", "debug")
        self.shutdown()

    def _configure_storage(self, config_path: Optional[str], plasma_store_size_gb_override: Optional[float]):
        """
        Loads configuration and determines the final Plasma store size based on a
        priority system: Direct Override > Percentage Config > Fixed Config > Default.
        """
        # Load configuration from file, or use an empty dict if not found
        config = {}
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'data_handler.json')
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self._log_message(f"Loaded configuration from {config_path}", "debug")
        except FileNotFoundError:
            self._log_message(f"Configuration file not found. Using default settings.", "warning")

        source = ""
        final_size_gb = None

        # Priority 1: Direct override from the constructor (your idea #2)
        if plasma_store_size_gb_override is not None:
            final_size_gb = plasma_store_size_gb_override
            source = "direct override"

        # Priority 2: Percentage of free memory from config (your idea #1)
        elif 'PLASMA_STORE_SIZE_PERCENT_FREE' in config:
            try:
                import psutil
                percent = float(config['PLASMA_STORE_SIZE_PERCENT_FREE'])
                if not (0 < percent <= 100):
                    raise ValueError("Percentage must be between 1 and 100.")
                
                available_bytes = psutil.virtual_memory().available
                # Calculate the size in bytes based on the percentage of free memory
                self._plasma_size_bytes = int(available_bytes * (percent / 100.0))
                source = f"{percent}% of available RAM"
            except ImportError:
                self._log_message("`psutil` is not installed. Cannot use percentage-based allocation. Please run `pip install psutil`.", "error")
                # Fall through to the next priority
            except Exception as e:
                self._log_message(f"Could not calculate percentage-based memory: {e}. Falling back to fixed size.", "warning")
                # Fall through
        
        # Priority 3 & 4: Fixed GB from config or a hardcoded default
        # This block runs if the above methods did not set _plasma_size_bytes.
        if not hasattr(self, '_plasma_size_bytes'):
            final_size_gb = config.get('PLASMA_STORE_SIZE_GB', 2) # Default to 2 GB
            source = "config file" if 'PLASMA_STORE_SIZE_GB' in config else "hardcoded default"
        
        # Convert GB to bytes if the size was determined in GB
        if final_size_gb is not None:
            self._plasma_size_bytes = int(final_size_gb * 1_000_000_000)

        self._log_message(f"Plasma store size set to {self._plasma_size_bytes / 1_000_000_000:.2f} GB (source: {source})", "info")
        
        # Load other configuration settings
        self._startup_timeout = config.get('PLASMA_STARTUP_TIMEOUT_SECONDS', 10)
        self._socket_dir_name = config.get('PLASMA_SOCKET_DIR_NAME', 'discomfort_plasma_store')

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

    def _initialize_plasma_store(self):
        """Validates environment, starts the Plasma store, and connects the client."""
        # Environment validation: Check if `plasma_store` is in the system's PATH.
        if not shutil.which('plasma_store'):
            raise FileNotFoundError(
                "The 'plasma_store' executable was not found. "
                "Please ensure Apache Arrow (pyarrow) is correctly installed."
            )

        # The socket path includes the process ID to prevent conflicts.
        socket_dir = os.path.join(tempfile.gettempdir(), self._socket_dir_name)
        os.makedirs(socket_dir, exist_ok=True)
        self._plasma_socket_path = os.path.join(socket_dir, f"plasma_{os.getpid()}{self.SERIALIZATION_EXT}")

        command = ['plasma_store', '-m', str(self._plasma_size_bytes), '-s', self._plasma_socket_path]
        self._log_message(f"Launching Plasma store with command: {' '.join(command)}", "debug")
        self._plasma_process = subprocess.Popen(command)
        
        start_time = time.time()
        while not os.path.exists(self._plasma_socket_path):
            if self._plasma_process.poll() is not None:
                raise RuntimeError("Plasma store process terminated unexpectedly during startup.")
            if time.time() - start_time > self._startup_timeout:
                self._plasma_process.kill()
                raise RuntimeError("Plasma store failed to start within the timeout period.")
            time.sleep(0.1)
        
        self._plasma_client = pyarrow.plasma.connect(self._plasma_socket_path)
        self._log_message(f"Plasma store started (PID: {self._plasma_process.pid}) and client connected.", "info")


    def save(self, unique_id: str, data: Any, use_ram: bool = True):
        """
        Saves data ephemerally to RAM or disk, using its unique_id as the key.
        This method is resilient, with auto-fallback to disk and safe overwrite handling.
        """
        old_storage_info = self._key_closet.get(unique_id)
        if old_storage_info:
            self._log_message(f"Overwriting existing data for unique_id: '{unique_id}'", "warning")

        try:
            if use_ram:
                try:
                    # Attempt to save to the Plasma store.
                    self._save_to_ram(unique_id, data)
                except pyarrow.plasma.PlasmaStoreFull:
                    # Fallback to disk if RAM is full.
                    self._log_message(f"Plasma store is full. Falling back to disk for '{unique_id}'", "warning")
                    self._save_to_disk(unique_id, data)
            else:
                # Save directly to disk if use_ram is False.
                self._save_to_disk(unique_id, data)

        except Exception as e:
            self._log_message(f"Failed to save data for '{unique_id}'. Original data (if any) is preserved. Error: {e}", "error")
            if old_storage_info:
                self._key_closet[unique_id] = old_storage_info
            raise

        # After a successful save, clean up the old data object if it existed.
        if old_storage_info:
            if old_storage_info["storage_type"] == "ram":
                self._log_message(f"Deleting old in-RAM object for overwritten unique_id: '{unique_id}'", "debug")
                self._plasma_client.delete([old_storage_info["internal_key"]])
            elif old_storage_info["storage_type"] == "disk":
                self._log_message(f"Deleting old on-disk object for overwritten unique_id: '{unique_id}'", "debug")
                os.remove(old_storage_info["internal_key"])


    def load(self, unique_id: str) -> Any:
        """
        Loads data from storage using its unique_id. It automatically determines
        whether to load from RAM or disk based on the key closet's records.
        """
        if unique_id not in self._key_closet:
            raise KeyError(f"No data found for unique_id: '{unique_id}'")
        storage_info = self._key_closet[unique_id]
        storage_type, internal_key = storage_info["storage_type"], storage_info["internal_key"]
        self._log_message(f"Loading '{unique_id}' from {storage_type}...", "debug")
        return self._load_from_ram(internal_key, self._plasma_client) if storage_type == "ram" else self._load_from_disk(internal_key)

    def export_data(self, unique_id: str, destination_path: str, overwrite: bool = False):
        """
        Makes ephemeral data permanent by moving or copying it to a specified path.
        """
        if unique_id not in self._key_closet:
            raise KeyError(f"Cannot export. No data found for unique_id: '{unique_id}'")
        if os.path.exists(destination_path) and not overwrite:
            raise FileExistsError(f"Destination file exists and overwrite is False: {destination_path}")

        storage_info = self._key_closet[unique_id]
        storage_type, internal_key = storage_info["storage_type"], storage_info["internal_key"]
        self._log_message(f"Exporting '{unique_id}' from {storage_type} to {destination_path}", "info")
        
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        if storage_type == "disk":
            shutil.move(internal_key, destination_path)
        else: # storage_type == "ram"
            data = self._load_from_ram(internal_key, self._plasma_client)
            self._save_to_disk(data, destination_path)
        
        del self._key_closet[unique_id]
        self._log_message(f"Successfully exported '{unique_id}'. It is now persistent.", "info")

    def list_keys(self) -> List[str]:
        """Returns a list of all unique_ids currently stored."""
        return list(self._key_closet.keys())

    def get_usage(self) -> Dict:
        """Reports current RAM and temporary disk usage."""
        # Get RAM usage by summing the size of all objects in the Plasma store.
        plasma_objects = self._plasma_client.list()
        ram_usage_bytes = sum(info['data_size'] for obj_id, info in plasma_objects.items())
        
        # Get temporary disk usage by summing the size of all files in the temp dir.
        disk_usage_bytes = sum(f.stat().st_size for f in os.scandir(self._temp_dir) if f.is_file())

        return {
            "ram_usage_mb": round(ram_usage_bytes / 1024**2, 2),
            "ram_capacity_mb": round(self._plasma_size_bytes / 1024**2, 2),
            "temp_disk_usage_mb": round(disk_usage_bytes / 1024**2, 2),
            "stored_keys_count": len(self._key_closet)
        }


    def shutdown(self):
        """
        Shuts down all services and cleans up all ephemeral resources for the run.
        """
        self._log_message("Shutting down DataHandler and cleaning up ephemeral resources...", "info")
        if self._plasma_client: self._plasma_client.disconnect()
        if self._plasma_process and self._plasma_process.poll() is None:
            self._plasma_process.terminate()
            try: self._plasma_process.wait(timeout=5)
            except subprocess.TimeoutExpired: self._plasma_process.kill()
        if os.path.exists(self._temp_dir): shutil.rmtree(self._temp_dir)
        self._key_closet.clear()
        self._log_message("Shutdown complete.", "info")


    def _save_to_disk(self, unique_id: str, data: Any):
        """
        Handles all logic for saving data to a temporary disk file.
        This includes creating the file path, serializing the data,
        and updating the key closet.
        """
        # This internal method now fully encapsulates the process of saving to disk.
        filepath = os.path.join(self._temp_dir, f"{unique_id}{self.SERIALIZATION_EXT}")
        with open(filepath, 'wb') as f:
            cloudpickle.dump(data, f)
        
        # It is now responsible for updating the key closet for its storage type.
        self._key_closet[unique_id] = {
            "storage_type": "disk",
            "internal_key": filepath
        }
        self._log_message(f"Saved '{unique_id}' to temporary disk file: {filepath}", "debug")


    def _save_to_ram(self, unique_id: str, data: Any):
        """
        Handles all logic for saving data to the in-memory Plasma store.
        This includes serializing the data, putting it in the store, and
        updating the key closet with the resulting ObjectID.
        """
        # This internal method now fully encapsulates the process of saving to RAM.
        object_id = pyarrow.plasma.ObjectID.from_random()
        serialized_buffer = pyarrow.serialize(data).to_buffer()
        self._plasma_client.put(serialized_buffer, object_id)

        # It is now responsible for updating the key closet for its storage type.
        self._key_closet[unique_id] = {
            "storage_type": "ram",
            "internal_key": object_id
        }
        self._log_message(f"Saved '{unique_id}' to RAM.", "debug")

    def _load_from_disk(self, filepath: str) -> Any:
        """Reads a file and deserializes with cloudpickle."""
        with open(filepath, 'rb') as f: return cloudpickle.load(f)

    def _load_from_ram(self, object_id: pyarrow.plasma.ObjectID, plasma_client) -> Any:
        """Gets the object buffer from the Plasma store and deserializes with PyArrow."""
        [buffer] = plasma_client.get_buffers([object_id])
        return pyarrow.deserialize(buffer)