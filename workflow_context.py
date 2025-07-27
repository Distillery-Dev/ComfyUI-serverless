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
import filelock
import atexit

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
    - The nested instance, which holds the context for any nested workflow run.

    Workflow contexts are shared between the orchestrator and the nested instance via the 
    receipt.json file using the _load_receipt and _save_receipt methods. This is a simple 
    way to ensure that all processes have access to the same context, and it is also a way 
    to ensure that the context is properly cleaned up when the processes exit.
    
    *Race Condition Handling**
    To prevent race conditions where multiple processes read/write to the receipt.json
    simultaneously, a file-based lock (`receipt.json.lock`) is used. All methods
    that interact with the receipt acquire this lock, ensuring that operations like
    read-modify-write are atomic and serialized, preventing data loss.
    
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
            run_id = uuid.uuid4().hex # The run_id is used to identify the context so that it can be loaded later
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

        # Initialize the file lock. The lock file will be next to the receipt.
        self.lock_path = self.receipt_path + ".lock"
        self.lock = filelock.FileLock(self.lock_path, timeout=10) # 10-second timeout

        if create:
            # Acquire lock before creating directory structure
            with self.lock:
                if os.path.exists(self._run_dir):
                    self._log_message(f"Run directory {self._run_dir} already exists. Re-creating.", "warning")
                    shutil.rmtree(self._run_dir)
                os.makedirs(self._run_dir, exist_ok=False)
                self._save_receipt()  # Save empty receipt while holding the lock
            self._log_message(f"Created new context for run_id: {self.run_id}", "info")
        else:
            if not os.path.exists(self.receipt_path):
                raise FileNotFoundError(f"No receipt found for run_id: {self.run_id}")
            # Acquire lock before initial load
            with self.lock:
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
            # Also register with atexit for guaranteed cleanup in subprocesses, regardless of thread.
            if self._creator:
                atexit.register(self.shutdown)

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
        """
        Loads the key closet from the receipt JSON.
        MUST be called within a file lock context.
        """
        try:
            with open(self.receipt_path, 'r') as f:
                self._key_closet = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self._log_message(f"Could not load receipt file: {e}. Key closet may be out of sync.", "error")
            self._key_closet = {} # Reset to avoid operating on stale data

    def _save_receipt(self):
        """
        Saves the key closet to the receipt JSON atomically.
        MUST be called within a file lock context.
        """
        temp_path = self.receipt_path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(self._key_closet, f, indent=4)
        os.replace(temp_path, self.receipt_path)
    
    def save(self, unique_id: str, data: Any, use_ram: bool = True, pass_by: str = "val"):
        """
        Saves data ephemerally in an atomic, thread-safe manner.

        This method acquires a lock to perform a `read-modify-write` cycle on the
        receipt file, preventing race conditions.

        Args:
            unique_id (str): The key for the data.
            data (Any): The data to store (can be a pickled object or a workflow JSON).
            use_ram (bool): Whether to prefer RAM storage.
            pass_by (str): The method of passing, 'val' (by value) or 'ref' (by reference).
        """
        # Acquire lock for the entire duration of the save operation.
        with self.lock:
            self._load_receipt() # This ensures we have the latest state before modifying it.
            
            old_storage_info = self._key_closet.get(unique_id)
            if old_storage_info:
                self._log_message(f"Overwriting existing data for unique_id: '{unique_id}'", "warning")
            try:
                self._log_message(f"Serializing data for unique_id: '{unique_id}' (pass_by: {pass_by})", "debug")
                serialized = cloudpickle.dumps(data)
                self._log_message(f"Serialized data for unique_id: '{unique_id}'", "debug")
                size = len(serialized)

                storage_details = {"size": size, "pass_by": pass_by}

                if use_ram:
                    try:
                        ram_details = self._save_to_ram(unique_id, serialized, size)
                        storage_details.update(ram_details)
                    except (MemoryError, OSError, ValueError) as e:
                        self._log_message(f"RAM save failed for '{unique_id}': {e}. Falling back to disk.", "warning")
                        disk_details = self._save_to_disk(unique_id, serialized, size)
                        storage_details.update(disk_details)
                else:
                    disk_details = self._save_to_disk(unique_id, serialized, size)
                    storage_details.update(disk_details)

                self._key_closet[unique_id] = storage_details

            except Exception as e:
                self._log_message(f"Failed to save data for '{unique_id}'. Original data (if any) is preserved. Error: {e}", "error")
                if old_storage_info:
                    self._key_closet[unique_id] = old_storage_info # Restore old info on failure
                else:
                    self._key_closet.pop(unique_id, None)
                # We are inside the lock, so we must save the potentially restored state.
                self._save_receipt()
                raise

            # If we successfully wrote new data, cleanup the old data it replaced.
            if old_storage_info:
                self._cleanup_old_storage(old_storage_info)
            
            # Finally, save the updated receipt with the new data information.
            self._save_receipt()

    def _check_ram_capacity(self, size: int) -> bool:
        """Checks if there's enough capacity in RAM for the new data."""
        current_usage = sum(d['size'] for d in self._key_closet.values() if d['storage_type'] == 'ram')
        return current_usage + size <= self._shared_max_bytes

    def _save_to_ram(self, unique_id: str, serialized: bytes, size: int) -> Dict[str, Any]:
        """Saves data to RAM and returns the storage details dictionary."""
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
            self._log_message(f"Saved '{unique_id}' to RAM (shm_name: {shm_name}, size: {size} bytes)", "debug")
            return {"storage_type": "ram", "shm_name": shm_name}
        except Exception as e:
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except:
                    pass
            raise ValueError(f"Shared memory allocation failed for {shm_name}: {e}")

    def _save_to_disk(self, unique_id: str, serialized: bytes, size: int) -> Dict[str, Any]:
        """Saves data to disk and returns the storage details dictionary."""
        filepath = os.path.join(self._run_dir, f"{unique_id}{self.SERIALIZATION_EXT}")
        self._log_message(f"Saving data for unique_id: '{unique_id}' to disk", "debug")
        with open(filepath, 'wb') as f:
            f.write(serialized)
        self._log_message(f"Saved '{unique_id}' to temporary disk file: {filepath}", "debug")
        return {"storage_type": "disk", "path": filepath}

    def _cleanup_old_storage(self, storage_info: Dict):
            """Cleans up old storage for overwritten data."""
            storage_type = storage_info.get('storage_type')
            if storage_type == "ram":
                shm_name = storage_info.get('shm_name')
                if not shm_name:
                    return
                try:
                    # Connect to the shared memory segment and immediately unlink it.
                    shm = shared_memory.SharedMemory(name=shm_name)
                    shm.close()
                    shm.unlink() # This deletes the resource.
                    self._log_message(f"Unlinked old RAM storage (shm_name: {shm_name})", "debug")
                except FileNotFoundError:
                    # The resource was already gone, which is not an error.
                    self._log_message(f"RAM storage (shm_name: {shm_name}) was already unlinked.", "debug")
                except Exception as e:
                    self._log_message(f"Error during RAM storage unlink for {shm_name}: {e}", "warning")

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
        Loads data from storage in a thread-safe manner.
        """
        # Acquire lock to ensure we read a consistent state.
        with self.lock:
            self._load_receipt()
            storage_info = self._key_closet.get(unique_id)

        if not storage_info:
            raise KeyError(f"No data found for unique_id: '{unique_id}' in run '{self.run_id}'")
        
        storage_type = storage_info["storage_type"]
        self._log_message(f"Loading '{unique_id}' from {storage_type}...", "debug")
        
        if storage_type == "ram":
            return self._load_from_ram(storage_info)
        else:
            return self._load_from_disk(storage_info)

    def get_storage_info(self, unique_id: str) -> Optional[Dict]:
        """
        Retrieves the full storage metadata for a given unique_id from the receipt
        in a thread-safe manner.
        """
        with self.lock:
            self._load_receipt()
            return self._key_closet.get(unique_id)

    def export_data(self, unique_id: str, destination_path: str, overwrite: bool = False):
        """Makes ephemeral data permanent in a thread-safe manner."""
        with self.lock:
            self._load_receipt()
            storage_info = self._key_closet.get(unique_id)
            if not storage_info:
                raise KeyError(f"Cannot export. No data found for unique_id: '{unique_id}'")
            if storage_info.get("pass_by") == "ref":
                raise TypeError(f"Cannot export '{unique_id}'. It is a reference and cannot be saved to a file directly.")
            
            # Load the data while still inside the initial lock scope for reading.
            data_to_export = self.load(unique_id)

            # Clean up the old storage and update the receipt
            self._cleanup_old_storage(storage_info)
            del self._key_closet[unique_id]
            self._save_receipt()
        
        # Perform file I/O outside the main lock to minimize lock holding time
        if os.path.exists(destination_path) and not overwrite:
            raise FileExistsError(f"Destination file exists and overwrite is False: {destination_path}")
        
        self._log_message(f"Exporting '{unique_id}' from memory to {destination_path}", "info")
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        with open(destination_path, 'wb') as f:
            cloudpickle.dump(data_to_export, f)
            
        self._log_message(f"Successfully exported '{unique_id}'. It is now persistent.", "info")


    def list_keys(self) -> List[str]:
        """Returns a list of all unique_ids in a thread-safe manner."""
        with self.lock:
            self._load_receipt()
            return list(self._key_closet.keys())

    def get_usage(self) -> Dict:
        """Reports current usage in a thread-safe manner."""
        with self.lock:
            self._load_receipt()
            ram_usage_bytes = sum(d['size'] for d in self._key_closet.values() if d['storage_type'] == 'ram')
            disk_usage_bytes = sum(d['size'] for d in self._key_closet.values() if d['storage_type'] == 'disk')
        
        return {
            "ram_usage_bytes": ram_usage_bytes, "ram_capacity_bytes": self._shared_max_bytes,
            "ram_usage_gb": round(ram_usage_bytes / 10**9, 3), "ram_capacity_gb": round(self._shared_max_bytes / 10**9, 3),
            "temp_disk_usage_bytes": disk_usage_bytes, "temp_disk_usage_mb": round(disk_usage_bytes / 1024**2, 2),
            "stored_keys_count": len(self._key_closet) # Note: this count is from the locked read
        }

    def shutdown(self):
        """Shuts down all services and cleans up all ephemeral resources for the run."""
        self._log_message("Shutting down WorkflowContext and cleaning up ephemeral resources...", "info")
        
        # Acquire the lock to safely clean up all resources.
        with self.lock:
            self._load_receipt() # Load one last time to get all storage info.
            for unique_id, storage_info in list(self._key_closet.items()):
                self._cleanup_old_storage(storage_info)
            self._key_closet.clear()

            # The creator of the context is responsible for deleting the entire run directory.
            # This is now done inside the lock to prevent another process from trying
            # to access the directory while it's being deleted.
            if self._creator and os.path.exists(self._run_dir):
                try:
                    shutil.rmtree(self._run_dir)
                    self._log_message(f"Removed temporary run directory: {self._run_dir}", "debug")
                except Exception as e:
                    self._log_message(f"Failed to remove run directory {self._run_dir}: {e}", "error")
        
        self._log_message("Shutdown complete.", "info")

