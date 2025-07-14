import uuid
import json
import urllib.request
import urllib.parse
import websocket # note: websocket-client (https://github.com/websocket-client/websocket-client)
import requests
import time
import os
import subprocess
import sys
import shutil

class ComfyConnector:
    """
    A connector to start and interact with a ComfyUI instance in a serverless-like environment.
    This class uses a singleton pattern to ensure that only one instance of the ComfyUI server
    is managed per process.
    """
    _instance = None
    _process = None

    def __new__(cls, *args, **kwargs):
        # Singleton pattern: ensures only one instance of ComfyConnector exists.
        if cls._instance is None:
            cls._instance = super(ComfyConnector, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path='comfy_connector.json'):
        """
        Initializes the ComfyConnector singleton. If it's the first time, it loads configuration,
        finds an available port, starts the ComfyUI server, and establishes a connection.
        """
        # The 'initialized' check ensures this heavy setup runs only once.
        if not hasattr(self, 'initialized'):
            # Load configuration from the specified JSON file.
            with open(config_path, 'r') as f:
                config = json.load(f)

            # --- Configuration Parameters ---
            # Command to start the ComfyUI server (e.g., "python3 main.py").
            self.api_command_line = config['API_COMMAND_LINE']
            # The directory from which to run the API command. Can be relative or absolute.
            self.api_working_directory = config.get('API_WORKING_DIRECTORY', '.') # Defaults to current dir
            # URL of the API server, without the port.
            self.api_url = config['API_URL']
            # The initial port to try for the server.
            self.initial_port = int(config['INITIAL_PORT'])
            # Path to the JSON file containing the test workflow.
            self.test_payload_file = config['TEST_PAYLOAD_FILE']
            # Max attempts to check if the ComfyUI server has started.
            self.max_start_attempts = int(config['MAX_COMFY_START_ATTEMPTS'])
            # Time in seconds to wait between startup attempts.
            self.start_attempts_sleep = float(config['COMFY_START_ATTEMPTS_SLEEP'])
            # Base path of the ComfyUI installation, used for direct file operations.
            self.comfyui_path = config.get('COMFYUI_PATH', './ComfyUI')

            # List to track files for deletion on shutdown.
            self.ephemeral_files = []

            # Find an available port starting from the initial_port.
            self.urlport = self._find_available_port()
            self.server_address = f"http://{self.api_url}:{self.urlport}"

            # Unique client ID for the WebSocket connection.
            self.client_id = str(uuid.uuid4())
            self.ws_address = f"ws://{self.api_url}:{self.urlport}/ws?clientId={self.client_id}"
            self.ws = websocket.WebSocket()

            # Start the ComfyUI server process.
            self._start_api()
            self.initialized = True

    def _find_available_port(self):
        """
        (Internal) Finds an available network port to start the API server on, beginning
        from the INITIAL_PORT specified in the config.
        """
        port = self.initial_port
        while True:
            try:
                # Attempt to connect to the port. If it fails, the port is likely free.
                with requests.get(f'http://{self.api_url}:{port}', timeout=0.1) as response:
                    # If we get a response, the port is in use. Increment and try the next one.
                    port += 1
            except requests.ConnectionError:
                # No server responded, so the port is available.
                return port

    def _start_api(self):
        """
        (Internal) Starts the ComfyUI server as a subprocess and waits until it is responsive.
        """
        if self._is_api_running():
            print("_start_api: API is already running.")
            return

        # Check if the process handle exists and if the process is still running.
        if self._process is None or self._process.poll() is not None:
            # Append the chosen port to the command line arguments.
            api_command_line = self.api_command_line.split() + [f"--port", str(self.urlport)]
            print(f"_start_api: Starting ComfyUI with command: {' '.join(api_command_line)}")
            print(f"_start_api: Working Directory: {os.path.abspath(self.api_working_directory)}")

            # Execute the command from the specified working directory.
            self._process = subprocess.Popen(
                api_command_line,
                cwd=self.api_working_directory
            )
            print(f"_start_api: API process started with PID: {self._process.pid}")

            # Wait for the server to become fully operational.
            attempts = 0
            while not self._is_api_running():
                if self._process.poll() is not None:
                    raise RuntimeError("API process terminated unexpectedly.")
                if attempts >= self.max_start_attempts:
                    self.kill_api() # Clean up the failed process.
                    raise RuntimeError(f"API startup failed after {attempts} attempts.")
                time.sleep(self.start_attempts_sleep)
                attempts += 1
            print(f"_start_api: API is responsive on port {self.urlport} (PID: {self._process.pid}).")

    def _is_api_running(self):
        """
        (Internal) Checks if the ComfyUI server is running and fully operational by executing a test workflow.
        This is more reliable than just checking for a 200 status code.
        """
        print(f"_is_api_running: Checking server at {self.server_address}...")
        try:
            # 1. Check for a basic web server response.
            response = requests.get(self.server_address, timeout=1)
            if response.status_code != 200:
                return False
            print("_is_api_running: Web server is responding. Connecting WebSocket...")

            # 2. Check if the WebSocket is connectable.
            if not self.ws.connected:
                self.ws.connect(self.ws_address)
            print("_is_api_running: WebSocket connected. Running test workflow...")

            # 3. Load the test workflow from the file specified in the config.
            with open(self.test_payload_file, 'r') as f:
                test_payload = json.load(f)

            # 4. Run the workflow and check for a valid history object.
            #    The test_server.json workflow inverts test_img.jpg.
            history = self.run_workflow(test_payload)
            if history and isinstance(history, dict) and len(history) > 0:
                print("_is_api_running: Test workflow executed successfully.")
                return True
            else:
                print("_is_api_running: Test workflow failed to return a valid history.")
                return False

        except Exception as e:
            # Any exception during the check means the API is not ready.
            print(f"_is_api_running: API not ready or encountered an error: {e}")
            if self.ws.connected:
                self.ws.close()
            return False

    def _delete_data(self):
        """
        (Internal) Deletes all files that were marked as ephemeral during the session.
        This is called during the shutdown process.
        """
        print("_delete_data: Cleaning up ephemeral files...")
        for fpath in self.ephemeral_files:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
                    print(f"_delete_data: Deleted '{fpath}'")
                else:
                    print(f"_delete_data: Warning: File '{fpath}' not found for deletion.")
            except Exception as e:
                print(f"_delete_data: Error deleting file '{fpath}': {e}")
        self.ephemeral_files.clear() # Clear the list after attempting deletion

    def kill_api(self):
        """
        Kills the ComfyUI server process, closes the WebSocket, cleans up ephemeral files,
        and resets the singleton state. This allows for a clean restart if needed.
        """
        print("kill_api: Initiating shutdown...")
        try:
            if self._process is not None and self._process.poll() is None:
                self._process.kill()
                self._process.wait(timeout=5) # Wait for the process to terminate.
                print(f"kill_api: API process {self._process.pid} killed.")
            if self.ws and self.ws.connected:
                self.ws.close()
                print("kill_api: WebSocket connection closed.")
        except Exception as e:
            print(f"kill_api: Warning during shutdown: {e}")
        finally:
            # Clean up any ephemeral files created during the session.
            self._delete_data()
            # Reset all instance attributes to allow for re-initialization.
            self._process = None
            self.ws = None
            self.initialized = False
            ComfyConnector._instance = None
            print("kill_api: Cleanup complete. Singleton instance has been reset.")

    def _get_history(self, prompt_id):
        """
        (Internal) Retrieves the execution history for a specific prompt ID. The history contains
        detailed information about the outputs of each node in the workflow.
        """
        with urllib.request.urlopen(f"{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def _queue_prompt(self, prompt):
        """
        (Internal) Submits a workflow prompt to the ComfyUI server for execution.
        """
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"{self.server_address}/prompt", data=data, headers={'Content-Type': 'application/json'})
        return json.loads(urllib.request.urlopen(req).read())

    def run_workflow(self, payload):
        """
        Submits a workflow to ComfyUI, waits for it to complete, and returns the full
        execution history. The history contains all generated outputs (images, etc.)
        and other execution data.
        """
        try:
            if not self.ws.connected:
                print("run_workflow: WebSocket is not connected. Reconnecting...")
                self.ws.connect(self.ws_address)

            # Queue the prompt and get its ID.
            prompt_id = self._queue_prompt(payload)['prompt_id']
            print(f"run_workflow: Prompt queued with ID: {prompt_id}")

            # Wait for the execution to finish.
            while True:
                out = self.ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    # The 'executing' message with a null node indicates the end of the queue.
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            print(f"run_workflow: Execution finished for prompt ID: {prompt_id}")
                            break
                # Other messages (like binary previews) are ignored.

            # Retrieve and return the complete history for the executed prompt.
            return self._get_history(prompt_id)

        except Exception as e:
            exc_type, _, exc_tb = sys.exc_info()
            print(f"run_workflow - Unhandled error at line {exc_tb.tb_lineno} in {exc_tb.tb_frame.f_code.co_filename}: {e}")
            return None

    def upload_data(self, source_path, folder_type='input', subfolder=None, overwrite=False, is_ephemeral=False):
        """
        Saves a file directly into the ComfyUI directory structure. This method avoids
        API requests and performs a direct file copy. It is designed to handle various
        data types like images, videos, audio, and model files.

        Args:
            source_path (str): The local path of the file to be saved.
            folder_type (str): The primary destination folder within the ComfyUI directory.
                - 'input', 'output', 'temp'
                - 'models' for model files.
            subfolder (str, optional): A specific sub-directory. For `folder_type='models'`,
                this is required and specifies the model type (e.g., 'checkpoints',
                'loras', 'vae', 'controlnet', 'gguf', 'upscale_models').
            overwrite (bool): If True, overwrites the destination file if it exists.
                              Defaults to False.
            is_ephemeral (bool): If True, the file will be deleted automatically on shutdown,
                                 unless it overwrites a pre-existing file.

        Returns:
            str: The full path to the newly saved file.
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file does not exist: {source_path}")

        # Determine the base directory for the upload.
        if folder_type == 'models':
            if not subfolder:
                raise ValueError("The 'subfolder' argument is required when folder_type is 'models'.")
            base_dir = os.path.join(self.comfyui_path, 'models', subfolder)
        else:
            base_dir = os.path.join(self.comfyui_path, folder_type)
            if subfolder:
                base_dir = os.path.join(base_dir, subfolder)

        # Create the destination directory if it doesn't exist.
        os.makedirs(base_dir, exist_ok=True)

        destination_path = os.path.join(base_dir, os.path.basename(source_path))

        # Check if the file existed before the upload to manage ephemeral logic correctly.
        file_existed_before_upload = os.path.exists(destination_path)

        # Check for overwrite condition before attempting to copy.
        if file_existed_before_upload and not overwrite:
            raise FileExistsError(f"Destination file already exists and overwrite is False: {destination_path}")

        try:
            # Use shutil.copy2 to preserve metadata.
            shutil.copy2(source_path, destination_path)
            print(f"upload_data: Successfully saved '{source_path}' to '{destination_path}'")

            # If the upload is ephemeral, add it to the list for later deletion,
            # but only if an overwrite of an existing file did not occur.
            if is_ephemeral:
                if overwrite and file_existed_before_upload:
                    print(f"upload_data: Warning: '{destination_path}' was overwritten and will not be marked for ephemeral deletion.")
                else:
                    self.ephemeral_files.append(destination_path)
                    print(f"upload_data: '{destination_path}' marked for ephemeral deletion.")

            return destination_path
        except Exception as e:
            print(f"upload_data: Failed to save file '{source_path}' to '{destination_path}'. Error: {e}")
            raise