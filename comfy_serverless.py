import uuid
import json
import urllib.request
import urllib.parse
import websocket  # Using the synchronous 'websocket-client' library as required by ComfyUI.
import requests
import time
import os
import subprocess
import sys
import shutil
from typing import Dict, Any
import asyncio  # Imported to run synchronous (blocking) code in a separate thread.
from playwright.async_api import async_playwright, Page, Browser  # Using Playwright's Async API to work with ComfyUI's async environment.

class ComfyConnector:
    """
    An ASYNCHRONOUS connector to start and interact with a ComfyUI instance in a serverless-like environment.
    This class uses a singleton pattern to ensure that only one instance of the ComfyUI server
    is managed per process, which is crucial for resource management.
    """
    _instance = None
    _process = None

    # Class attributes to hold the async Playwright objects.
    # These are stored on the class itself to be shared by the singleton instance.
    playwright = None
    _browser = None
    _page = None
    _init_lock = None  # An asyncio Lock to prevent race conditions during the first asynchronous initialization.

    def __new__(cls, *args, **kwargs):
        """
        The __new__ method is part of the singleton pattern. It ensures that only one
        instance of ComfyConnector is ever created.
        """
        if cls._instance is None:
            cls._instance = super(ComfyConnector, cls).__new__(cls)
            cls._instance.initialized = False  # The 'initialized' flag tracks if the expensive setup has been run.
            # The async lock is created here, once, to manage the first-time setup.
            if cls._init_lock is None:
                cls._init_lock = asyncio.Lock()
        return cls._instance

    def __init__(self, config_path=None):
        """
        Initializes the ComfyConnector singleton with basic configuration.
        This method is now lightweight and synchronous, as __init__ cannot be async.
        The actual server and browser startup is deferred to the async _ensure_initialized method.
        """
        # The 'config_loaded' flag ensures this part runs only once.
        if hasattr(self, 'config_loaded'):
            return

        module_dir = os.path.abspath(os.path.dirname(__file__))
        if config_path is None:
            config_path = os.path.join(module_dir, 'comfy_serverless.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # --- Configuration Parameters ---
        # These are all simple, synchronous assignments of config values.
        self.test_payload_file = os.path.join(module_dir, config['TEST_PAYLOAD_FILE'])
        self.api_command_line = config['API_COMMAND_LINE']
        self.api_working_directory = config.get('API_WORKING_DIRECTORY', '.')
        self.api_url = config['API_URL']
        self.initial_port = int(config['INITIAL_PORT'])
        self.max_start_attempts = int(config['MAX_COMFY_START_ATTEMPTS'])
        self.start_attempts_sleep = float(config['COMFY_START_ATTEMPTS_SLEEP'])
        self.comfyui_path = config.get('COMFYUI_PATH', './ComfyUI')
        self.ephemeral_files = []
        self.client_id = str(uuid.uuid4())
        self.ws = websocket.WebSocket()  # Instantiate the synchronous websocket object.
        self.config_loaded = True

    async def _ensure_initialized(self):
        """
        (Internal) Ensures the server and browser are started. This async method contains all
        the heavy setup logic and is designed to run only once, safely.
        """
        # The async lock prevents multiple concurrent calls from trying to initialize at the same time.
        async with self._init_lock:
            if self.initialized:
                return

            print("Initializing ComfyConnector for the first time...")
            self.urlport = self._find_available_port()
            self.server_address = f"http://{self.api_url}:{self.urlport}"
            self.ws_address = f"ws://{self.api_url}:{self.urlport}/ws?clientId={self.client_id}"

            # --- PLAYWRIGHT INITIALIZATION & AUTO-INSTALLER (ASYNC VERSION) ---
            print("Initializing headless browser with Async Playwright...")
            ComfyConnector.playwright = await async_playwright().start()
            try:
                ComfyConnector._browser = await ComfyConnector.playwright.chromium.launch(headless=True)
                print("Playwright and headless Chromium initialized successfully.")
            except Exception as e:
                if "Executable doesn't exist" in str(e) or "missing dependencies" in str(e):
                    print("\n--- ONE-TIME SETUP: PLAYWRIGHT BROWSER NOT FOUND ---")
                    print("Attempting to install Chromium now. This may take a few minutes...")
                    try:
                        install_command = [sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"]
                        process = await asyncio.create_subprocess_exec(
                            *install_command,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        stdout, stderr = await process.communicate()

                        if process.returncode != 0:
                            raise RuntimeError(f"Playwright install failed: {stderr.decode()}")
                        
                        print("--- INSTALLATION COMPLETE ---")
                        ComfyConnector._browser = await ComfyConnector.playwright.chromium.launch(headless=True)
                        print("Playwright and headless Chromium re-initialized successfully.")
                    except Exception as install_exc:
                        raise RuntimeError(f"Failed to automatically install Playwright's browser: {install_exc}") from install_exc
                else:
                    raise
            
            ComfyConnector._page = await ComfyConnector._browser.new_page()
            # --- END OF PLAYWRIGHT LOGIC ---

            # Start the ComfyUI API process.
            await self._start_api()

            # Run a full test to ensure the server is fully operational before proceeding.
            if not await self._test_server():
                await self.kill_api()  # If the test fails, perform a full cleanup.
                raise RuntimeError("Server failed to pass the test workflow execution.")

            self.initialized = True
            print("ComfyConnector initialization complete.")

    def _find_available_port(self):
        """
        (Internal) Finds an available network port. This method can remain synchronous
        as it performs fast, local checks.
        """
        port = self.initial_port
        while True:
            try:
                # A quick connection attempt to see if the port is occupied.
                with requests.get(f'http://{self.api_url}:{port}', timeout=0.1) as response:
                    port += 1  # If connection succeeds, port is busy.
            except requests.ConnectionError:
                return port  # If it fails, port is free.

    async def _start_api(self):
        """
        (Internal, Async) Starts the ComfyUI server as a subprocess and waits until it is fully responsive.
        """
        if await self._is_api_running():
            print("_start_api: API is already running.")
            return

        if self._process is None or self._process.poll() is not None:
            api_command_line = self.api_command_line.split() + [f"--port", str(self.urlport)]
            self._process = subprocess.Popen(api_command_line, cwd=self.api_working_directory)
            print(f"_start_api: API process started with PID: {self._process.pid}")

            attempts = 0
            while not await self._is_api_running():
                if self._process.poll() is not None:
                    raise RuntimeError("API process terminated unexpectedly during startup.")
                if attempts >= self.max_start_attempts:
                    await self.kill_api()
                    raise RuntimeError(f"API startup failed after {attempts} attempts.")
                await asyncio.sleep(self.start_attempts_sleep)  # Use asyncio.sleep in an async method.
                attempts += 1
            print(f"_start_api: API is responsive on port {self.urlport} (PID: {self._process.pid}).")

    async def _get_prompt_from_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        (Internal, Async) Converts a workflow JSON into an API-ready prompt by using the
        headless browser to execute ComfyUI's internal JavaScript functions.
        """
        await self._ensure_initialized()  # Make sure the browser page is ready.

        # Navigate to the page and wait for a key UI element to ensure the app is loaded.
        await ComfyConnector._page.goto(self.server_address)
        await ComfyConnector._page.wait_for_selector('#graph-canvas', timeout=20000)
        
        # Execute JavaScript in the page context to transform the workflow.
        await ComfyConnector._page.evaluate("app.loadGraphData(arguments[0])", workflow)
        prompt = await ComfyConnector._page.evaluate("""
            () => { 
                return (async () => { 
                    // We must return the result of the async graphToPrompt function.
                    const result = await app.graphToPrompt(); 
                    return result.output;
                })(); 
            }
        """)
        print("_get_prompt_from_workflow: Successfully converted workflow to prompt.")
        return prompt

    async def _test_server(self):
        """
        (Internal, Async) Tests if the server is fully operational by running a complete test workflow.
        """
        try:
            with open(self.test_payload_file, 'r') as f:
                test_workflow = json.load(f)
            test_prompt = await self._get_prompt_from_workflow(test_workflow)
            history = await self.run_workflow(test_prompt, use_workflow_json=False)
            return bool(history)  # A valid history object indicates success.
        except Exception as e:
            print(f"_test_server: Error during server test - {e}")
            return False

    async def _is_api_running(self):
        """
        (Internal, Async) Performs a full check to see if the server is fully operational,
        including running a test workflow.
        """
        try:
            # A simple, quick HTTP check to see if the server is listening at all.
            # We run the blocking 'requests.get' in a thread to avoid freezing the event loop.
            response = await asyncio.to_thread(requests.get, self.server_address, timeout=1)
            if response.status_code != 200:
                return False
            
            # If the basic check passes, run the comprehensive test workflow.
            return await self._test_server()
        except Exception:
            # Any exception here means the server is not ready.
            return False

    async def kill_api(self):
        """(Async) Kills the ComfyUI server process, closes connections, and resets the singleton state."""
        print("kill_api: Initiating shutdown...")
        if self._process and self._process.poll() is None:
            self._process.kill()
        if self.ws and self.ws.connected:
            self.ws.close()
        
        # Await the async closing operations for Playwright.
        if ComfyConnector._browser:
            await ComfyConnector._browser.close()
        if ComfyConnector.playwright:
            await ComfyConnector.playwright.stop()
        
        # Reset all state variables to allow for a clean restart.
        self._process = None
        self.ws = None
        ComfyConnector.playwright = None
        ComfyConnector._browser = None
        ComfyConnector._page = None
        self.initialized = False
        print("kill_api: Cleanup complete.")

    def _get_history(self, prompt_id):
        """(Internal, Sync) Retrieves the execution history for a specific prompt ID."""
        with urllib.request.urlopen(f"{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def _queue_prompt(self, prompt):
        """(Internal, Sync) Submits a workflow prompt to the ComfyUI server for execution."""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"{self.server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    async def run_workflow(self, payload, use_workflow_json=True):
        """
        (Async) Submits a workflow to ComfyUI, waits for it to complete, and returns the history.
        """
        await self._ensure_initialized()  # The first and most critical step.

        try:
            if not self.ws.connected:
                # Run the blocking websocket connect call in a separate thread.
                await asyncio.to_thread(self.ws.connect, self.ws_address)

            if use_workflow_json:
                prompt = await self._get_prompt_from_workflow(payload)
                prompt_id = self._queue_prompt(prompt)['prompt_id']
            else:
                prompt_id = self._queue_prompt(payload)['prompt_id']

            while True:
                # Run the blocking websocket receive call in a thread to avoid freezing the event loop.
                out = await asyncio.to_thread(self.ws.recv)
                if isinstance(out, str):
                    message = json.loads(out)
                    # This message indicates the workflow for our prompt ID has finished.
                    if message['type'] == 'executing' and message['data']['node'] is None and message['data']['prompt_id'] == prompt_id:
                        break
            
            return self._get_history(prompt_id)
        except Exception as e:
            exc_type, _, exc_tb = sys.exc_info()
            print(f"run_workflow - Unhandled error at line {exc_tb.tb_lineno}: {e}")
            return None

    def upload_data(self, source_path, folder_type='input', subfolder=None, overwrite=False, is_ephemeral=False):
        """
        (Sync) Saves a file directly into the ComfyUI directory structure. This method avoids
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

    def _delete_data(self):
        """
        (Internal, Sync) Deletes all files that were marked as ephemeral during the session.
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