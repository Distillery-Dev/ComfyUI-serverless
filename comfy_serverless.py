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
import logging  # For structured logging
import aiohttp  # Added for async HTTP requests to reduce blocking
import backoff  # For retry logic

class ComfyConnector:
    """
    An ASYNCHRONOUS connector to start and interact with a ComfyUI instance in a serverless-like environment.
    This class uses a singleton pattern to ensure that only one instance of the ComfyUI server
    is managed per process, which is crucial for resource management.
    Enhanced with explicit state management, validation, and idempotent operations for stability.
    """
    _instance = None  # Singleton instance
    _process = None  # Subprocess for ComfyUI API
    _state = "uninit"  # Lifecycle state: uninit | initializing | ready | error | killed
    playwright = None  # Playwright instance
    _browser = None  # Headless browser
    _page = None  # Browser page for JS execution
    _init_lock = asyncio.Lock()  # Lock to prevent race conditions during initialization
    _state_lock = asyncio.Lock()  # Finer-grained lock for state checks and mutations

    def _log_message(self, message: str, level: str = "info"):
        """Centralized logging method for ComfyConnector."""
        # Set up basic logging if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        
        msg = f"[ComfyConnector] {level.upper()}: {message}"
        levels = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR}
        level_val = levels.get(level.lower(), logging.INFO)
        
        # Use a dedicated logger for ComfyConnector to avoid interfering with ComfyUI's root logger
        comfyconnector_logger = logging.getLogger("ComfyConnector")
        comfyconnector_logger.setLevel(logging.DEBUG) # Ensure all levels can be processed
        
        # Prevent duplicate handlers
        if not comfyconnector_logger.handlers:
            # Add a handler if none exist, e.g., to print to console
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - [ComfyConnector] %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            comfyconnector_logger.addHandler(ch)
            comfyconnector_logger.propagate = False # Stop messages from going to the root logger

        # Get the actual logging function and call it
        log_func = getattr(comfyconnector_logger, level.lower(), comfyconnector_logger.info)
        log_func(message) # Pass the original message without the prefix, as the formatter handles it

    def __new__(cls, *args, **kwargs):
        """
        Ensures that only one instance of ComfyConnector is ever created (Singleton Pattern).
        """
        if cls._instance is None:
            cls._instance = super(ComfyConnector, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path=None):
        """
        Initializes the ComfyConnector singleton with basic configuration.
        This method is lightweight and synchronous. The actual server and browser startup
        is deferred to the async _ensure_initialized method.
        """
        # The 'config_loaded' flag ensures this part runs only once per instance.
        if hasattr(self, 'config_loaded'):
            return

        module_dir = os.path.abspath(os.path.dirname(__file__))
        if config_path is None:
            config_path = os.path.join(module_dir, 'comfy_serverless.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"[ComfyConnector] __init__: Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # --- Configuration Parameters ---
        self.test_payload_file = os.path.join(module_dir, config['TEST_PAYLOAD_FILE'])
        self.api_command_line = config['API_COMMAND_LINE']
        self.api_working_directory = config.get('API_WORKING_DIRECTORY', '.')
        self.api_url = config['API_URL']
        self.initial_port = int(config['INITIAL_PORT'])
        self.max_start_attempts = int(config['MAX_COMFY_START_ATTEMPTS'])
        self.start_attempts_sleep = float(config['COMFY_START_ATTEMPTS_SLEEP'])
        self.comfyui_path = config.get('COMFYUI_PATH', './ComfyUI')
        self.ephemeral_files = []  # List of files to delete on shutdown
        self.client_id = str(uuid.uuid4())  # Unique client ID for WS
        self.ws = websocket.WebSocket()  # Instantiate the synchronous websocket object.
        self._killed = False  # Flag to prevent multiple kill_api calls
        self.config_loaded = True

    @classmethod
    async def create(cls, config_path=None):
        """
        Async factory method to create and initialize the singleton instance.
        This is the designated entry point for creating a ready-to-use ComfyConnector.
        """
        instance = cls(config_path)  # Uses __new__ for singleton
        await instance._ensure_initialized()
        instance._log_message("create: instance called")
        return instance

    async def _validate_resources(self) -> bool:
        """
        (Internal, Async) Validate all underlying resources (process, browser, page, WS, HTTP).
        Returns True if all are operational.
        """
        async with self._state_lock:
            if self._process is None or self._process.poll() is not None:
                self._log_message("_validate_resources: Process is not alive", "warning")
                return False
            if ComfyConnector._browser is None or not ComfyConnector._browser.is_connected():
                self._log_message("_validate_resources: Browser is closed", "warning")
                return False
            if ComfyConnector._page is None or ComfyConnector._page.is_closed():
                self._log_message("_validate_resources: Page is closed", "warning")
                return False
            if not self.ws.connected:
                self._log_message("_validate_resources: WebSocket is not connected", "warning")
                return False
            # Async HTTP check using aiohttp for non-blocking validation
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.server_address, timeout=1) as response:
                        if response.status != 200:
                            self._log_message(f"_validate_resources: Server responded with {response.status}", "warning")
                            return False
            except Exception as e:
                self._log_message(f"_validate_resources: HTTP validation failed: {e}", "warning")
                return False
            return True

    async def _ensure_initialized(self):
        """
        (Internal) Ensures the server and browser are started. This async method contains all
        the heavy setup logic and is designed to run only once, safely. Enhanced to be idempotent,
        with retries and resource validation.
        """
        self._log_message("_ensure_initialized: Ensuring initialization...")
        # The async lock prevents multiple concurrent calls from trying to initialize at the same time.
        async with self._init_lock:
            # First, handle the 'ready' state. If it's ready, validate it.
            # If it's no longer valid, set state to 'error' and fall through to re-initialize.
            if self._state == "ready":
                if await self._validate_resources():
                    self._log_message("_ensure_initialized: Resources are ready and valid. Skipping initialization.", "debug")
                    return
                else:
                    self._log_message("_ensure_initialized: Resources were ready but failed validation. Forcing re-initialization.", "warning")
                    self._state = "error"

            # If we are in the middle of initializing, a re-entrant call (from _test_server)
            # can safely return, as the top-level call is managing the setup.
            if self._state == "initializing":
                return

            # If the state is uninitialized, killed, or has been set to error, proceed with a full setup.
            if self._state in ("error", "killed", "uninit"):
                await self._reset_state()

            self._state = "initializing"
            self.urlport = self._find_available_port()
            self.server_address = f"http://{self.api_url}:{self.urlport}"
            self.ws_address = f"ws://{self.api_url}:{self.urlport}/ws?clientId={self.client_id}"

            # Use backoff for retries on the entire init process
            @backoff.on_exception(backoff.expo, Exception, max_tries=3)
            async def init_with_retry():
                await self._init_playwright()
                await self._start_api()
                await self._refresh_resources()  # Lazy connect WS, etc.

            try:
                await init_with_retry()
                test_passed = await self._test_server()
                if test_passed:
                    self._state = "ready"
                    self._log_message("_ensure_initialized: Initialization complete and test passed", "debug")
                else:
                    raise RuntimeError("[ComfyConnector] _ensure_initialized: Test failed")
            except Exception as e:
                self._log_message(f"_ensure_initialized: Initialization failed after retries: {e}", "error")
                self._state = "error"
                await self.kill_api()
                raise

    async def _refresh_resources(self):
        """
        (Internal, Async) Lazy refresh of resources like WebSocket connection.
        Can be extended for partial recoveries (e.g., relaunch page without full init).
        """
        self._log_message("_refresh_resources: Refreshing resources...", "debug")
        if not self.ws.connected:
            await asyncio.to_thread(self.ws.connect, self.ws_address)
            self._log_message("_refresh_resources: WebSocket reconnected.", "debug")

    async def _reset_state(self):
        """
        (Internal, Async) Resets internal state variables for a clean re-init.
        """
        self._log_message("_reset_state: Resetting connector state.", "debug")
        self._process = None
        # Re-instantiate WebSocket client to clear any old connection state
        self.ws = websocket.WebSocket()
        ComfyConnector.playwright = None
        ComfyConnector._browser = None
        ComfyConnector._page = None
        self.ephemeral_files.clear()
        self._killed = False
        self._state = "uninit"
        self._log_message("_reset_state: state reset", "debug")

    async def _init_playwright(self):
        """
        (Internal, Async) Initialize Playwright and launch headless Chromium browser.
        Handles auto-installation if browser executable is missing.
        """
        self._log_message("_init_playwright: Initializing Playwright and headless browser", "debug")
        ComfyConnector.playwright = await async_playwright().start()
        try:
            ComfyConnector._browser = await ComfyConnector.playwright.chromium.launch(
                                headless=True,  # Temporary for debug; change back to True
                                args=['--enable-webgl', '--disable-gpu']
                                )
            self._log_message("_init_playwright: Playwright and headless Chromium initialized successfully.", "debug")
        except Exception as e:
            if "Executable doesn't exist" in str(e) or "missing dependencies" in str(e):
                self._log_message("_init_playwright: --- ONE-TIME SETUP: PLAYWRIGHT BROWSER NOT FOUND ---", "info")
                self._log_message("_init_playwright: Attempting to install Chromium now. This may take a few minutes...", "info")
                try:
                    install_command = [sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"]
                    process = await asyncio.create_subprocess_exec(
                        *install_command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()

                    if process.returncode != 0:
                        raise RuntimeError(f"[ComfyConnector] _init_playwright: Playwright install failed: {stderr.decode()}")
                    
                    self._log_message("_init_playwright: --- BROWSER INSTALLATION COMPLETE ---", "info")
                    ComfyConnector._browser = await ComfyConnector.playwright.chromium.launch(headless=True)
                    self._log_message("_init_playwright: Playwright and headless Chromium re-initialized successfully.", "debug")
                except Exception as install_exc:
                    raise RuntimeError(f"[ComfyConnector] _init_playwright: Failed to automatically install Playwright's browser: {install_exc}") from install_exc
            else:
                raise
        ComfyConnector._page = await ComfyConnector._browser.new_page()

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
                self._log_message(f"_find_available_port: Server will be hosted in port {port}.", "debug")
                return port  # If it fails, port is free.

    async def _start_api(self):
        """
        (Internal, Async) Starts the ComfyUI server as a subprocess and waits until it is fully responsive.
        Enhanced with max attempts and sleep for stability.
        """
        self._log_message("_start_api: Starting ComfyUI API process...", "debug")
        if await self._is_api_running():
            self._log_message("_start_api: API is already running.", "debug")
            return

        if self._process is None or self._process.poll() is not None:
            api_command_line = self.api_command_line.split() + [f"--port", str(self.urlport)]
            self._process = subprocess.Popen(api_command_line, cwd=self.api_working_directory)
            self._log_message(f"_start_api: API process started with PID: {self._process.pid}", "debug")

            attempts = 0
            while not await self._is_api_running():
                if self._process.poll() is not None:
                    raise RuntimeError("[ComfyConnector] _start_api: API process terminated unexpectedly during startup.")
                if attempts >= self.max_start_attempts:
                    await self.kill_api()
                    raise RuntimeError(f"[ComfyConnector] _start_api: API startup failed after {attempts} attempts.")
                await asyncio.sleep(self.start_attempts_sleep)  # Use asyncio.sleep in an async method.
                attempts += 1
            self._log_message(f"_start_api: API is responsive on port {self.urlport} (PID: {self._process.pid}).", "debug")

    async def _connect_websocket(self):
        """(Internal, Async) Connects the WebSocket client to the server."""
        self._log_message(f"_connect_websocket: Connecting WebSocket to {self.ws_address}...", "debug")
        # Use asyncio.to_thread to run the blocking connect method
        await asyncio.to_thread(self.ws.connect, self.ws_address)
        self._log_message("_connect_websocket: WebSocket connected successfully.", "debug")

    async def _is_api_running(self):
        """
        (Internal, Async) Checks if the server is listening (HTTP response only; no full test to avoid recursion).
        Uses aiohttp for non-blocking check.
        """
        self._log_message("_is_api_running: Checking if server is running...", "debug")
        try:
            # A simple, quick async HTTP check
            async with aiohttp.ClientSession() as session:
                async with session.get(self.server_address, timeout=1) as response:
                    return response.status == 200
            self._log_message("_is_api_running: Server is running.", "debug")
        except Exception:
            self._log_message("_is_api_running: Server is NOT running.", "debug")
            # Any exception means the server is not ready.
            return False

    async def _get_prompt_from_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        (Internal, Async) Converts a workflow JSON into an API-ready prompt by using the
        headless browser to execute ComfyUI's internal JavaScript functions.
        Enhanced with waits for stability in headless mode.
        """
        self._log_message("_get_prompt_from_workflow: Converting workflow to prompt...", "debug")

        if not await self._validate_resources():
            self._log_message("_get_prompt_from_workflow: Resources are not valid. Cannot convert workflow to prompt.", "error")
            raise RuntimeError("[ComfyConnector] _get_prompt_from_workflow: Resources are not valid. Cannot convert workflow to prompt.")
        
        self._log_message("_get_prompt_from_workflow: Converting workflow to prompt via headless browser...", "debug")
        try:
            await ComfyConnector._page.goto(self.server_address) # Go to the server address
            await ComfyConnector._page.wait_for_function("() => typeof window.app !== 'undefined'", timeout=60000) # Wait for the app to be loaded
            await ComfyConnector._page.evaluate("async (wf) => { await window.app.loadGraphData(wf); }", workflow) # Load the workflow
            prompt_data = await ComfyConnector._page.evaluate("async () => { return await window.app.graphToPrompt(); }") # Convert the workflow to a prompt json
            
            self._log_message("_get_prompt_from_workflow: Successfully converted workflow to prompt.", "debug")
            return prompt_data['output']
        except Exception as e:
            self._log_message(f"_get_prompt_from_workflow: Page load or app init failed: {e}", "error")
            raise

    async def _execute_prompt_and_wait(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        (Internal, Async) THE CORE EXECUTION LOGIC.
        This new helper method queues a prompt and waits for its execution to complete.
        It is used by both run_workflow and _test_server to avoid the deadlock.
        """
        try:
            # 1. Queue the prompt using a blocking HTTP request in a separate thread.
            req = urllib.request.Request(f"{self.server_address}/prompt", data=json.dumps({"prompt": prompt, "client_id": self.client_id}).encode('utf-8'))
            response = await asyncio.to_thread(urllib.request.urlopen, req)
            prompt_id = json.loads(response.read())['prompt_id']
            self._log_message(f"_execute_prompt_and_wait: Prompt queued with ID: {prompt_id}", "debug")

            # 2. Wait for the execution to finish by listening on the WebSocket.
            while True:
                # Run the blocking websocket receive call in a thread.
                out = await asyncio.to_thread(self.ws.recv)
                if isinstance(out, str):
                    message = json.loads(out)
                    # The 'executing' message with a null 'node' indicates the prompt is done.
                    if message['type'] == 'executing' and message['data']['node'] is None and message['data']['prompt_id'] == prompt_id:
                        self._log_message(f"_execute_prompt_and_wait: Execution finished for prompt ID: {prompt_id}", "debug")
                        break
                await asyncio.sleep(0.05)  # Small delay to avoid busy-waiting.
            
            # 3. Retrieve the final history for the completed prompt.
            with urllib.request.urlopen(f"{self.server_address}/history/{prompt_id}") as response:
                return json.loads(response.read())
        except Exception as e:
            self._log_message(f"_execute_prompt_and_wait: Error during prompt execution: {e}", "error")
            return None

    async def _test_server(self):
        """
        (Internal, Async) Tests server readiness by running a simple workflow.
        This uses the internal _execute_prompt_and_wait method to avoid deadlocks.
        """
        self._log_message("_test_server: Running server validation test...", "debug")
        try:
            with open(self.test_payload_file, 'r') as f:
                test_workflow = json.load(f)
            
            test_prompt = await self._get_prompt_from_workflow(test_workflow)
            history = await self._execute_prompt_and_wait(test_prompt)
            
            # A valid, non-empty history object indicates a successful test.
            return bool(history)
        except Exception as e:
            self._log_message(f"_test_server: Server test failed: {e}", "error")
            return False

    async def kill_api(self):
        """
        (Async) Kills the server, closes connections, and resets the singleton state. Idempotent.
        """
        async with self._state_lock:
            if self._killed or self._state == "killed":
                self._log_message("kill_api: Already killed, skipping.", "debug")
                return
            self._killed = True
        
        self._log_message("kill_api: Shutting down ComfyUI instance...", "debug")
        if self._process and self._process.poll() is None:
            self._process.kill()
            await asyncio.to_thread(self._process.wait) # Wait for process to terminate
        if self.ws and self.ws.connected:
            self.ws.close()
        if ComfyConnector._browser:
            await ComfyConnector._browser.close()
        if ComfyConnector.playwright:
            await ComfyConnector.playwright.stop()
        
        await self._reset_state()
        self._state = "killed"
        self._log_message("kill_api: Shutdown complete.", "debug")

    def _get_history(self, prompt_id):
        """(Internal, Sync) Retrieves the execution history for a specific prompt ID."""
        with urllib.request.urlopen(f"{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    async def run_workflow(self, payload, use_workflow_json=True):
        """
        (Public, Async) The main method to submit a workflow for execution.
        It is separate from _test_server to avoid deadlocks.
        """
        await self._ensure_initialized() # Guarantees a ready and validated instance.
        self._log_message(f"ComfyConnector run_workflow: Running workflow with payload: {json.dumps(payload)[:50]}...", "debug") # only show the first 50 characters of the payload
        prompt = await self._get_prompt_from_workflow(payload) if use_workflow_json else payload
        if not prompt:
            raise ValueError("[ComfyConnector] run_workflow: Failed to generate a valid prompt from the workflow.")
            
        return await self._execute_prompt_and_wait(prompt)

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
            raise FileNotFoundError(f"[ComfyConnector] upload_data: Source file does not exist: {source_path}")

        # Determine the base directory for the upload.
        if folder_type == 'models':
            if not subfolder:
                raise ValueError("[ComfyConnector] upload_data: The 'subfolder' argument is required when folder_type is 'models'.")
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
            raise FileExistsError(f"[ComfyConnector] upload_data: Destination file already exists and overwrite is False: {destination_path}")

        try:
            # Use shutil.copy2 to preserve metadata.
            shutil.copy2(source_path, destination_path)
            self._log_message(f"upload_data: Successfully saved '{source_path}' to '{destination_path}'", "debug")

            # If the upload is ephemeral, add it to the list for later deletion,
            # but only if an overwrite of an existing file did not occur.
            if is_ephemeral:
                if overwrite and file_existed_before_upload:
                    self._log_message(f"upload_data: '{destination_path}' was overwritten and will not be marked for ephemeral deletion.", "warning")
                else:
                    self.ephemeral_files.append(destination_path)
                    self._log_message(f"upload_data: '{destination_path}' marked for ephemeral deletion.", "debug")

            return destination_path
        except Exception as e:
            self._log_message(f"upload_data: Failed to save file '{source_path}' to '{destination_path}'. Error: {e}", "error")
            raise

    def _delete_data(self):
        """
        (Internal, Sync) Deletes all files that were marked as ephemeral during the session.
        This is called during the shutdown process.
        """
        self._log_message("_delete_data: Cleaning up ephemeral files...", "debug")
        for fpath in self.ephemeral_files:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
                    self._log_message(f"_delete_data: Deleted '{fpath}'", "debug")
                else:
                    self._log_message(f"_delete_data: Warning: File '{fpath}' not found for deletion.", "warning")
            except Exception as e:
                self._log_message(f"_delete_data: Error deleting file '{fpath}': {e}", "error")
        self.ephemeral_files.clear()  # Clear the list after attempting deletion