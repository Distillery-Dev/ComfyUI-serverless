import uuid
import json
import urllib.request
import urllib.parse
import websocket  # Using the synchronous 'websocket-client' library as required by ComfyUI.
import requests
import os
import subprocess
import sys
import shutil
from typing import Dict, Any
import asyncio  # Imported to run synchronous (blocking) code in a separate thread.
from playwright.async_api import async_playwright, Page, Browser  # Using Playwright's Async API to work with ComfyUI's async environment.
import aiohttp  # Added for async HTTP requests to reduce blocking
import backoff  # For retry logic
import atexit # For registering cleanup functions

class ComfyConnector:
    """
    An ASYNCHRONOUS connector to start and interact with a ComfyUI instance in a serverless-like environment.
    This class uses a singleton pattern to ensure that only one instance of the ComfyUI server
    is managed per process, which is crucial for resource management.
    Enhanced with explicit state management, validation, and idempotent operations for stability.
    """
    logger = None # Logger for the class
    _instance = None  # Singleton instance
    _process = None  # Subprocess for ComfyUI API
    _state = "uninit"  # Lifecycle state: uninit | initializing | ready | error | killed
    playwright = None  # Playwright instance
    _browser = None  # Headless browser
    _page = None  # Browser page for JS execution
    _init_lock = asyncio.Lock()  # Lock to prevent race conditions during initialization
    _state_lock = asyncio.Lock()  # Finer-grained lock for state checks and mutations
    _cleanup_registered = False # Flag to prevent multiple cleanup registrations

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

        logger = logging.getLogger(f"ComfyConnector_{id(self)}")
        logger.propagate = False
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)
            formatter = CustomFormatter('%(asctime)s - [ComfyConnector] (%(caller_funcName)s) %(levelname)s - %(message)s')
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def _log_message(self, message: str, level: str = "info"):
        """Centralized logging method for the handler."""
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(message)

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
            raise FileNotFoundError(f"[ComfyConnector] (__init__) Configuration file not found: {config_path}")

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
        instance.logger = instance._get_logger()
        await instance._ensure_initialized()
        instance._log_message("create: instance called")
        if not cls._cleanup_registered:
            # Register the synchronous cleanup method directly
            atexit.register(cls._sync_cleanup)
            cls._cleanup_registered = True
            instance._log_message("Cleanup function registered successfully.", "info")
        return instance

    async def _validate_resources(self) -> bool:
        """
        (Internal, Async) Validate all underlying resources (process, browser, page, WS, HTTP).
        Returns True if all are operational.
        """
        async with self._state_lock:
            if self._process is None or self._process.poll() is not None:
                self._log_message("Process is not alive", "warning")
                return False
            if ComfyConnector._browser is None or not ComfyConnector._browser.is_connected():
                self._log_message("Browser is closed", "warning")
                return False
            if ComfyConnector._page is None or ComfyConnector._page.is_closed():
                self._log_message("Page is closed", "warning")
                return False
            if not self.ws.connected:
                self._log_message("WebSocket is not connected", "warning")
                return False
            # Async HTTP check using aiohttp for non-blocking validation
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.server_address, timeout=1) as response:
                        if response.status != 200:
                            self._log_message(f"Server responded with {response.status}", "warning")
                            return False
            except Exception as e:
                self._log_message(f"HTTP validation failed: {e}", "warning")
                return False
            return True

    async def _ensure_initialized(self):
        """
        (Internal) Ensures the server and browser are started. This async method contains all
        the heavy setup logic and is designed to run only once, safely. Enhanced to be idempotent,
        with retries and resource validation.
        """
        self._log_message("Ensuring initialization...")
        # The async lock prevents multiple concurrent calls from trying to initialize at the same time.
        async with self._init_lock:
            # First, handle the 'ready' state. If it's ready, validate it.
            # If it's no longer valid, set state to 'error' and fall through to re-initialize.
            if self._state == "ready":
                if await self._validate_resources():
                    self._log_message("Resources are ready and valid. Skipping initialization.", "debug")
                    return
                else:
                    self._log_message("Resources were ready but failed validation. Forcing re-initialization.", "warning")
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
            @backoff.on_exception(backoff.expo, Exception, max_tries=20)
            async def init_with_retry():
                await self._init_playwright()
                await self._start_api()
                await self._ensure_fresh_websocket()

            try:
                await init_with_retry()
                test_passed = await self._test_server()
                if test_passed:
                    self._state = "ready"
                    self._log_message("Initialization complete and test passed", "debug")
                else:
                    self._log_message("Test failed", "error")
                    raise RuntimeError("[ComfyConnector] (_ensure_initialized) Test failed")
            except Exception as e: # This is the only place where we raise an error
                self._log_message(f"Initialization failed after retries: {e}", "error")
                self._state = "error"
                await self.kill_api()
                raise

    async def _reset_state(self):
        """
        (Internal, Async) Resets internal state variables for a clean re-init.
        """
        self._log_message("Resetting connector state.", "debug")
        self._process = None
        # Re-instantiate WebSocket client to clear any old connection state
        self.ws = websocket.WebSocket()
        ComfyConnector.playwright = None
        ComfyConnector._browser = None
        ComfyConnector._page = None
        self.ephemeral_files.clear()
        self._killed = False
        self._state = "uninit"
        self.client_id = str(uuid.uuid4())
        self._cleanup_registered = False
        self._log_message("State reset.", "debug")

    async def _init_playwright(self):
        """
        (Internal, Async) Initialize Playwright and launch headless Chromium browser.
        Handles auto-installation if browser executable is missing.
        """
        self._log_message("Initializing Playwright and headless browser", "debug")
        ComfyConnector.playwright = await async_playwright().start()
        try:
            ComfyConnector._browser = await ComfyConnector.playwright.chromium.launch(
                                headless=True,  # Temporary for debug; change back to True
                                args=['--enable-webgl', '--disable-gpu']
                                )
            self._log_message("Playwright and headless Chromium initialized successfully.", "debug")
        except Exception as e:
            if "Executable doesn't exist" in str(e) or "missing dependencies" in str(e):
                self._log_message("--- ONE-TIME SETUP: PLAYWRIGHT BROWSER NOT FOUND ---", "info")
                self._log_message("Attempting to install Chromium now. This may take a few minutes...", "info")
                try:
                    install_command = [sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"]
                    process = await asyncio.create_subprocess_exec(
                        *install_command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()

                    if process.returncode != 0:
                        self._log_message(f"Playwright install failed: {stderr.decode()}", "error")
                        raise RuntimeError(f"[ComfyConnector] (_init_playwright) Playwright install failed: {stderr.decode()}")
                    
                    self._log_message("--- BROWSER INSTALLATION COMPLETE ---", "info")
                    ComfyConnector._browser = await ComfyConnector.playwright.chromium.launch(headless=True)
                    self._log_message("Playwright and headless Chromium re-initialized successfully.", "debug")
                except Exception as install_exc:
                    self._log_message(f"Failed to automatically install Playwright's browser: {install_exc}", "error")
                    raise RuntimeError(f"[ComfyConnector] (_init_playwright) Failed to automatically install Playwright's browser: {install_exc}") from install_exc
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
        self._log_message("Starting ComfyUI API process...", "debug")
        if await self._is_api_running():
            self._log_message("API is already running.", "debug")
            return

        if self._process is None or self._process.poll() is not None:
            api_command_line = self.api_command_line.split() + [f"--port", str(self.urlport)]
            self._process = subprocess.Popen(api_command_line, cwd=self.api_working_directory)
            self._log_message(f"API process started with PID: {self._process.pid}", "debug")

            attempts = 0
            while not await self._is_api_running():
                if self._process.poll() is not None:
                    self._log_message("API process terminated unexpectedly during startup.", "error")
                    raise RuntimeError("[ComfyConnector] (_start_api) API process terminated unexpectedly during startup.")
                if attempts >= self.max_start_attempts:
                    await self.kill_api()
                    self._log_message(f"API startup failed after {attempts} attempts.", "error")
                    raise RuntimeError(f"[ComfyConnector] (_start_api) API startup failed after {attempts} attempts.")
                await asyncio.sleep(self.start_attempts_sleep)  # Use asyncio.sleep in an async method.
                attempts += 1
            self._log_message(f"API is responsive on port {self.urlport} (PID: {self._process.pid}).", "debug")

    async def _is_api_running(self):
        """
        (Internal, Async) Checks if the server is listening (HTTP response only; no full test to avoid recursion).
        Uses aiohttp for non-blocking check.
        """
        self._log_message("Checking if server is running...", "debug")
        try:
            # A simple, quick async HTTP check
            async with aiohttp.ClientSession() as session:
                async with session.get(self.server_address, timeout=1) as response:
                    return response.status == 200
            self._log_message("Server is running.", "debug")
        except Exception:
            self._log_message("Server is NOT running.", "debug")
            # Any exception means the server is not ready.
            return False

    async def get_prompt_from_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        (Async) Converts a workflow JSON into an API-ready prompt by using the
        headless browser to execute ComfyUI's internal JavaScript functions.
        Enhanced with waits for stability in headless mode.
        """
        self._log_message("Converting workflow to prompt...", "debug")

        if not await self._validate_resources():
            self._log_message("Resources are not valid. Cannot convert workflow to prompt.", "error")
            raise RuntimeError("[ComfyConnector] (get_prompt_from_workflow) Resources are not valid. Cannot convert workflow to prompt.")
        
        self._log_message("Converting workflow to prompt via headless browser...", "debug")
        try:
            await ComfyConnector._page.goto(self.server_address) # Go to the server address
            await ComfyConnector._page.wait_for_function("() => typeof window.app !== 'undefined'", timeout=60000) # Wait for the app to be loaded
            await ComfyConnector._page.evaluate("async (wf) => { await window.app.loadGraphData(wf); }", workflow) # Load the workflow
            prompt_data = await ComfyConnector._page.evaluate("async () => { return await window.app.graphToPrompt(); }") # Convert the workflow to a prompt json
            
            self._log_message("Successfully converted workflow to prompt.", "debug")
            return prompt_data['output']
        except Exception as e:
            self._log_message(f"Page load or app init failed: {e}", "error")
            raise

    def _execute_prompt_and_wait(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        (Internal, Sync) Queues a prompt, waits for its execution via WebSocket, and retrieves the history.
        This synchronous version processes messages without artificial delays.
        """
        try:
            # 1. Queue the prompt using a synchronous HTTP request.
            req = urllib.request.Request(f"{self.server_address}/prompt", data=json.dumps({"prompt": prompt, "client_id": self.client_id}).encode('utf-8'))
            with urllib.request.urlopen(req) as response:
                prompt_id = json.loads(response.read())['prompt_id']
            self._log_message(f"Prompt queued with ID: {prompt_id}", "debug")

            # 2. Wait for the execution to finish by listening on the WebSocket.
            while True:
                out = self.ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    # The 'executing' message with a null 'node' indicates the prompt is done.
                    if message['type'] == 'executing' and message['data']['node'] is None and message['data']['prompt_id'] == prompt_id:
                        self._log_message(f"Execution finished for prompt ID: {prompt_id}", "debug")
                        break

            # 3. Retrieve the final history for the completed prompt.
            with urllib.request.urlopen(f"{self.server_address}/history/{prompt_id}") as response:
                return json.loads(response.read())
        except Exception as e:
            self._log_message(f"Error during prompt execution: {e}", "error")
            return None

    async def _test_server(self):
        """
        (Internal, Async) Tests server readiness by running a simple workflow.
        This uses the internal _execute_prompt_and_wait method to avoid deadlocks.
        """
        self._log_message("Running server validation test...", "debug")
        try:
            with open(self.test_payload_file, 'r') as f:
                test_workflow = json.load(f)
            
            test_prompt = await self.get_prompt_from_workflow(test_workflow)
            history = await asyncio.to_thread(self._execute_prompt_and_wait,test_prompt)
            
            # A valid, non-empty history object indicates a successful test.
            return bool(history)
        except Exception as e:
            self._log_message(f"Server test failed: {e}", "error")
            return False

    async def kill_api(self):
        """
        (Async) Kills the server, closes connections, and resets the singleton state. Idempotent.
        """
        async with self._state_lock:
            if self._killed or self._state == "killed":
                self._log_message("Already killed, skipping.", "debug")
                return
            self._killed = True        
        self._log_message("Shutting down ComfyUI instance...", "debug")
        # Cleanup ephemeral files
        self._delete_data()        
        # Close WebSocket connection
        if self.ws and self.ws.connected:
            try:
                self.ws.close()
            except Exception as e:
                self._log_message(f"Error closing WebSocket: {e}", "warning")        
        # Close browser - handle event loop closure gracefully
        if ComfyConnector._browser:
            try:
                # Check if we're in an active event loop
                loop = asyncio.get_event_loop()
                if loop.is_running() and not loop.is_closed():
                    await ComfyConnector._browser.close()
                else:
                    # If no active loop, use sync approach
                    self._log_message("Event loop closed, skipping async browser close", "debug")
            except (RuntimeError, asyncio.CancelledError) as e:
                if "Event loop is closed" in str(e) or "different loop" in str(e):
                    self._log_message(f"Ignored expected shutdown error for browser: {e}", "debug")
                else:
                    self._log_message(f"Could not gracefully close browser: {e}", "warning")
            except Exception as e:
                self._log_message(f"Unexpected error closing browser: {e}", "warning")        
        # Stop Playwright - also handle event loop closure
        if ComfyConnector.playwright:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running() and not loop.is_closed():
                    await ComfyConnector.playwright.stop()
                else:
                    self._log_message("Event loop closed, skipping async playwright stop", "debug")
            except (RuntimeError, asyncio.CancelledError) as e:
                if "Event loop is closed" in str(e) or "different loop" in str(e):
                    self._log_message(f"Ignored expected shutdown error for playwright: {e}", "debug")
                else:
                    self._log_message(f"Could not gracefully stop Playwright: {e}", "warning")
            except Exception as e:
                self._log_message(f"Unexpected error stopping Playwright: {e}", "warning")        
        # Forcefully terminate the subprocess
        if self._process and self._process.poll() is None:
            self._log_message(f"Forcefully terminating process with PID: {self._process.pid}", "info")
            self._process.kill()
            try:
                # Check if we can use async wait
                loop = asyncio.get_event_loop()
                if loop.is_running() and not loop.is_closed():
                    await asyncio.to_thread(self._process.wait)
                else:
                    # Fallback to sync wait
                    self._process.wait(timeout=5)
            except asyncio.TimeoutError:
                self._log_message("Process termination timed out", "warning")
            except Exception as e:
                self._log_message(f"Error during process termination: {e}", "warning")        
        # Reset the state
        await self._reset_state()
        self._state = "killed"
        self._log_message("Shutdown complete.", "debug")
        self.logger = None # Reset the logger

    @classmethod
    def _sync_cleanup(cls):
        """Synchronous cleanup method for atexit handler"""
        instance = cls._instance
        if instance is None:
            return
        instance._log_message("Main ComfyUI process is shutting down. Cleaning up nested server.", "info")
        # Cleanup ephemeral files
        if hasattr(instance, '_delete_data'):
            instance._delete_data()        
        # Close WebSocket
        if hasattr(instance, 'ws') and instance.ws and instance.ws.connected:
            try:
                instance.ws.close()
            except Exception as e:
                instance._log_message(f"Error closing WebSocket: {e}", "warning")        
        # Kill the subprocess directly (sync)
        if hasattr(instance, '_process') and instance._process and instance._process.poll() is None:
            instance._log_message(f"Forcefully terminating process with PID: {instance._process.pid}", "info")
            instance._process.kill()
            try:
                instance._process.wait(timeout=5)
            except Exception as e:
                instance._log_message(f"Error during process termination: {e}", "warning")        
        # Note: We skip browser/playwright cleanup in atexit as they require async
        instance._log_message("Sync cleanup complete.", "info")

    def _get_history(self, prompt_id):
        """(Internal, Sync) Retrieves the execution history for a specific prompt ID."""
        with urllib.request.urlopen(f"{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    async def _ensure_fresh_websocket(self):
        """
        (Internal, Async) Ensures the WebSocket connection is fresh and configured with keepalives.

        This helper is called before each execution to prevent hangs on stale connections
        by proactively reconnecting with robust keepalive options.
        """
        self._log_message("Ensuring fresh WebSocket connection...", "debug")
        try:
            # Proactively close any existing connection.
            if self.ws.connected:
                self.ws.close()
        except Exception:
            pass  # Ignore errors, e.g., if it was already closed.

        # Reconnect with keepalive options enabled.
        # `ping_interval` automatically sends pings to keep the connection alive.
        # `ping_timeout` ensures that if the server stops responding, the
        # connection is properly terminated, causing `recv()` to fail fast instead of hanging.
        await asyncio.to_thread(
            self.ws.connect,
            self.ws_address,
            ping_interval=100 # Send a keepalive ping every 100 seconds
        )
        self._log_message("WebSocket reconnected with keepalive enabled.", "info")

    async def run_workflow(self, payload, use_workflow_json=True):
        """
        (Public, Async) The main method to submit a workflow for execution.
        It is separate from _test_server to avoid deadlocks.
        """
        await self._ensure_initialized() # Guarantees a ready and validated instance.
        await self._ensure_fresh_websocket() # Ensure the WebSocket connection is fresh and configured with keepalives.
        self._log_message(f"Running workflow with payload: {json.dumps(payload)[:50]}...", "debug") # only show the first 50 characters of the payload
        prompt = await self.get_prompt_from_workflow(payload) if use_workflow_json else payload
        if not prompt:
            self._log_message("Failed to generate a valid prompt from the workflow.", "error")
            raise ValueError("[ComfyConnector] (_run_workflow) Failed to generate a valid prompt from the workflow.")
        return await asyncio.to_thread(self._execute_prompt_and_wait,prompt)

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
            self._log_message(f"Source file does not exist: {source_path}", "error")
            raise FileNotFoundError(f"[ComfyConnector] (_upload_data) Source file does not exist: {source_path}")

        # Determine the base directory for the upload.
        if folder_type == 'models':
            if not subfolder:
                self._log_message("The 'subfolder' argument is required when folder_type is 'models'.", "error")
                raise ValueError("[ComfyConnector] (_upload_data) The 'subfolder' argument is required when folder_type is 'models'.")
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
            self._log_message(f"Destination file already exists and overwrite is False: {destination_path}", "error")
            raise FileExistsError(f"[ComfyConnector] (_upload_data) Destination file already exists and overwrite is False: {destination_path}")

        try:
            # Use shutil.copy2 to preserve metadata.
            shutil.copy2(source_path, destination_path)
            self._log_message(f"Successfully saved '{source_path}' to '{destination_path}'", "debug")

            # If the upload is ephemeral, add it to the list for later deletion,
            # but only if an overwrite of an existing file did not occur.
            if is_ephemeral:
                if overwrite and file_existed_before_upload:
                    self._log_message(f"'{destination_path}' was overwritten and will not be marked for ephemeral deletion.", "warning")
                else:
                    self.ephemeral_files.append(destination_path)
                    self._log_message(f"'{destination_path}' marked for ephemeral deletion.", "debug")

            return destination_path
        except Exception as e:
            self._log_message(f"Failed to save file '{source_path}' to '{destination_path}'. Error: {e}", "error")
            raise

    def _delete_data(self):
        """
        (Internal, Sync) Deletes all files that were marked as ephemeral during the session.
        This is called during the shutdown process.
        """
        self._log_message("Cleaning up ephemeral files...", "debug")
        for fpath in self.ephemeral_files:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
                    self._log_message(f"Deleted '{fpath}'", "debug")
                else:
                    self._log_message(f"Warning: File '{fpath}' not found for deletion.", "warning")
            except Exception as e:
                self._log_message(f"Error deleting file '{fpath}': {e}", "error")
        self.ephemeral_files.clear()  # Clear the list after attempting deletion