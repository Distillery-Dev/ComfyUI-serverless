# TASK PLAN

## Refactoring Plan: Encapsulating Logic into a `Discomfort` Master Class

**Objective:** Refactor the existing codebase to encapsulate the core components (`WorkflowTools`, `WorkflowContext`, `ComfyConnector`) within a single, developer-friendly `Discomfort` class. This will provide a simplified and intuitive API for users to programmatically control ComfyUI, as outlined in `MISSION.md`. The end goal is to make the example code provided in the mission statement fully operational.

-----

### **Task 1: Create the `Discomfort` Class and Project Structure**

This initial task focuses on setting up the new main class file and its basic structure without breaking existing code.

  * **Action:** Create a new Python file: `discomfort.py`. This file will house the new master class.
  * **Action:** Inside `discomfort.py`, define the main `Discomfort` class.
  * **Action:** Import the necessary existing classes into `discomfort.py`. This follows the directive to not create a monolithic file and instead use imports.
    ```python
    # discomfort.py
    from .workflow_tools import WorkflowTools
    from .workflow_context import WorkflowContext
    from .comfy_serverless import ComfyConnector
    from typing import Dict, List, Any, Optional
    ```

-----

### **Task 2: Implement the `Discomfort` Class Core**

This task involves defining the initialization, factory, and shutdown methods for the `Discomfort` class. This establishes how it will manage its component parts according to the project specifications.

  * **Action:** Define the `__init__` method. It should be lightweight, initializing instance holders for its tools and worker, and exposing the `Context` class.
      * It will create an instance of `WorkflowTools` and assign it to `self.Tools`.
      * It will assign the `WorkflowContext` class directly to a class attribute `Context`. This is key to allowing users to call `discomfort.Context()` to create new context managers.
      * It will initialize `self.Worker` to `None`. The actual `ComfyConnector` instance will be created asynchronously by the factory method.
  * **Action:** Create an `async` factory method `Discomfort.create()`. This will be the standard, user-facing way to get a ready-to-use instance of the `Discomfort` class.
      * This `@classmethod` will create an instance of `Discomfort`.
      * It will then call `await ComfyConnector.create()` to get the singleton instance of the worker.
      * It will assign the returned worker instance to `instance.Worker`, making it accessible.
      * Finally, it will return the fully initialized `Discomfort` instance.
  * **Action:** Create an `async shutdown()` method to provide a clean way to terminate the managed ComfyUI instance by calling `await self.Worker.kill_api()`.
  * **File to Modify:** `discomfort.py`
  * **Proposed Code Structure:**
    ```python
    class Discomfort:
        Context = WorkflowContext # Expose Context class directly

        def __init__(self):
            self.Tools = WorkflowTools()
            self.Worker = None # Placeholder for the ComfyConnector instance

        @classmethod
        async def create(cls, config_path=None):
            """Creates and initializes a Discomfort instance and its worker."""
            self = cls()
            self.Worker = await ComfyConnector.create(config_path=config_path)
            return self

        async def shutdown(self):
            """Shuts down the managed ComfyUI worker instance."""
            if self.Worker:
                await self.Worker.kill_api()

        # The run() method will be added in Task 3
    ```

-----

### **Task 3: Implement the `Discomfort.run()` Orchestration Method**

This is the central part of the refactoring. It involves migrating the `run_sequential` logic into the new `Discomfort` class to provide the primary, high-level execution API.

  * **Action:** Define a new `async` method `run()` within the `Discomfort` class.
  * **Action:** Update the method signature to make the `inputs` parameter optional by giving it a default value of `None`. This natively supports workflows that do not require external data injection.
    ```python
    async def run(self, workflow_paths: List[str], inputs: Optional[Dict[str, Any]] = None, iterations: int = 1, use_ram: bool = True, context: Optional[WorkflowContext] = None):
    ```
  * **Action:** Implement the context-aware logic.
      * **If an external `context` object is passed** to `run()`, the method must use that object directly. This is crucial for allowing users to manage state across multiple `run()` calls within a single `with discomfort.Context() as context:` block.
      * **If the `context` argument is `None`**, the method should create its own temporary `WorkflowContext` instance using a `with` statement. This preserves functionality for simple, single-shot runs.
  * **Action:** Adapt the core logic from `WorkflowTools.run_sequential`.
      * The main execution loop and workflow processing logic (discovering ports, handling inputs, stitching 'ref' variables, preparing prompts, executing, and processing outputs) will be moved from `run_sequential` into `Discomfort.run()`.
      * The method will use `self.Tools` for helper functions and `self.Worker` to execute the workflow.
  * **Action:** Once `Discomfort.run()` is fully tested, the original `WorkflowTools.run_sequential` method should be marked for deprecation or removed to establish `Discomfort.run()` as the single source of truth for orchestration.

-----

### **Task 4: Update Exports, Integration, and Internal Tests**

Ensure the new `Discomfort` class is correctly exposed as the primary entry point for the package and that internal components use the new API.

  * **Action:** Modify the main `__init__.py` file (`__init__.py`) to import and export the new `Discomfort` class. This will make it easily importable for end-users via `from custom_nodes.discomfort import Discomfort`.
    ```python
    # __init__.py (Proposed)

    from .discomfort import Discomfort # New import
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "Discomfort"]
    ```
  * **Action:** Update the `DiscomfortTestRunner` node in `nodes.py`.
      * The `run_test` method currently instantiates `WorkflowTools` and calls `run_sequential`. This must be updated to use the new `Discomfort` class and its `run()` method (`discomfort = await Discomfort.create()`, then `await discomfort.run(...)`). This ensures your internal tests remain valid and are testing the primary, user-facing API.

-----

### **Task 5: Finalize Documentation**

The final and most critical step is to update all user-facing documentation to reflect the new, simplified API, making the project accessible and easy to adopt.

  * **Action:** Perform a comprehensive revision of `README.md`.
      * Update the "Core Concept" and "Architecture" sections to introduce the `Discomfort` class as the central orchestrator.
      * Replace all old usage examples with the new, cleaner syntax as demonstrated in `MISSION.md`. This should be the very first code example a new user sees.
      * Clearly explain the roles of the accessible components: `discomfort.Worker`, `discomfort.Tools`, and `discomfort.Context()`.
      * Ensure the documentation and examples cover the use of `run()` both with and without the optional `inputs` parameter.
  * **Action:** Add the final, working version of the iterative image enhancement example from `MISSION.md` to the `README.md` to showcase the power and simplicity of the new design.
  * **Action:** Once this refactoring is complete, you can safely delete `MISSION.md`, as its purpose will have been fulfilled. This `CURRENT_ISSUES.md` document will serve as the historical record of the planned and executed changes.