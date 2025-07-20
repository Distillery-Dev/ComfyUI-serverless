*Refactoring Strategy: Integrating `WorkflowContext` for High-Performance I/O*


1. Executive Summary

The current data handling mechanism within the Discomfort repository, particularly in ``run_sequential``, suffers from a critical performance bottleneck. Its reliance on base64 encoding/decoding for passing data between workflows introduces significant overhead, turning seconds of computation into minutes of waiting. The existing system is a patchwork of direct serialization, history inspection, and inline data injection that is both inefficient and difficult to maintain.

To solve this, we have developed the `WorkflowContext`, a robust, high-performance class designed to manage all ephemeral data I/O for a given workflow run. This document outlines the strategic plan to refactor `DiscomfortPort` and `DiscomfortDataLoader` to exclusively use this new handler, thereby eliminating the current bottleneck and creating a standardized, resilient, and maintainable data-passing architecture.

The core principle of this refactor is: The `run_sequential` method will own the `WorkflowContext` instance, and all data will be passed through it.


2. The New Architecture: A Centralized Handler

The new design is centered around the `WorkflowContext` and its "per-run" lifecycle.

Instantiation and Ownership: For every call to `run_sequential`, a new `WorkflowContext` instance will be created. This is best done using a with statement to guarantee that its `shutdown()` method is called, ensuring all temporary resources (the Plasma store and disk cache) are automatically cleaned up.

A Single Source of Truth: This handler instance will be the single, authoritative channel for all data that needs to be passed between workflows or iterations. The old methods of serializing data into the prompt JSON or parsing it from the execution history will be completely removed.

Passing the Handler: The `run_sequential` orchestrator will be responsible for making the handler available to the nodes that need it. Since nodes within a ComfyUI workflow are isolated, we cannot pass the instance directly. Instead, we will establish a simple, run-specific global registry to hold the active handler instance, which the nodes can then access.

Benefits of this Approach:
Performance: Eliminates the base64 bottleneck entirely. Data is passed efficiently through shared memory (Plasma) or fast disk serialization (cloudpickle), drastically reducing I/O overhead.

Resilience: The handler's automatic RAM-to-disk fallback ensures that workflows can complete even under memory pressure.

Simplicity & Maintainability: The logic for saving and loading data is centralized in one well-tested class. `DiscomfortPort` and `DiscomfortDataLoader` become simple interfaces to this powerful backend, making their code cleaner and easier to understand.

Ephemeral by Default: The architecture guarantees that no temporary data is left behind after a run, preventing disk space leaks and state corruption between runs.


3. The Refactoring Task List

To implement this new architecture, we will perform a series of targeted refactors across the codebase.

Task 1: Modify `workflow_tools.py` - The Orchestrator
The `run_sequential` method will be the heart of the new design.

Instantiate the Handler: At the very beginning of `run_sequential`, instantiate the `WorkflowContext` within a with block. This ensures its lifecycle is tied to the run.

Create a Global Registry: Implement a simple, thread-safe dictionary at the module level to act as a registry for active handlers.

```
# In workflow_tools.py
import threading
_active_handlers = {}
_handler_lock = threading.Lock()
```

Register the Handler: Inside the with block, generate a unique `run_id` and register the handler instance in the `_active_handlers` dictionary. This run_id will be passed to the nodes.

Modify Prompt Injection: The _inject_data_loaders_into_prompt method will be simplified. Instead of injecting serialized data, it will now only need to pass two things to the `DiscomfortDataLoader` node:

The `unique_id` of the data to load.

The `run_id` that maps to the active `WorkflowContext`.

Modify Output Extraction: The logic for parsing execution history will be completely removed. Since `DiscomfortPort` will now save data directly via the handler, `run_sequential` only needs to load the final outputs from the handler instance by their unique_id after the workflow execution is complete.

Unregister the Handler: In the finally part of the with block (or automatically via `__exit__`), remove the handler from the `_active_handlers` registry to prevent memory leaks.

Task 2: Refactor nodes.py - The `DiscomfortPort`
The `DiscomfortPort` will be simplified to become a pure "save" interface.

Add `run_id` Input: Add a new hidden input to the `DiscomfortPort` to receive the `run_id` from the orchestrator.

Remove Serialization Logic: All existing code related to `serialize()`, `storage_type`, JSON encoding, and preparing UI dictionaries for history extraction will be deleted.

Implement `handler.save()`: The `process_port` method will be streamlined. If it's in OUTPUT mode, it will:
a.  Look up the active `WorkflowContext` from the global registry using its run_id.
b.  Call `handler.save(unique_id=self.unique_id, data=input_data)`.
c.  Pass the original input_data through as its result.
This makes the port's logic trivial and delegates all the complex I/O work to the handler.

Task 3: Refactor `nodes_internal.py` - The `DiscomfortDataLoader`
The `DiscomfortDataLoader` will become a pure "load" interface.

Update Inputs: The `INPUT_TYPES` will be changed. It no longer needs `storage_type` or a large `storage_key`. It will now only require:

`run_id`: The ID of the active run.

`unique_id`: The ID of the data it needs to load.

Remove Deserialization Logic: All existing code for handling different storage types and calling `deserialize()` will be deleted.

Implement handler.load(): The load_data method will be simplified to:
a.  Look up the active `WorkflowContext` from the global registry using its `run_id`.
b.  Call `return (handler.load(unique_id=self.unique_id),)`.
This offloads all the complexity of figuring out where and how the data is stored to the handler.

4. Conclusion
By executing this refactoring plan, we will transform the Discomfort I/O system from a slow, brittle, and complex mechanism into a modern, high-performance, and maintainable architecture. The `WorkflowContext` will become the central pillar of this new system, providing a robust foundation for all future development and ensuring that the Discomfort extension is both powerful and efficient. This change directly addresses the most significant performance issue in the repository and aligns the codebase with best practices for managing stateful, data-intensive operations.