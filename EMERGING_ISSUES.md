# Plan for Stabilizing ComfyConnector and Refactoring workflow_tools

We are seeing seeing errors — particularly a 'NoneType' on ws.connected and test failures when skipping init — that point to state inconsistencies in the singleton, likely from incomplete resets, race conditions in async ops, or over-reliance on the initialized flag without validating underlying resources (e.g., ws, browser, process). This can worsen with multiple TestRunners, as the first might kill the API, leaving the singleton in a "initialized but invalid" state for the second. The async nature adds complexity with locks and to_thread calls, but we can preserve it while adding guards and lazy checks.

To allow for the code to be more understandable and for errors to be better trackable, another refactoring of run_sequential() is needed. Currently, run_sequential() is a monolith handling setup, iteration, execution, extraction, and cleanup. Modularizing it into smaller, testable methods will make it clearer, easier to debug, and more extensible (e.g., for future conditionals/branches). The goal is to turn it into a high-level orchestrator that delegates to helper methods, focusing on loop logic and flow control.

Below, I've outlined detailed task lists for each effort. These are sequenced logically: start with analysis/debugging, then fixes/refactors, testing, and optimizations. We can tackle them iteratively, one sub-task at a time, to avoid overwhelming changes. I've thought about edge cases (e.g., multiple concurrent calls, failed inits, resource leaks) and best practices (e.g., async locks for state checks, idempotent operations).

IMPORTANT: whatever we do, we must ALWAYS preserve and expand/correct the comments in the code. We should always keep our codebase as one where only extremely well documented code is used.

## Task 1: Stabilizing ComfyConnector (Preserving Async Nature)

The core strategy: Introduce state validation beyond just the 'initialized' flag. Make init and kill idempotent (safe to call multiple times). Reduce unnecessary _ensure_initialized calls by making dependent methods (e.g., run_workflow) self-sufficient with lazy checks. Use finer-grained locks for critical sections, and add logging/tracing for async flows. Avoid breaking async by keeping awaits where needed and using to_thread only for true blockers.

*ComfyConnector is a critical component of the Discomfort repository. It must be (a) lightweight, (b) work with production-grade code fully explained via comments, (c) not rely on any workflow_tools imports (in fact it should only do imports of established packages), (d) and be extremely reliable for the tasks it purports to do, which is, plainly, to be a general-purposed Python wrapper for ComfyUI.*

### Task List:
1. **Debug and Analyze Current Issues**:
   - Reproduce the error consistently: Create a minimal test with two sequential DiscomfortTestRunners (or direct calls to run_sequential) and log all singleton state (e.g., initialized, _process poll, ws existence/connected, browser/page status) before/after each major op (init, run_workflow, kill).
   - Trace async call stacks: Add debug prints with asyncio.get_running_loop() and task names to identify race conditions, e.g., if kill_api is awaited properly or if to_thread ops overlap.
   - Identify unnecessary _ensure_initialized calls: Audit all callers (e.g., run_workflow, _get_prompt_from_workflow) and note where it's redundant (e.g., if already called upstream).

2. **Enhance State Management**:
   - Define explicit states: Add a class attr like _state = "uninit" | "initializing" | "ready" | "error" | "killed" to track lifecycle more granularly than just initialized bool.
   - Add validation methods: Create async _validate_resources() that checks process alive, browser open, page loaded, ws connectable—call this lazily in methods like run_workflow before ops.
   - Make _ensure_initialized idempotent and retryable: If state is "error" or resources invalid, reset and re-init; use a backoff retry (e.g., 3 attempts with sleep) for startup failures.

3. **Fix Singleton and Async Brittleness**:
   - Strengthen the lock: Ensure _init_lock protects all state mutations (e.g., setting playwright/browser/page); test for reentrancy by simulating concurrent calls.
   - Handle ws properly: In run_workflow, always check/reconnect ws if not connected (instead of assuming init handles it); wrap ws ops in try/except to reset on errors.
   - Improve kill_api: Make it fully reset state (set initialized=False, _state="killed", clear ephemeral_files); await all closes (browser, playwright); add a _wait_for_shutdown to poll process termination.
   - Reduce to_thread usage: Audit sync ops (e.g., requests.get, ws.connect/recv)—ensure they're only for true blockers; consider async alternatives (e.g., aiohttp for HTTP, websockets lib for async WS).

4. **Optimize Call Patterns**:
   - Lazy init components: Defer browser launch and page load until first needed (e.g., in _get_prompt_from_workflow); similarly for ws in run_workflow.
   - Cache and reuse: If initialized, skip full init but validate; add a _refresh_resources async method for partial recoveries (e.g., reconnect ws without relaunching browser).
   - Handle multiple instances: Though singleton, add guards for parallel calls (e.g., queue runs if initializing); test with concurrent TestRunners.

5. **Testing and Monitoring**:
   - Unit tests: Write async tests (using pytest-asyncio) for scenarios: first init success, re-init after kill, failed test_server triggering reset, concurrent calls.
   - Add logging: Use structured logging (e.g., with levels) for all state changes, awaits, and errors; include timings to spot bottlenecks.
   - Edge cases: Test with _test_server failures (mock unresponsive server), resource exhaustion (e.g., kill process mid-run), and rapid init/kill cycles.

6. **Final Polish**:
   - Document invariants: Add comments on expected states and async guarantees (e.g., "This method assumes lock is held").
   - Performance check: Ensure no unnecessary awaits block the loop; profile with asyncio debug modes.

## Task 2: Refactoring workflow_tools.py (Modularizing run_sequential)

Strategy: Break run_sequential into private helper methods, each focused on one responsibility (e.g., setup, per-iteration logic, execution, cleanup). Keep it async. The refactored run_sequential will be a thin async orchestrator: setup once, then loop over iterations (check conditions, process workflows sequentially, extract/update state), finally cleanup. This enables unit testing of helpers and easier extension (e.g., plug in branching).

### Task List:
1. **Analyze and Map Current Logic**:
   - Decompose run_sequential: List all sub-tasks (e.g., storage setup, connector init, port discovery, prompt conversion/injection, execution, output extraction, condition check, cleanup).
   - Identify reusable parts: Group into categories like setup/teardown, per-workflow, per-iteration.
   - Note dependencies: E.g., port_info needed for injection/extraction; ensure methods take/return minimal data (e.g., dicts for inputs/outputs).

2. **Define New Internal Methods**:
   - Create _setup_storage: Handle temp/persist dir creation, return storage_dir and is_ephemeral.
   - Create _init_connector: Await ComfyConnector.create() and wait for initialized; handle retries on failures.
   - Create _prepare_workflow: For a single path, load JSON, discover ports, convert to prompt (via connector).
   - Create _inject_inputs: Take prompt, inputs_dict, port_info, storage_dir—is_ephemeral, connector, use_ram; return modified_prompt (as current _inject_data_loaders_into_prompt, but rename/refine).
   - Create _execute_workflow: Take modified_prompt, connector; await run_workflow, return execution_result (history).
   - Create _extract_outputs: Take execution_result, port_info['outputs']; deserialize and return outputs_dict.
   - Create _evaluate_condition: Take condition_port/expression, current_outputs/inputs; return bool (extend for branches).
   - Create _cleanup: Await connector.kill_api, delete temp dir if ephemeral, clear memory.

3. **Refactor run_sequential Structure**:
   - Make it async orchestrator: Await _setup_storage and _init_connector once.
   - Outer loop: For iter in range(iterations):
     - Check loop condition via _evaluate_condition (break if false).
     - Inner loop: For each workflow_path:
       - Await _prepare_workflow to get prompt and port_info.
       - Await _inject_inputs with current loop_inputs.
       - Await _execute_workflow to get result.
       - extracted = await _extract_outputs.
       - Update loop_inputs and final_outputs with extracted.
     - Handle branch/then/else: If branch_condition, route to sub-loops using same helpers (recursive or separate _process_branch method).
   - After loops: Await _cleanup; return final_outputs.
   - Error handling: Wrap loops in try/finally for cleanup; propagate exceptions with context (e.g., "Iteration X, Workflow Y failed").

4. **Improve Modularity and Testability**:
   - Make methods private (prefix _ ) and self-contained: Pass self for tool access (e.g., serialize/deserialize).
   - Add type hints: For params/returns (e.g., async def _extract_outputs(self, result: Dict -> Dict[str, Any).
   - Extract constants: Move magic strings (e.g., 'discomfort_output') to class vars.

5. **Testing**:
   - Unit tests: Mock connector/execution results; test each helper in isolation (e.g., _inject_inputs produces correct prompt).
   - Integration: Test full refactored run_sequential with simple workflows; verify multi-iter, conditions.
   - Edge cases: Empty workflows, failed executions, large data (disk fallback), condition breaks.

6. **Documentation and Polish**:
   - Docstrings: For each new method, describe purpose, params, returns, side effects.
   - Logging: Centralize via self.log_message in helpers.
   - Extensibility: Design for future (e.g., _process_branch takes then/else lists, calls same inner loop).

This plan should make both components more robust and maintainable. We can start with Task 1's debugging to confirm root causes, then proceed step-by-step.