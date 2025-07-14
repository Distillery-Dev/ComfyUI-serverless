"# High-Level Plan for IF/THEN and DO/WHILE in DiscomfortLoopExecutor

The node (to be added in `nodes.py`) will be a custom ComfyUI node that wraps `DiscomfortWorkflowTools().run_sequential`, exposing loop params as inputs/widgets while adding conditional logic. It'll use `current_inputs`-style state internally (or call run_sequential with callbacks for evals). This design enables stateful, dependent iterations with programmatic control flows, building on port unique_ids as 'variables' for chaining and conditions. To streamline, we unify into a single loop mode (DO_WHILE) with optional nested branches (IF_THEN), reducing UI choices while supporting nested logic. Here's the proposed design, broken down with expansions for clarity, examples, and edge cases:

1. **Node Basics**:
   - **Class Name**: DiscomfortLoopExecutor
   - **CATEGORY**: "discomfort/control"
   - **OUTPUT_NODE**: True (to expose final outputs in history/UI, allowing integration with other ComfyUI nodes).
   - **RETURN_TYPES**: (any_typ,) or a tuple based on exposed outputs (e.g., dynamically map final loop_inputs values to output slots using inferred types from ports; fallback to ANY for unknowns).
   - **FUNCTION**: "execute_loop" (main method to orchestrate parsing, state management, conditional evaluation, and run_sequential calls).
   - **Additional Notes**: The node maintains an internal state (e.g., loop_inputs dict) across executions, ensuring dependent loops where outputs feed subsequent inputs via unique_id matching. It prioritizes safety (e.g., max_iterations cap) and logging for debugging. Unification avoids separate modes, defaulting to a capped loop with optional branches for decisions within iterations.

2. **INPUT_TYPES** (UI Params):
   - **Required**:
     - "workflow_paths": (STRING, {"multiline": True, "default": "path/to/A.json\npath/to/B.json"}) — List of JSON paths (split by newline for UI ease). These form the main chain executed per iteration or branch. Example: For daisy-chaining, list in order; stitching integrates them into a single DAG if multiple.
     - "max_iterations": (INT, {"default": 1, "min": 1, "max": 1000}) — Maximum loops (safety cap for DO_WHILE to prevent infinite runs; also used in default condition_expression).
     - "initial_inputs": (STRING, {"multiline": True, "default": "unique_id1: value1\nunique_id2: value2"}) — Serialized dict for startup (parse to dict in code, e.g., split lines and convert values with type inference). Supports simple formats like 'key: 42' (INT) or 'key: [1,2,3]' (LIST); used to seed loop_inputs.
   - **Optional**:
     - "loop_condition_expression": (STRING, {"multiline": True, "default": "discomfort_loop_counter <= max_iterations"}) — Safe-parsed expression for the outer DO_WHILE loop (evaluated after each main run; continue if true).
     - "branch_condition_expression": (STRING, {"multiline": True, "default": ""}) — Optional expression for inner IF_THEN branching (evaluated post-main if provided; enables nesting).
     - "condition_port": (STRING, {"default": ""}) — Unique_id of a primary port to evaluate (e.g., for simple bool checks; used in expressions as 'port_value' or directly if no expression).
     - "then_workflows": (STRING, {"multiline": True, "default": ""}) — Alt paths for TRUE branch (newline-separated). Special keywords: 'LOOP:BREAK' (exit loop), 'LOOP:PASS' (no-op, continue), 'LOOP:CONTINUE' (skip to next iteration, equivalent to PASS).
     - "else_workflows": (STRING, {"multiline": True, "default": ""}) — For FALSE branch (optional; supports same special keywords).
     - "use_ram": (BOOLEAN, {"default": True}) — Prefer in-RAM storage for intermediates (falls back to disk if False or large data detected).
     - "persist_prefix": (STRING, {"default": ""}) — Directory prefix for permanent disk saves (overwriting files; empty for temp only).
   - **Hidden/Advanced**: 
     - "break_called": (BOOLEAN, {"default": False, "forceInput": True}) — Internal flag set True if 'LOOP:BREAK' is evaluated; checked first in each iteration to short-circuit.
     - Tags for filtering ports, feedback mappings if needed (but minimize per your preference for unique_id matching). Additional hidden inputs could include timeouts or eval globals restrictions.

3. **Core Logic in execute_loop**:
   - **Initialization**: Create Tools instance; parse inputs (e.g., split workflow_paths into list, parse initial_inputs to dict with type coercion like int/float/str detection). Initialize loop_inputs = parsed_initial_inputs; set loop_inputs['discomfort_loop_counter'] = 1 (starts at 1 for inclusive <= max_iterations default); set break_called = False.
   - **Unified Loop Flow (DO_WHILE with Optional Branches)**:
     - For i in range(1, max_iterations + 1):
       - If break_called: break.
       - Run main workflow_paths via run_sequential (injecting loop_inputs, stitching if multiple).
       - Update loop_inputs with outputs.
       - If branch_condition_expression provided: Evaluate it on loop_inputs; if True, handle then_workflows (run if paths, or special keyword); if False, handle else_workflows.
       - Increment loop_inputs['discomfort_loop_counter'] = i + 1.
       - Evaluate loop_condition_expression on loop_inputs; if False, break.
     - Special Keyword Handling: 'LOOP:BREAK' sets break_called=True; 'LOOP:PASS'/'LOOP:CONTINUE' skips remaining steps in the iteration but proceeds to next (if condition allows).
   - **Post-execution**: Extract relevant final loop_inputs (e.g., based on tags or all non-internal keys), return as node outputs. Clean up temp storage if applicable.
   - **Error Handling**: Log type mismatches in evals (coerce where possible, e.g., bool(float)); timeout after excessive iterations; safe eval to block code injection (e.g., no imports); handle missing loop_inputs keys (default 0/False with warning).
   - **Examples**:
     - Simple Loop: Empty branch expr, default loop expr—runs main max_iterations times.
     - Nested IF in Loop: loop_condition="discomfort_loop_counter <= 5", branch_condition="score > 0.5", then="refine.json"—per iteration, run main, if score high run refine, then check outer condition.
     - Pure IF: max_iterations=1, set branch_condition, then/else—acts as non-looping IF_THEN.

4. **Integration with run_sequential and loop_inputs**:
   - The node calls run_sequential for main and branch segments, passing/receiving loop_inputs to maintain state (e.g., after main, use updated state for branches).
   - For evals: Post-run_sequential, query returned dict by unique_id; support multi-var expressions (e.g., 'input_a + input_b > 10').
   - Built-in State: 'discomfort_loop_counter' auto-managed in loop_inputs for expressions or ports.
   - Batches: Process sequentially; conditions can eval per-batch (e.g., break if any fails).
   - Stitching: Call for multi-path mains/branches to create unified DAGs.

5. **Testing and Edge Cases**:
   - **Simple DO_WHILE**: Default expr—runs max_iterations times, counter correct.
   - **Nested Branch**: Loop with inner IF, branches execute conditionally per iteration.
   - **Special Keywords**: 'LOOP:CONTINUE' skips branch but continues loop; 'LOOP:BREAK' exits early.
   - **Edge Cases**: max_iterations=1 as pure IF; missing expr (treat as True); invalid expr (False with log); counter overwrite by user port (warning); batches with per-item branches; OOM in branches (fallback to disk).
   - **Validation**: break_called persistence; nesting without recursion limits; cross-iteration state dependency." 

6. **State Encapsulation Principles**:
   - **Core Rule**: All loop state is encapsulated in the loop_inputs dict—evals, injections, updates, and outputs reference only its keys/values. No external factors (e.g., global vars, ComfyUI session state, or direct file reads) affect logic, ensuring determinism and modularity.
   - **Why?**: Enables self-contained flows (predictable from initial_inputs and workflows alone), simplifies serialization (state is portable via unified format), and aligns with type-agnostic design. It treats unique_ids as 'variables' for all computations.
   - **Enforcement**: Evals use restricted namespaces (e.g., simpleeval with names=loop_inputs); internal logic reads/writes solely via the dict. Reserved keys like 'discomfort_loop_counter' are protected (warn on overwrites). If a key is missing in evals, default safely (e.g., False/0) with logs.
   - **Examples**: Condition 'score > 0.5' pulls 'score' from loop_inputs (set by a port output); attempting external refs (e.g., 'os.path.exists()') fails safely. For batches, store as sub-structures (e.g., {'batch_state': {...}}) within the dict.
   - **Edge Cases**: User-overwritten reserved keys (preserve built-in, log conflict); empty loop_inputs (default evals to False); serialization of state for persistence (ensures totality is saved/loaded)." 