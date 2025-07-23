# ComfyUI Discomfort Extension

You have an awesome collection of workflows, carefully crafted and curated. Individually they work great, but making them work *together* -- deciding which ones to use, chaining their results, iterating workflows, etc -- is a pain. You have to open a workflow, run it manually, then decide which workflow to use next, run it manually again, check its results, etc...

What if you could just write script that would tell ComfyUI exactly what it must do for you?

I built Discomfort to resolve this. Discomfort is a ComfyUI extension that **allows ComfyUI to be fully run via code**.

(1) Discomfort enables **stateful run of ComfyUI workflows**, exposing its variables in Python and allowing them to be fully manipulated via code. 

(2) Discomfort allows users to **create conditionals and iterations** through two tricks: using nested ComfyUI runs that processes, and some clever handling the workflow JSON objects.

(3) Discomfort also allows users to **utilize partial workflows** (ex: a workflow that only contains sampling logic, another that only loads ControlNet, etc.), obviating the need to manually stitch workflow parts.

(4) Discomfort **does not touch ComfyUI's core code** and is fully compatible with ComfyUI's license; it fully leverages on ComfyUI's execution logic and is expected to work stably with future versions of ComfyUI.

Discomfort is 100% built on Python and is designed to be easy to use and learn.



## ⚠️ Current Status: Core Functionality Partially Operational

**IMPORTANT**: The core `run_sequential()` functionality is **working**, but it remains in active development and testing. While we have achieved successful executions in basic test cases, full robustness across all scenarios is not yet achieved. IMPORTANT: the project is under a major refactoring.

### What IS Working ✅
- **DiscomfortPort nodes**: Individual nodes function correctly in ComfyUI workflows, supporting INPUT (data injection), OUTPUT (data extraction), and PASSTHRU (data propagation) modes. Ports handle type-agnostic data passing, with serialization/deserialization for outputs.
- **Port discovery**: `discover_ports()` successfully analyzes workflows, identifies DiscomfortPorts, classifies their modes (INPUT/OUTPUT/PASSTHRU), infers data types from connections, and computes topological execution order.
- **Workflow stitching**: `stitch_workflows()` can merge multiple workflow JSONs by renumbering nodes/links, preserving connections, and enabling cross-workflow data flow via shared unique IDs.
- **Basic testing**: Test nodes like DiscomfortExtenderWorkflowRunner successfully execute workflows via `run_sequential()`, including data injection/extraction for single-iteration runs. Minimal workflows (e.g., test_server.json) and more complex ones (e.g., image extension pipelines) complete with correct outputs.
- **`run_sequential()` basics**: The method is a *WORK IN PROGRESS* but already executes workflows on a nested ComfyUI server.

### What is NOT Working ❌
- **Full iteration support**: While single-iteration execution works, multi-iteration loops (e.g., cross-iteration data flow) are untested and may require additional fixes for state preservation.
- **Conditional logic**: Branching (IF/THEN/ELSE) and advanced loop conditions are not yet implemented.
- **Large-scale scenarios**: Handling ultra-large models (e.g., GB-scale tensors) between orchestrator and nested server, multiple inputs/outputs per workflow, and disk fallback for memory-intensive cases need thorough testing.
- **Edge cases**: Potential issues with async timing, namespace conflicts, or validation in complex graphs remain possible until comprehensive testing is complete.
- **Logging segregation**: Discomfort logs are currently interleaved with ComfyUI's terminal output, making debugging harder in verbose scenarios.

## 🎯 Core Concept

ComfyUI's native execution model follows a Directed Acyclic Graph (DAG) pattern, executing nodes once in topological order. While powerful for single-pass workflows, this model doesn't natively support:
- Iterative refinement workflows
- Conditional execution paths  
- State preservation across iterations
- Dynamic workflow composition
- Programmatic access and management of ComfyUI instances

Discomfort addresses these limitations by introducing an execution layer on top of ComfyUI's DAG model, using pre-execution graph manipulation and a nested ComfyUI server for isolated runs, and a self-managed data store that handles context throughout the run.

Discomfort was designed to be trivially easy for anyone to use it. The only thing a user is required to know is how to add DiscomfortPorts to their existing workflows; everything else must be handled by Discomfort's internal logic.


## 🏗️ Architecture

### Key Components

#### 1. DiscomfortPort (The Foundation)
DiscomfortPort is the cornerstone node that enables dynamic I/O injection and extraction. It operates in three modes, automatically determined by its connections:

- **INPUT mode**: No incoming connections - serves as an injection point for external data.
- **OUTPUT mode**: No outgoing connections - serves as an extraction point, serializing data for capture via ComfyUI's history mechanism.
- **PASSTHRU mode**: Both incoming and outgoing connections - passes data through unchanged, without serialization or history capture.

Ports use a single output in the UI for simplicity, with internal logic handling mode-specific behavior.

#### 2. WorkflowTools (The Engine)
This utility class provides the core functionality:

- **discover_ports()**: Analyzes workflows to identify DiscomfortPorts, infer their types/modes, and compute execution order
- **stitch_workflows()**: Merges multiple workflow JSONs by renumbering nodes/links and creating cross-workflow connections
- **run_sequential()**: General-purpose runner that executes workflows with state preservation (90% complete; basic single-iteration runs work, full testing ongoing).

The engine relies on the WorkFlowContext for object I/O and manipulates prompts pre-execution to correctly insert/extract data to/from the nested workflows.

#### 3. WorkflowContext (The Data Store)
This utility class provides the core functionality:

- **save()**: saves an object to the data store for use in a subsequent workflow
- **load()**: loads an object from the data store to a workflow
- **export_data()**: saves an object to disk in a persistent way
- **list_keys()**: lists the all `unique_id` keys that are present in the data store
- **get_usage()**: displays the current memory usage of the data store
- **shutdown()**: gracefully shuts down the data store

The data store uses a hybrid storage strategy (in-RAM for speed, fallback to on-disk if required).

This class is designed to be instantiated for each run and is best used as a context manager (`with` statement). At any point in time, there should be two WorkflowContext instances running:
- The orchestrator instance, which holds the context for *all* runs and ensures the context is preserved across the run. It is best used as a context manager using a `with` statement that encapsulates the whole logic.
- The nested instance, which holds the context for any single workflow run. It is created in the beginning of every nested workflow run. Once create, the nested instance receives a reference to the orchestrator context (called a "receipt"), loads/saves data from it accordingly, and then finally returns the receipt back to the orchestrator before the nested instance shuts down.


#### 4. ComfyConnector (The Instance Wrapper)

ComfyConnector is a singleton class that acts as a Python wrapper to ComfyUI and allows us to pass workflows directly to its API. By choosing to nest a second ComfyUI server from within the runtime of ComfyUI, we liberate ourselves from the need to monkey patch the main ComfyUI instance in order to run a separate workflow, completely obviating our validation issues.

Nesting two ComfyUI servers will also future-proof the code since we all it will take to run the run_sequential method is to pre-edit the workflow.json script to add to it the DiscomfortDataLoader node on top of the INPUT DiscomfortPorts. As the workflow.json structure is pretty solid at this point in time in the ComfyUI repository, this should make our code much less fragile to future developments of ComfyUI.

ComfyConnector offers 5 handy methods:

- **create()**, which instantiates the singleton and starts a ComfyUI server.
- **upload_data()**, to save to ComfyUI's folders any required data (images, models etc) for execution. To avoid memory bloating, uploads that are flagged as ephemeral are automatically deleted when the server is killed.
- **run_workflow()**, to queue a workflow.json for execution on the ComfyUI server managed by ComfyConnector, receiving its history object in the end (which contains the outputs of the workflow run).
- **kill_api()**, to kill the nested ComfyUI server upon the end of the workflow run.
- **get_prompt_from_workflow()**, to generate a ComfyUI-compatible prompt JSON from a workflow JSON. 

(Note: ComfyConnector is a standalone class. It does not depend on the rest of Discomfort and may be useful to anyone that needs to automatically launch, kill, queue workflows, or otherwise manage ComfyUI instances.)


#### 4. Internal Nodes (NOT user-facing, but used internally by the Discomfort code)
- **DiscomfortContextSaver**: this is the node responsible for SAVING data TO context, ensuring a stateful workflow run.
- **DiscomfortContextLoader**:  this is the node responsible for LOADING data FROM context, ensuring a stateful workflow run.

The Internal Nodes replace DiscomfortPorts at runtime, simply by changing the class of the node inside the prompt JSON that will be sent for ComfyUI processing.

## 🚨 Current Issues

### The Core Problem: Integrating `WorkflowContext` into `run_sequential()`

The `run_sequential()` method has been refactored to use pre-prompt manipulation and a nested ComfyUI server, resolving earlier validation and injection failures. It now successfully runs for tested cases, but it still is not relying on `WorkflowContext` for data I/O.

See the "WORKFLOW_CONTEXT_INTEGRATION.md" for the plan to finalize and test.

## 🚀 Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/fidecastro/comfyui-discomfort.git discomfort
```

2. Install dependencies:
```bash
cd discomfort
pip install -r requirements.txt
```

3. Restart ComfyUI

## 📖 Usage

### Example (What Works)

```python
from custom_nodes.discomfort.workflow_tools import DiscomfortWorkflowTools

tools = DiscomfortWorkflowTools()

# This works - discover ports in a workflow
ports = tools.discover_ports("workflow.json")
print(f"Found {len(ports)} DiscomfortPorts")

# This works - stitch multiple workflows
merged = tools.stitch_workflows(["workflow1.json", "workflow2.json"])

# This works for basic cases - single-iteration execution
result = await tools.run_sequential(
    workflow_paths=["workflow.json"],
    inputs={"port1": your_data},  # Map unique_ids to input values
    iterations=1,
    use_ram=True
)
print(result)  # Outputs extracted via unique_ids
```

(Other examples to be added soon)

## 📚 Documentation

- `README.md`: Comprehensive technical overview, current issues, testing instructions
- `CURRENT_ISSUES.md`: Main development task at hand

## 🔧 Technical Details

### WorkflowContext Storage Strategy
- **In-RAM (Default)**: Fast, using inline JSON for small-to-medium data.
- **In-Disk (Fallback)**: For large objects, saving to temp files with automatic cleanup.

Execution uses a nested ComfyUI server via ComfyConnector for isolation, with pre-injection of DiscomfortDataLoader nodes for inputs.

## 🎯 Vision

Once the core execution mechanism is fully tested and stable, Discomfort will enable:
- **Iterative refinement pipelines**: Progressively improve outputs
- **Conditional workflows**: Branch execution based on intermediate results
- **State machines**: Complex multi-stage processing with memory
- **Dynamic workflow composition**: Build workflows programmatically
- **Batch processing**: As a special case of loops with independent iterations
- **Distributed Execution**: Support for multi-machine setups
- **Caching**: Smart caching of intermediate results
- **Visual Debugging**: UI tools to visualize data flow between iterations
- **Programming language for ComfyUI**: enable the complete instantiation and execution of ComfyUI workflows, including recursion and conditionals, exclusively via Python.

## 🧪 Testing

The best testing environment is ComfyUI itself, with test nodes created for that purpose, that are run within the UI itself. Focus on incremental validation: Start with single workflows, progress to stitched/multi-iteration, and stress-test with large data.
