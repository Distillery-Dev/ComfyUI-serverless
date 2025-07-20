# ComfyUI Discomfort Extension

A ComfyUI extension that enables **loop-based workflow execution** with dynamic data flow between iterations. Discomfort enables stateful run of ComfyUI workflows, allowing for conditionals and iterations to be made -- all without touching the ComfyUI original code.

## ⚠️ Current Status: Core Functionality Partially Operational

**IMPORTANT**: The core `run_sequential()` functionality is **sort of working**, but it remains in active development and testing. While we have achieved successful executions in basic test cases, full robustness across all scenarios is not yet achieved. IMPORTANT: the project is under a major refactoring -- refer to the CURRENT ISSUES section below.

### What IS Working ✅
- **DiscomfortPort nodes**: Individual nodes function correctly in ComfyUI workflows, supporting INPUT (data injection), OUTPUT (data extraction), and PASSTHRU (data propagation) modes. Ports handle type-agnostic data passing, with serialization/deserialization for outputs.
- **Port discovery**: `discover_ports()` successfully analyzes workflows, identifies DiscomfortPorts, classifies their modes (INPUT/OUTPUT/PASSTHRU), infers data types from connections, and computes topological execution order.
- **Workflow stitching**: `stitch_workflows()` can merge multiple workflow JSONs by renumbering nodes/links, preserving connections, and enabling cross-workflow data flow via shared unique IDs.
- **Supporting nodes**: DiscomfortFolderImageLoader and DiscomfortImageDescriber work independently for batch image loading and AI-based description generation.
- **Basic testing**: Test nodes like DiscomfortExtenderWorkflowRunner successfully execute workflows via `run_sequential()`, including data injection/extraction for single-iteration runs. Minimal workflows (e.g., test_server.json) and more complex ones (e.g., image extension pipelines) complete with correct outputs.
- **`run_sequential()` basics**: The method is a *WORK IN PROGRESS* but already executes workflows on a nested ComfyUI server, injects data into INPUT ports via pre-prompt manipulation (replacing with DiscomfortDataLoader), and extracts from OUTPUT ports (BUT via history inspection, NOT via the WorkflowContext object as it should).

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

Discomfort addresses these limitations by introducing a loop-enabled execution layer on top of ComfyUI's DAG model, using pre-execution graph manipulation and a nested ComfyUI server for isolated, repeatable runs.

## 🏗️ Architecture

### Key Components

#### 1. DiscomfortPort (The Foundation)
DiscomfortPort is the cornerstone node that enables dynamic I/O injection and extraction. It operates in three modes, automatically determined by its connections:

- **INPUT mode**: No incoming connections - serves as an injection point for external data.
- **OUTPUT mode**: No outgoing connections - serves as an extraction point, serializing data for capture via ComfyUI's history mechanism.
- **PASSTHRU mode**: Both incoming and outgoing connections - passes data through unchanged, without serialization or history capture.

Ports use a single output in the UI for simplicity, with internal logic handling mode-specific behavior.

#### 2. DiscomfortWorkflowTools (The Engine)
This utility class provides the core functionality:

- **discover_ports()**: Analyzes workflows to identify DiscomfortPorts, infer their types/modes, and compute execution order ✅
- **stitch_workflows()**: Merges multiple workflow JSONs by renumbering nodes/links and creating cross-workflow connections ✅
- **run_sequential()**: Executes workflows in a loop with state preservation (90% complete; basic single-iteration runs work, full testing ongoing)
- **serialize()/deserialize()**: Handles data persistence across iterations ✅

The engine relies on the WorkFlowContext for object I/O and manipulates prompts pre-execution to inject data loaders.

#### 3. WorkflowContext (The Data Store)
This utility class provides the core functionality:

- **save()**: saves an object to the data store for use in a subsequent workflow
- **load()**: loads an object from the data store to a workflow
- **export_data()**: saves an object to disk in a persistent way
- **list_keys()**: lists the all `unique_id` keys that are present in the data store
- **get_usage()**: displays the current memory usage of the data store
- **shutdown()**: gracefully shuts down the data store

The data store uses a hybrid storage strategy (in-RAM for speed, fallback to on-disk if required).


#### 4. The `ComfyConnector` Class (The Wrapper)

ComfyConnector is a singleton class that acts as a Python wrapper to ComfyUI and allows us to pass workflows directly to its API. By choosing to nest a second ComfyUI server from within the runtime of ComfyUI, we liberate ourselves from the need to monkey patch the main ComfyUI instance in order to run a separate workflow, completely obviating our validation issues.

Nesting two ComfyUI servers will also future-proof the code since we all it will take to run the run_sequential method is to pre-edit the workflow.json script to add to it the DiscomfortDataLoader node on top of the INPUT DiscomfortPorts. As the workflow.json structure is pretty solid at this point in time in the ComfyUI repository, this should make our code much less fragile to future developments of ComfyUI.

ComfyConnector offers 4 methods that will be helpful:

- **create()**, which instantiates the singleton.
- **upload_data()**, to add (via disk memory) any required data (images, models etc) to the nested ComfyUI for execution. To avoid memory bloating, uploads flagged as ephemeral are automatically deleted when the server is killed.
- **run_workflow()**, to queue the pre-edited workflow.json for execution on the nested ComfyUI, receiving its history object in the end (which should contain the outputs of the workflow run)
- **kill_api()**, to kill the nested ComfyUI server upon the end of the workflow run.


#### 4. Supporting Nodes
- **DiscomfortFolderImageLoader**: Loads images from folders as batched tensors ✅
- **DiscomfortImageDescriber**: Generates AI descriptions via OpenAI-compatible APIs ✅
- **DiscomfortLoopExecutor**: (Planned) User-facing node for configuring loop execution, iterations, and conditions.

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

### What You Can Do Now

1. **Individual DiscomfortPort nodes**: Add them to workflows and use them for data passing within a single execution. Ports auto-adapt to INPUT/OUTPUT/PASSTHRU based on connections.
2. **Port discovery**: Use `discover_ports()` to analyze workflows.
3. **Workflow stitching**: Use `stitch_workflows()` to merge multiple workflow JSONs.
4. **Supporting nodes**: Use DiscomfortFolderImageLoader and DiscomfortImageDescriber independently.
5. **Basic sequential execution**: Use `run_sequential()` for single-iteration runs on stitched or individual workflows, with data injection/extraction. (WARNING: run_sequential still WIP)

### What You Cannot Do Yet

- **Full loop execution**: Multi-iteration with state preservation is untested.
- **Conditional workflows**: Branching logic is planned but not implemented.
- **Advanced testing scenarios**: Large models, multiple I/O, and disk fallback need validation.

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

## 📚 Documentation

- `README.md`: Comprehensive technical overview, current issues, testing instructions
- `PLAN_FOR_LOOP_EXECUTOR_NODE.md`: Loop executor design
- `WORKFLOW_CONTEXT_INTEGRATION.md`: next step to implement

## 🔧 Technical Details

### Serialization Strategy
Must handle arbitrary data types using a unified format.
```json
{
  "type": "TORCH_TENSOR|STRING|JSON|CUSTOM|etc",
  "content": "serialized_data"
}
```
Must support Torch tensors (via torch.save/base64), primitives, JSON, and rely mostly on cloudpickle for I/O.

### Storage Strategy
- **In-RAM (Default)**: Fast, using inline JSON for small-to-medium data.
- **In-Disk (Fallback)**: For large objects, saving to temp files with automatic cleanup.

Execution uses a nested ComfyUI server via ComfyConnector for isolation, with pre-injection of DiscomfortDataLoader nodes for inputs.

## 🎯 Future Vision

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
