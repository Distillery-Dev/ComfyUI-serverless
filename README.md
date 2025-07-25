# ComfyUI Discomfort Extension

You have an awesome collection of workflows, carefully crafted and curated. Individually they work great, but making them work *together* -- deciding which ones to use, chaining their results, iterating workflows, etc -- is a pain. You have to open a workflow, run it manually, then decide which workflow to use next, run it manually again, check its results, etc...

What if you could just write script that would tell ComfyUI exactly what it must do for you?

I built Discomfort to resolve this. Discomfort is a ComfyUI extension that **allows ComfyUI workflows to be fully run via code**.

(1) Discomfort enables **stateful run of ComfyUI workflows**, exposing its variables in Python and allowing them to be fully manipulated via code. 

(2) Discomfort allows users to **create conditionals and iterations** by controlling ComfyUI runs with a robust ComfyUI wrapper and through some clever handling the workflow JSON objects.

(3) Discomfort also allows users to **utilize partial workflows** (ex: a workflow that only contains sampling logic, another that only loads ControlNet, etc.), obviating the need to manually stitch workflow parts.

(4) Discomfort **does not touch ComfyUI's core code** and is fully compatible with ComfyUI's license; it fully leverages on ComfyUI's execution logic and is expected to work stably with future versions of ComfyUI.

Discomfort is 100% built on Python and is designed to be easy to use and learn.



## ⚠️ Current Status: Core Functionality Operational

**IMPORTANT**: The core `run_sequential()` functionality is **working**. While we have achieved successful executions in basic test cases, full robustness across all scenarios is yet to be tested.

### What IS Working ✅
- **DiscomfortPort nodes**: Individual nodes function correctly in ComfyUI workflows, supporting INPUT (data injection), OUTPUT (data extraction), and PASSTHRU (data propagation) modes. Ports handle type-agnostic data passing, with serialization/deserialization for outputs.
- **Port discovery**: `discover_ports()` successfully analyzes workflows, identifies DiscomfortPorts, classifies their modes (INPUT/OUTPUT/PASSTHRU), infers data types from connections, and computes topological execution order.
- **ComfyUI instance management**: The `ComfyConnector` manages the ComfyUI instance appropriately. 
- **Workflow stitching**: `stitch_workflows()` can merge multiple workflow JSONs by renumbering nodes/links, preserving connections, and enabling cross-workflow data flow via shared unique IDs. Most critically, it underscores the whole logic of pass-by-reference variables, which is working well.
- **Basic testing**: Test nodes like DiscomfortTestRunner successfully execute workflows via `run_sequential()`, including data injection/extraction. Minimal workflows (e.g., test_server.json) and more complex ones (e.g., image extension and upscaling pipelines, handling of 'ref' and 'val' variables) complete with correct outputs.

### What may NOT be Working ❌
- **Unintuitive structure**: The module structure is unintuitive and must be better fleshed out, ideally by subclassing them all into a single Discomfort class.
- **Logging segregation**: Discomfort logs are currently interleaved with ComfyUI's terminal output, making debugging harder in verbose scenarios.
- **Error handling**: The code is still too brittle. Errors, when they do occur, usually require a full restart. The error handling logic must be thoroughly reviewed and improved so that it fails gracefully.
- **Memory leaks or mishandling on Context**: The WorkflowContext object is still in alpha stage and may not be deemed fully tested. There may be memory leaks, and there potentially are unaddressed scenarios to consider (ex: temp folder reaching maximum size on Linux machines).
- **Large-scale scenarios**: Handling ultra-large models (e.g., GB-scale tensors) between orchestrator and nested server, multiple inputs/outputs per workflow, and disk fallback for memory-intensive cases need thorough testing.
- **Edge cases**: Potential issues with async timing, namespace conflicts, or validation in complex graphs remain possible until comprehensive testing is complete.


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

#### 1. DiscomfortPort (I/O Nodes)
DiscomfortPort is the cornerstone node that enables dynamic I/O injection and extraction. It operates in three modes, automatically determined by its connections:

- **INPUT mode**: No incoming connections - serves as an injection point for external data.
- **OUTPUT mode**: No outgoing connections - serves as an extraction point, serializing data for capture via ComfyUI's history mechanism.
- **PASSTHRU mode**: Both incoming and outgoing connections - passes data through unchanged, without serialization or history capture.

Ports use a single output in the UI for simplicity, with internal logic handling mode-specific behavior.

#### 2. WorkflowTools
This utility class provides the core functionality:

- **discover_ports()**: Analyzes workflows to identify DiscomfortPorts, infer their types/modes, and compute execution order
- **stitch_workflows()**: Merges multiple workflow JSONs by renumbering nodes/links and creating cross-workflow connections
- **run_sequential()**: General-purpose runner that executes workflows with state preservation (full testing ongoing).
- plus, all the internal methods that are integral for `run_sequential` to work

`run_sequential` relies on the WorkFlowContext for object I/O and manipulates prompts pre-execution to correctly insert/extract data to/from the workflows.

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
- The master instance, which holds the context for *all* runs and ensures the context is preserved across the run. It is best used as a context manager using a `with` statement that encapsulates the whole logic.
- The worker instance, which holds the context for any single workflow run. It is created in the beginning of every workflow run. Once created, the worer instance receives a reference to the master context (called a "receipt"), loads/saves data from it accordingly, and then finally returns the receipt back to the master before the worker instance shuts down.


#### 4. ComfyConnector (The Instance Wrapper)

ComfyConnector is a singleton class that acts as a Python wrapper to ComfyUI and allows us to pass workflows directly to its API. This allows us to change variables at runtime *without actually messing with ComfyUI's runtime* -- we just run workflows (or even workflow segments) one at a time. ComfyConnector offers five handy methods:

- **create()**, which instantiates the singleton and starts a ComfyUI server.
- **upload_data()**, to save to ComfyUI's folders any required data (images, models etc) for execution. To avoid memory bloating, uploads that are flagged as ephemeral are automatically deleted when the server is killed.
- **run_workflow()**, to queue a workflow.json for execution on the ComfyUI server managed by ComfyConnector, receiving its history object in the end (which contains the outputs of the workflow run). Crucially, it accepts both WORKFLOW or PROMPT JSON objects as inputs.
- **kill_api()**, to kill the managed ComfyUI server upon the end of the workflow run.
- **get_prompt_from_workflow()**, to generate a ComfyUI-compatible prompt JSON from a workflow JSON. 

(Note: ComfyConnector is a standalone class. It does not depend on the rest of Discomfort and may be useful to anyone that needs to automatically launch, kill, queue workflows, or otherwise manage ComfyUI instances.)


#### 5. Internal Nodes (NOT user-facing, but used internally by the Discomfort code)
- **DiscomfortContextSaver**: this is the node responsible for SAVING data TO context, ensuring a stateful workflow run.
- **DiscomfortContextLoader**:  this is the node responsible for LOADING data FROM context, ensuring a stateful workflow run.

The Internal Nodes replace DiscomfortPorts at runtime, simply by changing the class of the node inside the prompt JSON that will be sent for ComfyUI processing.


##$ 🔧 Other technical details

1. The code structure also allows ComfyConnector to be called from inside a ComfyUI run by a custom node -- creating a nested ComfyUI instance that can run workflows inside ComfyUI itself, with the master ComfyUI instance acting as an orchestrator UI. By nesting a second ComfyUI server from within the runtime of ComfyUI, we liberate ourselves from the need to monkey patch the main ComfyUI instance in order to run a separate workflow. This allows loops, conditionals, importing of workflows etc. to be run within the ComfyUI interface itself.

2. Variables managed by Discomfort's WorkflowContext are called by their respective identifiers, `unique_id`. These variables are the heart of the stateful run of Discomfort. Each `unique_id` may be passed to context in one of two possibilities:
- **pass-by-value ("val")**: the default setting. Those are variables that are directly saved to context.
- **pass-by-reference ("ref")**: Those variables are passed to context by saving to context the *minimal workflow.json that leads to it*. This minimal workflow is then stitched to the incoming workflow.json objects it must connect to, using the `stitch_workflows()` method.


## 🎯 Vision

Once the core execution mechanism is fully tested and stable, Discomfort will enable:
- **Programming language for ComfyUI**: enable the complete instantiation and execution of ComfyUI workflows, including recursion and conditionals, directly via Python.
- **Iterative refinement pipelines**: Progressively improve outputs
- **Conditional workflows**: Branch execution based on intermediate results
- **State machines**: Complex multi-stage processing with memory
- **Dynamic workflow composition**: Build workflows programmatically
- **Batch processing**: As a special case of loops with independent iterations
- **Distributed Execution**: Support for multi-machine setups
- **Caching**: Smart caching of intermediate results
- **Visual Debugging**: UI tools to visualize data flow between iterations


## 🚨 Current Issues

See the **CURRENT_ISSUES.md** for further information.


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


### Simple Examples that works

```python
from custom_nodes.discomfort.workflow_tools import DiscomfortWorkflowTools

tools = DiscomfortWorkflowTools()

# This works - discover the DiscomfortPort nodes in a workflow
ports = tools.discover_ports("workflow.json")
print(f"Found {len(ports)} DiscomfortPorts")

# This works - stitch multiple workflows (each with corresponding DiscomfortPorts for INPUT and OUTPUT)
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

### Example of WHAT DISCOMFORT WILL ALLOW (Discomfort class still TBD!)

```python
# ----------------------------------------------------------------------------------------------------
# Example of code that uses some image enhancement workflow, and another that asks a vision LLM to rate the quality of the image
# The idea is that the enhancement would only stop once the vision LLM is satisfied with its quality (>70%)
# ----------------------------------------------------------------------------------------------------
from comfyui-discomfort import Discomfort

discomfort = Discomfort.create()
image = Image.open("initial_image.png")
with discomfort.Context() as context: # create the context of the run
    context.save(image, "input_image") # save the initial_image.png to context as the "input_image" `unique_id`
    quality_index = 0 # initial value so the first iteration always runs
    while context.load("quality_index") < 0.7:
        discomfort.run(["workflow_that_improves_images.json"], inputs = {"input_image": image}, context = context) # Assume this workflow improves an image somehow
        image = context.load("output_image")  # loads the output image after the workflow run
        discomfort.run(["workflow_that_assesses_quality_of_image.json"], inputs = {"llm_prompt": "Rate the quality of this image from 0 to 100%.", "image_to_assess": image}, context = context) # Assume this workflow is calling a vision LLM and getting it to rate the image from 0-100%
        image = context.load("image_to_assess") # loads the image from the context for returning it OR for the next iteration
        if context.load("quality_index") > 0.7:
            return image
```



## 📚 Documentation

- `README.md`: Comprehensive technical overview, current issues, testing instructions
- `CURRENT_ISSUES.md`: Main development task at hand


## 🧪 Testing

Testing so far has been made within the environment of ComfyUI, with test nodes created for that purpose, that are run within the UI itself.
