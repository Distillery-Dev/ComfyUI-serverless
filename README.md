# Discomfort: Control ComfyUI with Python

![alt text](images/logo_512.png)

You have an awesome collection of workflows. Individually they work great, but making them work *together* -- deciding which ones to use, chaining their results, iterating workflows, etc -- *is a pain*.

You spend 95% of your time making Comfy spaghetti instead of just creating. _What if you could just write script that would tell ComfyUI exactly what it must do for you?_

I built Discomfort to resolve this. Discomfort is a ComfyUI extension that **allows ComfyUI workflows to be fully run via code**.

- **Loops! Conditionals!** Discomfort enables stateful run of ComfyUI workflows, exposing its variables in Python and allowing them to be fully manipulated via code. 

- **No more spaghetti!** Discomfort also allows users to **utilize partial workflows** (ex: a workflow that only contains sampling logic, another that only loads ControlNet, etc.), obviating the need to manually stitch workflow parts.

- **It's EASY!** Discomfort is 100% built on Python and is designed to be easy to use and learn.

- **It's FREE!** What's more, Discomfort does not touch ComfyUI's core code and is fully compatible with ComfyUI's license; it fully leverages on ComfyUI's execution logic and is expected to work stably with future versions of ComfyUI.

##  Current Status: ✅ Core Functionality Operational

While we're still in alpha stage, **Discomfort is fully operational**. We have achieved successful executions in basic and more advanced test cases, but full robustness in all scenarios is yet to be tested. There is still plenty of opportunity to improve the code.

All examples in this page work well. Test nodes like DiscomfortTestRunner successfully execute workflows via `Discomfort.run()`, including data injection/extraction. Minimal workflows (e.g., test_server.json) and more complex ones (e.g., image extension and upscaling pipelines, handling of 'ref' and 'val' variables, composition of complex workflows using partial/unrunnable workflows) complete with correct outputs.


## 🚀 Usage

### Step 1 - Install Discomfort

This is just a Python extension of ComfyUI. Just add it as you would any other custom node:

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

### Step 2 - Creating your Discomfort workflows

Discomfort workflows are just regular ComfyUI workflows, but with added DiscomfortPort nodes to signal their inputs and outputs.

To prepare a Discomfort workflow: 
1. Open your workflow
2. Insert a DiscomfortPort node
3. Connect it to whatever you'd like to handle via code
4. Write a **unique_id** inside the DiscomfortPort node
5. Repeat 2-4 until all the nodes you'd like to control are connected to 

**IMPORTANT:**

(a) Make sure connect DiscomfortPorts in the correct way: 
- if you're passing an input via code, then the DiscomfortPort is an INPUT for your workflow and therefore it should START your workflow (connect things on its right side).
- if you're collecting an output to code, then the DiscomfortPort is an OUTPUT for your workflow and therefore it should END your workflow (connect things on its left side).

(b) Discomfort handles partial workflows. This means you can extract a group of nodes from a workflow, add some DiscomfortPorts to it and then use Discomfort to stitch all your workflows together. Stop building spaghetti workflows!

(c) DO NOT connect things to both sides of a DiscomfortPort. If you do that, then it becomes an inactive PASSTHRU node and will NOT interact with Discomfort. This is important for Discomfort's architecture and must be kept this way.

(d) Remember: Discomfort is designed to handle **any type of input/output**. Want to manage a model? The prompt? Save latents? Handle some obscure command in a random node? Go nuts!

### Step 3 - Running your Discomfort script

The new `Discomfort` class provides a clean, unified API for all ComfyUI automation. All you need to do is initiate a context, save to/load from it, and then call `Discomfort.run()` with your Discomfort workflows.

See examples below:

```python
# Simple Test Script: add this to the ComfyUI folder and run
# In this simple example, a test workflow iterates over CFG and seed

import asyncio
from custom_nodes.discomfort.discomfort import Discomfort

async def main():

    discomfort = await Discomfort.create()

    with discomfort.Context() as context:
        cfg = 4.0
        seed = 42069
        for i in range(8):
            print(f"--- Iteration {i+1}: Running with CFG = {cfg:.1f} and SEED = {seed} ---")
            cfg = cfg + i*0.2
            seed = seed + i
            context.save("cfg", cfg)
            context.save("seed", seed)
            await discomfort.run(["custom_nodes/discomfort/discomfort_test.json"], context=context)
    await discomfort.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

```python
# Test Script: add this to the ComfyUI folder and run
# In this example, we run different seeds on an img2img workflow

import asyncio
from custom_nodes.discomfort.discomfort import Discomfort

async def main():

    discomfort = await Discomfort.create()
    image = discomfort.Tools.open_image_as_tensor("custom_nodes/discomfort/support/test_woman.png")

    with discomfort.Context() as context:
        seed = 1000
        context.save("input_image", image)
        for i in range(10):
            context.save("seed", seed)
            print(f"--- Iteration {i+1}: Running with SEED = {context.load("seed")} ---")
            seed = seed + i            
            await discomfort.run(["custom_nodes/discomfort/support/discomfort_test2.json"], context=context)
        await discomfort.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

```python
# Test Script: add this to the ComfyUI folder and run
# In this example, we stitch workflows first, run the img2img workflow, then save to disk
import asyncio
from custom_nodes.discomfort.discomfort import Discomfort

async def main():
    discomfort = await Discomfort.create()
    image = discomfort.Tools.open_image_as_tensor("test_woman.png")
    prompt = "A beautiful scifi woman with long blonde hair and blue eyes, masterpiece"
    model_name = "mohawk.safetensors"
    lora_name = "scifixl.safetensors"

    load_model_and_lora_workflow = "custom_nodes/discomfort/support/sdxl_load_model_and_lora.json"
    latent_from_image_workflow = "custom_nodes/discomfort/support/latent_from_input_image.json"
    sampler_workflow = "custom_nodes/discomfort/support/sdxl_run_ksampler.json"
    workflows_to_stitch = [load_model_and_lora_workflow, latent_from_image_workflow, sampler_workflow]
    inputs = {
        "prompt": prompt,
        "model_name": model_name,
        "lora_name": lora_name,
        "input_image": image,
        "lora_strength": 0.8,
        "denoise": 0.5
    }
    stitched_workflow = discomfort.Tools.stitch_workflows(workflows_to_stitch)["stitched_workflow"]

    with discomfort.Context() as context:
        await discomfort.run([stitched_workflow], inputs=inputs, context=context) # Run the full workflow
        discomfort.Tools.save_comfy_image_to_disk(context.load("output_image"), f"output_image.png") # Save the output image
        await discomfort.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Example

```python
# ----------------------------------------------------------------------------------------------------
# Example of code with iterations, conditionals, etc. using 4 different workflows, send to/collect 
# from context models, loras, clip, latent, image and float variables, using partial workflows.
# Fully functional. Run it from the ComfyUI folder.
# ----------------------------------------------------------------------------------------------------
import asyncio
from custom_nodes.discomfort.discomfort import Discomfort

async def main():
    discomfort = await Discomfort.create()
    image = discomfort.Tools.open_image_as_tensor("test_woman.png")
    prompt = "A beautiful scifi woman with long blonde hair and blue eyes, masterpiece"
    model_name = "mohawk.safetensors"
    lora_name = "scifixl.safetensors"

    load_model_and_lora_workflow = "custom_nodes/discomfort/support/sdxl_load_model_and_lora.json"
    latent_empty_workflow = "custom_nodes/discomfort/support/latent_1024x1024_empty.json"
    latent_from_image_workflow = "custom_nodes/discomfort/support/latent_from_input_image.json"
    sampler_workflow = "custom_nodes/discomfort/support/sdxl_run_ksampler.json"

    with discomfort.Context() as context:
        # Save the initial parameters needed by the first workflow
        context.save("prompt", prompt)
        context.save("model_name", model_name)
        context.save("lora_name", lora_name)
        
        for i in range(10):
            context.save("lora_strength", i * 0.1)
            # This populates the context with "model", "clip", and "vae" objects.
            await discomfort.run([load_model_and_lora_workflow], context=context)
            if i % 2 == 0: # empty latent
                denoise = 1
                await discomfort.run([latent_empty_workflow], context=context)
            else: # img2img latent                
                denoise = 0.5
                context.save("input_image", image)
                await discomfort.run([latent_from_image_workflow], inputs={"input_image": image}, context=context)
            await discomfort.run([sampler_workflow], inputs={"denoise": denoise}, context=context) # Run the KSampler
            discomfort.Tools.save_comfy_image_to_disk(context.load("output_image"), f"img_{i}.png") # Save the output image

        await discomfort.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```


## 🎯 Core Concept

ComfyUI's native execution model follows a Directed Acyclic Graph (DAG) pattern, executing nodes once in topological order. While powerful for single-pass workflows, this model doesn't natively support:
- Iterative refinement workflows
- Conditional execution paths  
- State preservation across iterations
- Dynamic workflow composition
- Programmatic access and management of ComfyUI instances

Discomfort addresses these limitations by introducing an execution layer on top of ComfyUI's DAG model, using pre-execution graph manipulation and a self-managed ComfyUI server for isolated runs, as well as a simple data store that handles context throughout the run.

Discomfort was designed to be trivially easy for anyone to use it. The only thing a user is required to know is how to add DiscomfortPorts to their existing workflows, and a microscopic bit of Python to write the execution code.


## 🏗️ Architecture

### Key Components

#### 1. DiscomfortPort (I/O Nodes)
DiscomfortPort is the cornerstone node that enables dynamic I/O injection and extraction. It operates in three modes, automatically determined by its connections:

- **INPUT mode**: No incoming connections - serves as an injection point for external data.
- **OUTPUT mode**: No outgoing connections - serves as an extraction point, serializing data for capture for the Context.
- **PASSTHRU mode**: Both incoming and outgoing connections - passes data through unchanged.

Ports use a single output in the UI for simplicity, with internal logic handling mode-specific behavior.

#### 2. Tools Class
This utility class provides the core functionality:

- **discover_ports()**: Analyzes workflows to identify DiscomfortPorts, infer their types/modes, and compute execution order
- **stitch_workflows()**: Merges multiple workflow JSONs by renumbering nodes/links and creating cross-workflow connections
- plus, most of the internal methods that are integral for `Discomfort.run()` to work

`Discomfort.run()` relies on the WorkFlowContext for object I/O and manipulates prompts pre-execution to correctly insert/extract data to/from the workflows.

#### 3. Context Class (The Data Store)
This utility class provides the core functionality:

- **save()**: saves an object to the data store for use in a subsequent workflow
- **load()**: loads an object from the data store to a workflow
- **export_data()**: saves an object to disk in a persistent way
- **list_keys()**: lists the all `unique_id` keys that are present in the data store
- **get_usage()**: displays the current memory usage of the data store
- **shutdown()**: gracefully shuts down the data store

The data store uses a hybrid storage strategy (in-RAM for speed, fallback to on-disk if required).

This class is designed to be instantiated for each run and is best used as a context manager (`with` statement). See examples above.

#### 4. Worker Class (The Instance Wrapper)

The worker is an improved version of ComfyConnector, which is a singleton class that acts as a Python wrapper to ComfyUI and allows us to pass workflows directly to its API. This allows us to change variables at runtime *without actually messing with ComfyUI's runtime* -- we just run workflows (or even workflow segments) one at a time. ComfyConnector offers five handy methods:

- **create()**, which instantiates the singleton and starts a ComfyUI server.
- **upload_data()**, to save to ComfyUI's folders any required data (images, models etc) for execution. To avoid memory bloating, uploads that are flagged as ephemeral are automatically deleted when the server is killed.
- **run_workflow()**, to queue a workflow.json for execution on the ComfyUI server managed by ComfyConnector, receiving its history object in the end (which contains the outputs of the workflow run). Crucially, it accepts both WORKFLOW or PROMPT JSON objects as inputs.
- **kill_api()**, to kill the managed ComfyUI server upon the end of the workflow run.
- **get_prompt_from_workflow()**, to generate a ComfyUI-compatible prompt JSON from a workflow JSON. 

(Note: ComfyConnector is a standalone class. It does not depend on the rest of Discomfort and may be useful to anyone that needs to automatically launch, kill, queue workflows, or otherwise manage ComfyUI instances.)

#### 5. Internal Nodes (NOT user-facing, but used internally by the Discomfort code)
- **DiscomfortContextSaver**: this is the node responsible for SAVING data TO context, ensuring a stateful workflow run.
- **DiscomfortContextLoader**:  this is the node responsible for LOADING data FROM context, ensuring a stateful workflow run.

The Internal Nodes replace DiscomfortPorts at Discomfort's runtime, simply by changing the class of the node inside the prompt JSON that will be sent for ComfyUI processing.

##$ 🔧 Other technical details/discussion

1. The code structure also allows ComfyConnector to be called from inside a ComfyUI run by a custom node -- creating a nested ComfyUI instance that can run workflows inside ComfyUI itself, with the master ComfyUI instance acting as an orchestrator UI. By nesting a second ComfyUI server from within the runtime of ComfyUI, we liberate ourselves from the need to monkey patch the main ComfyUI instance in order to run a separate workflow. This allows loops, conditionals, importing of workflows etc. to be run within the ComfyUI interface itself.

2. Variables managed by Discomfort's Context are called by their respective identifiers, `unique_id`. These variables are the heart of the stateful run of Discomfort. Each `unique_id` may be passed to context in one of two possibilities:
- **pass-by-value ("val")**: the default setting. Those are variables that are directly saved to context.
- **pass-by-reference ("ref")**: Those variables are passed to context by saving to context the *minimal workflow.json that leads to it*. This minimal workflow is then stitched to the incoming workflow.json objects it must connect to, using the `stitch_workflows()` method.

3. At any point in time, there should be two Context instances running:
- The master instance, which holds the context for *all* runs and ensures the context is preserved across the run. It is best used as a context manager using a `with` statement that encapsulates the whole logic.
- The worker instance, which holds the context for any single workflow run. It is created in the beginning of every workflow run. Once created, the worer instance receives a reference to the master context (called a "receipt"), loads/saves data from it accordingly, and then finally returns the receipt back to the master before the worker instance shuts down.
Context is passed to/from both instances by means of a `receipts.json` that is saved by an instance and subsequently loaded by the next instance handling the context.


## 🎯 Vision

Discomfort will enable things like:
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

Discomfort is still in **alpha** stage.

### What IS Working ✅
- **Discomfort Class**: Yes! MAKE COMFY GREAT AGAIN!!! _See the examples above._
- **ComfyUI instance management**: The `ComfyConnector` manages the ComfyUI instance appropriately and can be used as a standalone ComfyUI wrapper.
- **Workflow stitching**: `stitch_workflows()` can merge multiple workflow JSONs by renumbering nodes/links, preserving connections, and enabling cross-workflow data flow via shared unique IDs. Most critically, it underscores the whole logic of pass-by-reference variables, which is working well. It can be used as a standalone method for the odd user that wishes to use DiscomfortPorts for that purpose.

### What may NOT be Working ❌
- **Logging segregation**: Discomfort logs are currently interleaved with ComfyUI's terminal output, making debugging harder in verbose scenarios.
- **Error handling**: The code is still brittle. Errors, when they do occur, usually require a full restart. The error handling logic must be thoroughly reviewed and improved so that it fails gracefully.
- **Possible memory leaks on Context**: The WorkflowContext object is still in alpha stage. There may be memory leaks, and there potentially are unaddressed scenarios to consider (ex: temp folder reaching maximum size on Linux machines).
- **Large-scale scenarios**: Handling ultra-large Discomfort scripts, multiple inputs/outputs per workflow, and disk fallback for memory-intensive cases need thorough testing.
- **Edge cases**: Potential issues with async timing, namespace conflicts, or validation in complex graphs remain possible until comprehensive testing is complete.


## 📚 Documentation

- `README.md`: Comprehensive technical overview, current issues, testing instructions
- **Discomfort Scripts and Workflow Library**: COMING SOON


## 🧪 Testing

See the support folder for a few testing workflows.
For simple testing: use the DiscomfortTestRunner inside a running ComfyUI instance.