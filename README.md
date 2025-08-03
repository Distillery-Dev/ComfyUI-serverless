# Discomfort: Control ComfyUI with Python

![alt text](images/logo_512.png)

**Tired of building "Comfy spaghetti"? Discomfort is a ComfyUI extension that lets you control your workflows with the power of Python, enabling loops, conditionals, and stateful execution.**

Spend less time wiring nodes and more time creating. With Discomfort, you can write simple scripts to automate complex image generation tasks, turning your workflows into reusable, programmable components.

  - **Loops and Conditionals**: Iterate on your creations, run parameter sweeps, and implement complex logic that's impossible with nodes alone.
  - **Eliminate "Comfy Spaghetti"**: Break down large workflows into smaller, reusable parts and stitch them together with code.
  - **Easy to Learn**: Built on Python, Discomfort is designed to be intuitive and easy to pick up.
  - **Free and Open Source**: Discomfort is fully compatible with ComfyUI's license and doesn't modify any core code.


_**Discomfort's mission is to 10x the development and deployment speeds of any complex ComfyUI pipeline.**_

-----

## ✅ Current Status: Alpha

Discomfort is currently in the **alpha** stage but is fully operational. The core functionality is complete and has been tested in a variety of scenarios. However, as an alpha release, there may still be bugs and opportunities for improvement.

-----

## 🚀 Quick Start

### Installation

1.  Clone this repository into your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/fidecastro/comfyui-discomfort.git discomfort
    ```
2.  Install the required dependencies:
    ```bash
    cd discomfort
    pip install -r requirements.txt
    ```
3.  Restart ComfyUI.

-----

### Tutorials & How to Use

\<video width="100%" controls\>
\<source src="[https://github.com/fidecastro/discomfort-docs/raw/main/static/videos/tutorial-1-running-a-workflow.mp4](https://www.google.com/search?q=https://github.com/fidecastro/discomfort-docs/raw/main/static/videos/tutorial-1-running-a-workflow.mp4)" type="video/mp4" /\>
\</video\>

\<video width="100%" controls\>
\<source src="[https://github.com/fidecastro/discomfort-docs/raw/main/static/videos/tutorial-2-using-partial-workflows.mp4](https://www.google.com/search?q=https://github.com/fidecastro/discomfort-docs/raw/main/static/videos/tutorial-2-using-partial-workflows.mp4)" type="video/mp4" /\>
\</video\>

\<video width="100%" controls\>
\<source src="[https://github.com/fidecastro/discomfort-docs/raw/main/static/videos/tutorial-3-a-simple-loop-on-comfy.mp4](https://www.google.com/search?q=https://github.com/fidecastro/discomfort-docs/raw/main/static/videos/tutorial-3-a-simple-loop-on-comfy.mp4)" type="video/mp4" /\>
\</video\>

-----

### Your First Discomfort Script

Here's a simple example of how to run a workflow that iterates over different CFG and seed values:

```python
# Save this script in your ComfyUI root folder and run it
import asyncio
from custom_nodes.discomfort.discomfort import Discomfort

async def main():
    # Initialize Discomfort
    discomfort = await Discomfort.create()

    with discomfort.Context() as context:
        cfg = 4.0
        seed = 42069
        for i in range(8):
            print(f"--- Iteration {i+1}: Running with CFG = {cfg:.1f} and SEED = {seed} ---")
            cfg += i * 0.2
            seed += i
            context.save("cfg", cfg)
            context.save("seed", seed)
            await discomfort.run(["custom_nodes/discomfort/examples/workflows/discomfort_test1.json"], context=context)

    # Shutdown the Discomfort server
    await discomfort.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

See the [examples/README.md] page and the full documentation for additional information.

-----

## 🎯 Core Concepts

Discomfort introduces a layer of programmatic control on top of ComfyUI's existing graph-based execution model. This is achieved through a few key components:

### DiscomfortPorts

These are special nodes that you add to your ComfyUI workflows to create I/O points for your Python scripts. They have three modes, which are determined automatically based on their connections:

  - **INPUT**: No incoming connections. Injects data from your Python script into the workflow.
  - **OUTPUT**: No outgoing connections. Extracts data from the workflow and sends it to your Python script.
  - **PASSTHRU**: Both incoming and outgoing connections. The node is inactive and simply passes data through.

### The `Discomfort` Class

This is the main entry point for all of Discomfort's functionality. It provides a simple API for running workflows and managing their execution context.

### The `Context` Class

The `Context` class is a data store that allows you to save and load data between workflow executions. This is what enables stateful, iterative workflows.

### The `ComfyConnector` Class

The `ComfyConnector` class (subclassed as `Discomfort.Worker`) is the engine of Discomfort, automatically operating an isolated ComfyUI instance. It handles its launch, shutdown, and workflow queuing.

(Note: ComfyConnector is a standalone class. It does not depend on the rest of Discomfort and may be useful to anyone that needs to automatically launch, kill, queue workflows, or otherwise manage ComfyUI instances.)

-----

## 🏗️ Why Discomfort?

ComfyUI's native execution model follows a Directed Acyclic Graph (DAG) pattern, executing nodes once in topological order. While powerful for single-pass workflows, this model doesn't natively support things like iterations, backpropagation of data, or conditional execution paths.

Discomfort addresses these limitations by introducing an execution layer on top of ComfyUI's DAG model, using pre-execution graph manipulation and a self-managed ComfyUI server for isolated runs, as well as a simple data store that handles context throughout the run.

Discomfort's architecture is designed to be simple and non-intrusive:

1.  **Self-Managed ComfyUI Server**: Discomfort starts and manages its own ComfyUI instance in the background. This ensures that it doesn't interfere with your regular ComfyUI sessions.
2.  **Pre-Execution Graph Manipulation**: Before a workflow is executed, Discomfort modifies the graph to inject and extract data through the `DiscomfortPort` nodes.
3.  **Intelligent Data Handling**: Discomfort can handle different types of data, passing large objects like models by reference to avoid unnecessary memory overhead.

Discomfort was designed to be trivially easy for anyone to use it. The only thing a user is required to know is how to add DiscomfortPorts to their existing workflows, and a little bit of Python to write the execution code.

The vision for Discomfort is to enable things like:
- **Programming language for ComfyUI**: complete instantiation and execution of ComfyUI workflows, including recursion and conditionals, directly via Python
- **Iterative refinement pipelines**: Progressively improve outputs
- **Conditional workflows**: Branch execution based on intermediate results
- **State machines**: Complex multi-stage processing with memory
- **Dynamic workflow composition**: Build workflows programmatically
- **Batch processing**: As a special case of loops with independent iterations

-----

## 🚨 Known Issues

  - **Logging**: Discomfort's log messages are currently mixed with ComfyUI's output, which can make debugging difficult.
  - **Error Handling**: Error handling is still being improved. In some cases, an error may require a full restart.
  - **Memory Management**: The `WorkflowContext` is still in the alpha stage and may have memory leaks in some edge cases.