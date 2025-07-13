# ComfyUI Discomfort Extension

A ComfyUI extension that enables **loop-based workflow execution** with dynamic data flow between iterations. Originally conceived for batch image processing, Discomfort has evolved to support dependent iterative workflows where outputs from iteration N feed into iteration N+1.

## ⚠️ Current Status: Core Functionality Broken

**IMPORTANT**: The core `run_sequential()` functionality is currently **NOT WORKING**. While the individual components are implemented, the data injection mechanism that enables loop execution is failing. See the "Current Issues" section below for details.

### What IS Working ✅
- **DiscomfortPort nodes**: Individual nodes function correctly in ComfyUI workflows
- **Port discovery**: `discover_ports()` successfully analyzes workflows and identifies DiscomfortPorts
- **Workflow stitching**: `stitch_workflows()` can merge multiple workflow JSONs
- **Serialization**: `serialize()/deserialize()` handles data persistence across iterations
- **Supporting nodes**: DiscomfortFolderImageLoader and DiscomfortImageDescriber work independently
- **Basic testing**: Test scripts can run and validate individual components

### What is NOT Working ❌
- **`run_sequential()` method**: The core loop execution functionality is broken
- **Data injection**: Cannot inject data into INPUT ports during workflow execution
- **Loop execution**: Cannot run workflows in sequence with state preservation
- **Cross-iteration data flow**: Outputs from one iteration cannot feed into the next

## 🎯 Core Concept

ComfyUI's native execution model follows a Directed Acyclic Graph (DAG) pattern, executing nodes once in topological order. While powerful for single-pass workflows, this model doesn't natively support:
- Iterative refinement workflows
- Conditional execution paths  
- State preservation across iterations
- Dynamic workflow composition

Discomfort addresses these limitations by introducing a loop-enabled execution layer on top of ComfyUI's DAG model.

## 🏗️ Architecture

### Key Components

#### 1. DiscomfortPort (The Foundation)
DiscomfortPort is the cornerstone node that enables dynamic I/O injection and extraction. It operates in three modes, automatically determined by its connections:

- **INPUT mode**: No incoming connections - serves as an injection point
- **OUTPUT mode**: No outgoing connections - serves as an extraction point  
- **PASSTHRU mode**: Both incoming and outgoing connections - passes data through, exposing it only if no matching OUTPUT exists

#### 2. DiscomfortWorkflowTools (The Engine)
This utility class provides the core functionality:

- **discover_ports()**: Analyzes workflows to identify DiscomfortPorts, infer their types, and compute execution order ✅
- **stitch_workflows()**: Merges multiple workflow JSONs by renumbering nodes/links and creating cross-workflow connections ✅
- **run_sequential()**: Executes workflows in a loop with state preservation ❌ **BROKEN**
- **serialize()/deserialize()**: Handles data persistence across iterations ✅

#### 3. Supporting Nodes
- **DiscomfortFolderImageLoader**: Loads images from folders as batched tensors ✅
- **DiscomfortImageDescriber**: Generates AI descriptions via OpenAI-compatible APIs ✅
- **DiscomfortLoopExecutor**: (Planned) User-facing node for configuring loop execution

## 🚨 Current Issues

### The Core Problem: Data Injection Failure

The `run_sequential()` method is failing at the data injection step. The mechanism attempts to:

1. Store actual data in `_temp_execution_data[execution_id][uid]`
2. Inject validation placeholders into the prompt
3. Have DiscomfortPort nodes retrieve real data during execution

**Why it's failing**:
- **Validation vs Execution Context**: ComfyUI validates the entire workflow before execution, but placeholders must satisfy type validation without having the actual data
- **Execution Isolation**: ComfyUI may run nodes in isolated contexts, making global variables like `_temp_execution_data` inaccessible
- **Async Execution Model**: Data injection timing may be misaligned with ComfyUI's async execution model
- **Import Structure**: Namespace issues between modules could be preventing data access

### Technical Challenges

1. **Validation Placeholders**: The current placeholder structure may not be properly formatted for ComfyUI's validator
2. **Global State Access**: The DiscomfortPort's `process_port` method may not be correctly looking up data from the global store
3. **Execution Timing**: The synchronous wait loop might be missing execution completion signals

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

1. **Individual DiscomfortPort nodes**: Add them to workflows and use them for data passing within a single execution
2. **Port discovery**: Use `discover_ports()` to analyze workflows
3. **Workflow stitching**: Use `stitch_workflows()` to merge multiple workflow JSONs
4. **Supporting nodes**: Use DiscomfortFolderImageLoader and DiscomfortImageDescriber independently

### What You Cannot Do Yet

- **Loop execution**: The `run_sequential()` method is broken
- **Cross-iteration data flow**: Cannot pass outputs from one iteration to the next
- **Iterative workflows**: Cannot run workflows multiple times with state preservation

### Example (What Works)

```python
from custom_nodes.discomfort.workflow_tools import DiscomfortWorkflowTools

tools = DiscomfortWorkflowTools()

# This works - discover ports in a workflow
ports = tools.discover_ports("workflow.json")
print(f"Found {len(ports)} DiscomfortPorts")

# This works - stitch multiple workflows
merged = tools.stitch_workflows(["workflow1.json", "workflow2.json"])

# This DOES NOT WORK - loop execution is broken
# result = await tools.run_sequential(
#     workflow_paths=["workflow.json"],
#     inputs={"input_data": your_data},
#     iterations=5,
#     use_ram=True
# )
```

## 🧪 Testing

See `TEST_README.md` for detailed testing instructions. Two test approaches are available:

1. **Module Test** (`test_discomfort.py`): Direct module testing ✅
2. **API Test** (`test_simple.py`): HTTP API-based testing ✅

**Note**: These tests validate individual components but cannot test the broken `run_sequential()` functionality.

## 📚 Documentation

- `discomfort_doc_overview.md`: Comprehensive technical overview and current issues
- `discomfort_doc_DiscomfortLoopExecutor_high_level_logic.md`: Loop executor design
- `SOLUTION_SUMMARY.md`: Implementation details and attempted solutions
- `TEST_README.md`: Testing instructions and troubleshooting

## 🔧 Technical Details

### Serialization Strategy
Handles arbitrary ComfyUI data types using a unified format:
```json
{
  "type": "TORCH_TENSOR|STRING|JSON|CUSTOM|etc",
  "content": "serialized_data"
}
```

### Storage Strategy
- **In-RAM (Default)**: Fast, direct passing between iterations
- **In-Disk (Fallback)**: For large objects or when RAM is limited

## 🛠️ Next Steps to Fix

### 1. Revise Data Injection Mechanism
Instead of relying on global state and placeholders, consider:
- Using ComfyUI's built-in caching system (investigate `CacheSet` in execution.py)
- Implementing a custom execution context that ComfyUI recognizes
- Exploring ComfyUI's "hidden" inputs mechanism for data passing

### 2. Debug Current Implementation
Add extensive logging to trace:
- When data is stored in `_temp_execution_data`
- When DiscomfortPort attempts to retrieve it
- The exact validation errors being encountered
- The state of the execution context at each step

### 3. Alternative Approaches
Consider these architectural alternatives:
- **Workflow Preprocessing**: Modify the workflow JSON to embed data directly before execution
- **Custom Executor**: Extend ComfyUI's PromptExecutor rather than working around it
- **Node Chaining**: Use ComfyUI's native link system more directly, creating temporary nodes that hold data

### 4. Simplify Initial Implementation
Start with a minimal viable approach:
- Single iteration, single workflow
- Simple data types (just IMAGE or STRING)
- No conditional logic
- Extensive logging at every step

## 🎯 Future Vision

Once the core execution mechanism is working, Discomfort will enable:
- **Iterative refinement pipelines**: Progressively improve outputs
- **Conditional workflows**: Branch execution based on intermediate results
- **State machines**: Complex multi-stage processing with memory
- **Dynamic workflow composition**: Build workflows programmatically
- **Batch processing**: As a special case of loops with independent iterations

## 🤝 Contributing

This project represents a significant extension of ComfyUI's capabilities, transforming it from a single-pass execution engine to a full iterative computation platform. **Contributions are especially welcome to help fix the core execution issues!**

## 📄 License

This project is open source. Please check the license file for details. 