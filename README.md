# ComfyUI Discomfort Extension

A ComfyUI extension that enables **loop-based workflow execution** with dynamic data flow between iterations. Originally conceived for batch image processing, Discomfort has evolved to support dependent iterative workflows where outputs from iteration N feed into iteration N+1.

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

- **discover_ports()**: Analyzes workflows to identify DiscomfortPorts, infer their types, and compute execution order
- **stitch_workflows()**: Merges multiple workflow JSONs by renumbering nodes/links and creating cross-workflow connections
- **run_sequential()**: Executes workflows in a loop with state preservation
- **serialize()/deserialize()**: Handles data persistence across iterations

#### 3. Supporting Nodes
- **DiscomfortFolderImageLoader**: Loads images from folders as batched tensors
- **DiscomfortImageDescriber**: Generates AI descriptions via OpenAI-compatible APIs
- **DiscomfortLoopExecutor**: (Planned) User-facing node for configuring loop execution

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

### Basic Workflow

1. Add `DiscomfortPort` nodes to your workflow
2. Configure them with unique IDs and optional tags
3. Use `DiscomfortWorkflowTools` to execute with loop logic

### Example

```python
from custom_nodes.discomfort.workflow_tools import DiscomfortWorkflowTools

tools = DiscomfortWorkflowTools()
result = await tools.run_sequential(
    workflow_paths=["workflow.json"],
    inputs={"input_data": your_data},
    iterations=5,
    use_ram=True
)
```

## 🧪 Testing

See `TEST_README.md` for detailed testing instructions. Two test approaches are available:

1. **Module Test** (`test_discomfort.py`): Direct module testing
2. **API Test** (`test_simple.py`): HTTP API-based testing

## 📚 Documentation

- `discomfort_doc_overview.md`: Comprehensive technical overview
- `discomfort_doc_DiscomfortLoopExecutor_high_level_logic.md`: Loop executor design
- `SOLUTION_SUMMARY.md`: Implementation details and solution approach

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

## 🎯 Future Vision

Once the core execution mechanism is working, Discomfort will enable:
- **Iterative refinement pipelines**: Progressively improve outputs
- **Conditional workflows**: Branch execution based on intermediate results
- **State machines**: Complex multi-stage processing with memory
- **Dynamic workflow composition**: Build workflows programmatically
- **Batch processing**: As a special case of loops with independent iterations

## 🤝 Contributing

This project represents a significant extension of ComfyUI's capabilities, transforming it from a single-pass execution engine to a full iterative computation platform. Contributions are welcome!

## 📄 License

This project is open source. Please check the license file for details. 