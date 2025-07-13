# Testing Discomfort

## Quick Start

The easiest way to test Discomfort without import issues:

```bash
# 1. Start ComfyUI server (from ComfyUI root)
python main.py

# 2. In another terminal, run the simple API test
python custom_nodes/discomfort/test_simple.py
```

## Import Issues?

If you're getting `ModuleNotFoundError` errors, it's because ComfyUI has a specific way of setting up Python paths. Here are solutions:

### Solution 1: Use Module Execution
```bash
# From ComfyUI root directory:
python -m custom_nodes.discomfort.test_discomfort
```

### Solution 2: Use the API Test
The `test_simple.py` script uses HTTP API and avoids all import issues:
```bash
python custom_nodes/discomfort/test_simple.py
```

### Solution 3: Test in UI
1. Open ComfyUI in your browser
2. Add a DiscomfortTestRunner node
3. Connect an image input
4. Run the workflow

## Troubleshooting

### "No module named 'utils.install_util'"
This happens when trying to import ComfyUI modules outside the normal execution context. Use one of the solutions above.

### "ComfyUI server not running"
Make sure to start ComfyUI first:
```bash
python main.py
```

### "DiscomfortPort not found"
Make sure ComfyUI loaded the custom nodes. Check the console output when starting ComfyUI for any errors loading the Discomfort nodes.

## What the Tests Do

Both test scripts create a simple workflow:
1. DiscomfortPort (INPUT) with unique_id "test_input"
2. DiscomfortPort (OUTPUT) with unique_id "test_output"
3. A link connecting them

The test injects a random tensor into "test_input" and verifies it comes out at "test_output". 