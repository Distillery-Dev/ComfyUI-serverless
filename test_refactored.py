#!/usr/bin/env python3
"""
Test script for the refactored Discomfort run_sequential functionality.
Run from ComfyUI root directory with the server already running:
    python -m custom_nodes.discomfort.test_refactored
"""

def main():
    import sys
    import os
    
    # --- Start of fix ---
    # Add the ComfyUI root directory to the path
    comfyui_root = os.getcwd()
    if comfyui_root not in sys.path:
        sys.path.insert(0, comfyui_root)
    # --- End of fix ---
    
    # Ensure we're running from ComfyUI root
    if not os.path.exists("main.py") or not os.path.exists("server.py"):
        print("ERROR: This script must be run from the ComfyUI root directory")
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    # Import required modules
    import asyncio
    import torch
    import json
    import time
    import uuid
    import logging
    
    # Import our refactored module
    from custom_nodes.discomfort.workflow_tools import DiscomfortWorkflowTools
    from custom_nodes.discomfort.nodes_internal import DiscomfortDataLoader, clear_all_memory_data
    
    # Register the DiscomfortDataLoader node
    import nodes
    if 'DiscomfortDataLoader' not in nodes.NODE_CLASS_MAPPINGS:
        nodes.NODE_CLASS_MAPPINGS['DiscomfortDataLoader'] = DiscomfortDataLoader
        print("Registered DiscomfortDataLoader node")
    
    async def test_workflow():
        print("=== Testing Refactored Discomfort run_sequential ===\n")
        
        # Create a simple test workflow
        test_workflow = {
            "last_node_id": 3,
            "last_link_id": 2,
            "nodes": [
                {
                    "id": 1,
                    "type": "DiscomfortPort",
                    "pos": [100, 100],
                    "size": [200, 100],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [
                        {"name": "output", "type": "*", "links": [1], "slot_index": 0}
                    ],
                    "properties": {},
                    "widgets_values": ["test_input", "image"]
                },
                {
                    "id": 2,
                    "type": "ImageInvert",  # Using a built-in node for testing
                    "pos": [400, 100],
                    "size": [200, 100],
                    "flags": {},
                    "order": 1,
                    "mode": 0,
                    "inputs": [
                        {"name": "image", "type": "IMAGE", "link": 1}
                    ],
                    "outputs": [
                        {"name": "IMAGE", "type": "IMAGE", "links": [2], "slot_index": 0}
                    ],
                    "properties": {}
                },
                {
                    "id": 3,
                    "type": "DiscomfortPort",
                    "pos": [700, 100],
                    "size": [200, 100],
                    "flags": {},
                    "order": 2,
                    "mode": 0,
                    "inputs": [
                        {"name": "input_data", "type": "*", "link": 2}
                    ],
                    "outputs": [],
                    "properties": {},
                    "widgets_values": ["test_output", "image"]
                }
            ],
            "links": [
                [1, 1, 0, 2, 0, "*"],
                [2, 2, 0, 3, 0, "IMAGE"]
            ],
            "groups": [],
            "config": {},
            "extra": {},
            "version": 0.4
        }
        
        # Save test workflow
        test_path = "test_workflow_refactored_temp.json"
        with open(test_path, 'w') as f:
            json.dump(test_workflow, f, indent=2)
        
        try:
            # Create test data (a simple gradient image)
            test_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            # Create a gradient
            for i in range(64):
                for j in range(64):
                    test_tensor[0, i, j, 0] = i / 63.0  # Red channel
                    test_tensor[0, i, j, 1] = j / 63.0  # Green channel
                    test_tensor[0, i, j, 2] = (i + j) / 126.0  # Blue channel
            
            print(f"Input tensor shape: {test_tensor.shape}")
            print(f"Input tensor range: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")
            
            # Set up logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Clear any existing memory data
            clear_all_memory_data()
            
            # Run workflow with the refactored method
            tools = DiscomfortWorkflowTools()
            print("\nRunning workflow...")
            result = await tools.run_sequential(
                workflow_paths=[test_path],
                inputs={"test_input": test_tensor},
                iterations=1,
                use_ram=True
            )
            
            print("\nWorkflow completed!")
            print(f"Results: {list(result.keys())}")
            
            if "test_output" in result:
                output = result["test_output"]
                if isinstance(output, torch.Tensor):
                    print(f"Output tensor shape: {output.shape}")
                    print(f"Output tensor range: [{output.min():.3f}, {output.max():.3f}]")
                    
                    # Check if inversion worked (values should be inverted)
                    # Since ImageInvert does 1.0 - image, we expect inverted values
                    expected = 1.0 - test_tensor
                    if torch.allclose(expected, output, atol=1e-5):
                        print("\n✅ SUCCESS: Output matches expected inverted image!")
                    else:
                        print("\n❌ FAILURE: Output does not match expected inverted image")
                        print(f"Max difference: {torch.abs(expected - output).max():.6f}")
                else:
                    print(f"❌ FAILURE: Output type is {type(output)}, expected torch.Tensor")
            else:
                print("❌ FAILURE: test_output not found in results!")
                
        except Exception as e:
            print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            if os.path.exists(test_path):
                os.remove(test_path)
                print("\nCleaned up test file")
            
            # Clear memory data
            clear_all_memory_data()
            print("Cleared memory data")
    
    # Check server
    try:
        import server
        if not hasattr(server.PromptServer, 'instance') or server.PromptServer.instance is None:
            print("ERROR: ComfyUI server not running!")
            print("Please start ComfyUI first with: python main.py")
            sys.exit(1)
    except Exception as e:
        print(f"WARNING: Cannot verify server status: {e}")
    
    print("ComfyUI server appears to be running")
    print(f"Working directory: {os.getcwd()}\n")
    
    # Run test
    asyncio.run(test_workflow())

if __name__ == "__main__":
    main()