#!/usr/bin/env python3
"""
Test script for Discomfort run_sequential functionality.
Run from ComfyUI root directory with the server already running:
    python -m custom_nodes.discomfort.test_discomfort
"""

def main():
    # All imports inside main() to ensure proper path setup
    import sys
    import os
    
    # Ensure we're running from ComfyUI root
    if not os.path.exists("main.py") or not os.path.exists("server.py"):
        print("ERROR: This script must be run from the ComfyUI root directory")
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    # Now we can safely import everything
    import asyncio
    import torch
    import json
    import time
    import uuid
    import logging
    
    # Import our module
    from custom_nodes.discomfort.workflow_tools import DiscomfortWorkflowTools
    
    async def test_workflow():
        print("=== Testing Discomfort run_sequential ===")
        
        # Create test workflow
        test_workflow = {
            "last_node_id": 2,
            "last_link_id": 1,
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
                    "widgets_values": ["test_input", ""]
                },
                {
                    "id": 2,
                    "type": "DiscomfortPort",
                    "pos": [400, 100],
                    "size": [200, 100],
                    "flags": {},
                    "order": 1,
                    "mode": 0,
                    "inputs": [
                        {"name": "input_data", "type": "*", "link": 1}
                    ],
                    "outputs": [],
                    "properties": {},
                    "widgets_values": ["test_output", ""]
                }
            ],
            "links": [[1, 1, 0, 2, 0, "*"]],
            "groups": [],
            "config": {},
            "extra": {},
            "version": 0.4
        }
        
        # Save test workflow
        test_path = "test_workflow_temp.json"
        with open(test_path, 'w') as f:
            json.dump(test_workflow, f, indent=2)
        
        try:
            # Create test data
            test_tensor = torch.rand((1, 64, 64, 3), dtype=torch.float32)
            print(f"Input tensor shape: {test_tensor.shape}")
            
            # Set up logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Run workflow
            tools = DiscomfortWorkflowTools()
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
                    print(f"Output matches input: {torch.allclose(test_tensor, output)}")
                else:
                    print(f"Output type: {type(output)}")
            else:
                print("ERROR: test_output not found!")
                
        except Exception as e:
            print(f"\nERROR: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            if os.path.exists(test_path):
                os.remove(test_path)
                print("\nCleaned up test file")
    
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