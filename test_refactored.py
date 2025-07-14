#!/usr/bin/env python3
"""
Test script for the refactored Discomfort run_sequential functionality.
This script is intended to be called by a runner script from the ComfyUI root.
"""
import sys
import os
import asyncio
import torch
import json
import time
import uuid
import logging

# Since this is now imported, we can safely get our modules
from custom_nodes.discomfort.workflow_tools import DiscomfortWorkflowTools
from custom_nodes.discomfort.nodes_internal import DiscomfortDataLoader, clear_all_memory_data

# The main test logic is now encapsulated in this async function
async def run_test():
    """The main test function to be called by the runner."""
    
    # Register the DiscomfortDataLoader node
    import nodes
    if 'DiscomfortDataLoader' not in nodes.NODE_CLASS_MAPPINGS:
        nodes.NODE_CLASS_MAPPINGS['DiscomfortDataLoader'] = DiscomfortDataLoader
        print("Registered DiscomfortDataLoader node")

    print("=== Testing Refactored Discomfort run_sequential ===\n")
    
    # Create a simple test workflow
    test_workflow = {
        "last_node_id": 3,
        "last_link_id": 2,
        "nodes": [
            {
                "id": 1,
                "type": "DiscomfortPort",
                "pos": [100, 100], "size": [200, 100], "flags": {}, "order": 0, "mode": 0, "inputs": [],
                "outputs": [{"name": "output", "type": "*", "links": [1], "slot_index": 0}],
                "properties": {}, "widgets_values": ["test_input", "image"]
            },
            {
                "id": 2,
                "type": "ImageInvert",
                "pos": [400, 100], "size": [200, 100], "flags": {}, "order": 1, "mode": 0,
                "inputs": [{"name": "image", "type": "IMAGE", "link": 1}],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [2], "slot_index": 0}],
                "properties": {}
            },
            {
                "id": 3,
                "type": "DiscomfortPort",
                "pos": [700, 100], "size": [200, 100], "flags": {}, "order": 2, "mode": 0,
                "inputs": [{"name": "input_data", "type": "*", "link": 2}],
                "outputs": [], "properties": {}, "widgets_values": ["test_output", "image"]
            }
        ],
        "links": [[1, 1, 0, 2, 0, "*"], [2, 2, 0, 3, 0, "IMAGE"]],
        "groups": [], "config": {}, "extra": {}, "version": 0.4
    }
    
    # Save test workflow
    test_path = "test_workflow_refactored_temp.json"
    with open(test_path, 'w') as f:
        json.dump(test_workflow, f, indent=2)
    
    try:
        # Create test data
        test_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        for i in range(64):
            for j in range(64):
                test_tensor[0, i, j, 0] = i / 63.0
                test_tensor[0, i, j, 1] = j / 63.0
                test_tensor[0, i, j, 2] = (i + j) / 126.0
        
        print(f"Input tensor shape: {test_tensor.shape}")
        
        # Run workflow
        tools = DiscomfortWorkflowTools()
        print("\nRunning workflow...")
        result = await tools.run_sequential(
            workflow_paths=[test_path],
            inputs={"test_input": test_tensor},
            iterations=1,
            use_ram=True
        )
        
        print("\nWorkflow completed!")
        
        if "test_output" in result:
            output = result["test_output"]
            if isinstance(output, torch.Tensor):
                expected = 1.0 - test_tensor
                if torch.allclose(expected, output, atol=1e-5):
                    print("\n✅ SUCCESS: Output matches expected inverted image!")
                else:
                    print("\n❌ FAILURE: Output does not match expected inverted image")
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
        clear_all_memory_data()
        print("Cleared memory data")

# This file is now a module, so the __main__ block is removed.