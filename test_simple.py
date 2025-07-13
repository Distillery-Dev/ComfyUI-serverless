#!/usr/bin/env python3
"""
Simple API-based test for Discomfort run_sequential functionality.
This version uses the ComfyUI HTTP API to avoid import issues.
Run from ComfyUI root: python custom_nodes/discomfort/test_simple.py
"""

import json
import requests
import time
import uuid
import sys

def test_discomfort_api():
    """Test Discomfort via ComfyUI API"""
    
    server_address = "http://127.0.0.1:8188"
    
    # Check if server is running
    try:
        response = requests.get(f"{server_address}/system_stats")
        if response.status_code != 200:
            print("ERROR: ComfyUI server not responding properly")
            return
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to ComfyUI server at", server_address)
        print("Please start ComfyUI with: python main.py")
        return
    
    print("=== Testing Discomfort via API ===")
    
    # Create a simple workflow that uses DiscomfortPort nodes
    workflow = {
        "1": {
            "class_type": "DiscomfortPort",
            "inputs": {
                "unique_id": "test_input",
                "tags": ""
            }
        },
        "2": {
            "class_type": "DiscomfortPort", 
            "inputs": {
                "unique_id": "test_output",
                "tags": "",
                "input_data": [1, 0]  # Link from node 1, output 0
            }
        }
    }
    
    # Create the prompt
    prompt_data = {
        "prompt": workflow,
        "client_id": str(uuid.uuid4())
    }
    
    print("Sending workflow to ComfyUI...")
    
    # Send the prompt
    response = requests.post(f"{server_address}/prompt", json=prompt_data)
    if response.status_code != 200:
        print(f"ERROR: Failed to queue prompt: {response.text}")
        return
        
    result = response.json()
    prompt_id = result.get("prompt_id")
    print(f"Queued with prompt_id: {prompt_id}")
    
    # Wait for completion
    print("Waiting for execution...")
    start_time = time.time()
    while True:
        if time.time() - start_time > 30:
            print("ERROR: Timeout waiting for execution")
            break
            
        # Check history
        history_response = requests.get(f"{server_address}/history/{prompt_id}")
        if history_response.status_code == 200:
            history = history_response.json()
            if prompt_id in history:
                execution_result = history[prompt_id]
                if "outputs" in execution_result:
                    print("Execution completed!")
                    print(f"Outputs: {json.dumps(execution_result['outputs'], indent=2)}")
                    break
                elif "status" in execution_result:
                    status = execution_result["status"]
                    if status.get("status_str") == "error":
                        print(f"ERROR: Execution failed: {status}")
                        break
        
        time.sleep(0.5)
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_discomfort_api() 