import json
import requests
import time
import uuid
import os
from PIL import Image
import numpy as np
import sys # Added for sys.exit

# --- Test Configuration ---
SERVER_ADDRESS = "http://127.0.0.1:8188"
DUMMY_IMAGE_NAME = "discomfort_test_image.png"
SUB_WORKFLOW_NAME = "discomfort_sub_workflow.json"

def create_dummy_files():
    """Creates the dummy image and sub-workflow JSON needed for the test."""
    
    # 1. Create a dummy image file
    try:
        img = Image.new('RGB', (64, 64), 'black')
        input_dir = "input" 
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        img.save(os.path.join(input_dir, DUMMY_IMAGE_NAME))
        print(f"✅ Created dummy image: {DUMMY_IMAGE_NAME}")
    except Exception as e:
        print(f"❌ ERROR: Could not create dummy image: {e}")
        return False

    # 2. Create the sub-workflow file in the correct format
    # --- START OF FIX ---
    sub_workflow = {
        "last_node_id": 3,
        "last_link_id": 2,
        "nodes": [
            {
                "id": 1,
                "type": "DiscomfortPort",
                "properties": {},
                "widgets_values": ["test_input"]
            },
            {
                "id": 2,
                "type": "ImageInvert",
                "properties": {},
                "inputs": [{"name": "image", "type": "IMAGE", "link": 1}]
            },
            {
                "id": 3,
                "type": "DiscomfortPort",
                "properties": {},
                "widgets_values": ["test_output"],
                "inputs": [{"name": "input_data", "type": "*", "link": 2}]
            }
        ],
        "links": [
            # [link_id, from_node_id, from_node_slot, to_node_id, to_node_slot, type]
            [1, 1, 0, 2, 0, "IMAGE"],
            [2, 2, 0, 3, 0, "*"]
        ],
        "version": 0.4
    }
    # --- END OF FIX ---
    try:
        with open(SUB_WORKFLOW_NAME, 'w') as f:
            json.dump(sub_workflow, f, indent=4) # Use indent for readability
        print(f"✅ Created sub-workflow file in the correct format: {SUB_WORKFLOW_NAME}")
    except Exception as e:
        print(f"❌ ERROR: Could not create sub-workflow file: {e}")
        return False
        
    return True

def cleanup_dummy_files():
    """Removes the files created for the test."""
    input_image_path = os.path.join("input", DUMMY_IMAGE_NAME)
    if os.path.exists(input_image_path):
        os.remove(input_image_path)
        print(f"✅ Cleaned up dummy image.")
        
    if os.path.exists(SUB_WORKFLOW_NAME):
        os.remove(SUB_WORKFLOW_NAME)
        print(f"✅ Cleaned up sub-workflow file.")

def run_api_test():
    """
    Submits a workflow using DiscomfortTestRunner to the ComfyUI API 
    and waits for the result.
    """
    print("\n--- Starting Discomfort API Test ---")
    
    # Check if the server is running
    try:
        response = requests.get(f"{SERVER_ADDRESS}/system_stats")
        if response.status_code != 200:
            print(f"❌ ERROR: ComfyUI server not responding at {SERVER_ADDRESS}. Is it running?")
            return
        print("✅ ComfyUI server is running.")
    except requests.ConnectionError:
        print(f"❌ ERROR: Cannot connect to ComfyUI server at {SERVER_ADDRESS}.")
        print("Please start ComfyUI first with: python main.py")
        return

    # Define the main workflow that uses our test node
    main_workflow = {
        "1": {
            "class_type": "LoadImage",
            "inputs": { "image": DUMMY_IMAGE_NAME }
        },
        "2": {
            "class_type": "DiscomfortTestRunner",
            "inputs": {
                "input_image": ["1", 0],
                "workflow_json": SUB_WORKFLOW_NAME 
            }
        },
        "3": {
            "class_type": "PreviewImage",
            "inputs": { "images": ["2", 0] }
        }
    }

    # Prepare and queue the prompt
    prompt_data = {
        "prompt": main_workflow,
        "client_id": str(uuid.uuid4())
    }
    
    print("\nSubmitting workflow to the server...")
    try:
        response = requests.post(f"{SERVER_ADDRESS}/prompt", json=prompt_data)
        response.raise_for_status()
        result = response.json()
        prompt_id = result.get("prompt_id")
        print(f"✅ Workflow queued successfully! Prompt ID: {prompt_id}")
    except Exception as e:
        print(f"❌ ERROR: Failed to queue prompt: {e}")
        return

    # Poll for completion
    print("\nWaiting for execution to complete...")
    start_time = time.time()
    while True:
        if time.time() - start_time > 60:
            print("❌ ERROR: Timeout waiting for execution.")
            break
            
        try:
            history_response = requests.get(f"{SERVER_ADDRESS}/history/{prompt_id}")
            history = history_response.json()
            if prompt_id in history:
                execution_data = history[prompt_id]
                # A successful run will have outputs, an error will not.
                if 'status' in execution_data and execution_data['status'].get('status_str') == 'error':
                    print("❌ ERROR: Workflow execution failed on the server. Check server logs.")
                    break
                if 'outputs' in execution_data:
                    print("✅ Execution completed successfully!")
                    print("✅ Outputs from DiscomfortTestRunner were generated.")
                    break
        except Exception as e:
            print(f"Warning: Could not check history: {e}")
        
        time.sleep(1)

if __name__ == "__main__":
    if not create_dummy_files():
        sys.exit(1)
    
    try:
        run_api_test()
    finally:
        cleanup_dummy_files()
        print("\n--- Test Finished ---")