import json
import requests
import time
import uuid
import os
import shutil
import sys
# This relative import will now work correctly because this file is imported as a module.
from .workflow_tools import DiscomfortWorkflowTools

# --- Test Configuration ---
SERVER_ADDRESS = "http://127.0.0.1:8188"
WORKFLOW_NAME = "discomfort_16-9_extender_with_flux.json"
IMAGE_NAME = "red_fox.png"
NODE_DIR = os.path.dirname(os.path.abspath(__file__))


def prepare_test_files():
    """Ensures the image is in the ComfyUI input directory."""
    source_image_path = os.path.join(NODE_DIR, IMAGE_NAME)
    input_dir = os.path.join(NODE_DIR, "..", "..", "input") # Go up to root and then to input
    destination_image_path = os.path.join(input_dir, IMAGE_NAME)

    if not os.path.exists(source_image_path):
        print(f"❌ ERROR: Source image not found at {source_image_path}")
        return False

    try:
        os.makedirs(input_dir, exist_ok=True)
        shutil.copy(source_image_path, destination_image_path)
        print(f"✅ Copied '{IMAGE_NAME}' to the ComfyUI 'input' directory.")
        return True
    except Exception as e:
        print(f"❌ ERROR: Could not prepare test files: {e}")
        return False


def cleanup_test_files():
    """Removes the image copied for the test."""
    input_dir = os.path.join(NODE_DIR, "..", "..", "input")
    destination_image_path = os.path.join(input_dir, IMAGE_NAME)
    if os.path.exists(destination_image_path):
        os.remove(destination_image_path)
        print(f"✅ Cleaned up test image from 'input' directory.")


async def run_real_workflow_test():
    """Loads, cleans, modifies, and runs a workflow via the API."""
    print("\n--- Starting Real-World Workflow Test ---")

    # 1. Check server status
    try:
        requests.get(f"{SERVER_ADDRESS}/system_stats").raise_for_status()
        print("✅ ComfyUI server is running.")
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Cannot connect to ComfyUI server at {SERVER_ADDRESS}.\n   Details: {e}")
        return

    # 2. Load the workflow
    workflow_path = os.path.join(NODE_DIR, WORKFLOW_NAME)
    try:
        with open(workflow_path, 'r') as f:
            workflow_data = json.load(f)
        print(f"✅ Successfully loaded workflow: {WORKFLOW_NAME}")
    except Exception as e:
        print(f"❌ ERROR: Could not load workflow file '{workflow_path}': {e}")
        return

    # 3. Remove Reroute nodes
    tools = DiscomfortWorkflowTools()
    clean_workflow = tools._get_workflow_with_reroutes_removed(workflow_data)
    print("✅ Removed Reroute nodes for backend execution.")

    # 4. Dynamically modify the workflow to inject the image loader
    try:
        # The input DiscomfortPort is node 264 in the updated workflow
        node_to_replace = next((n for n in clean_workflow["nodes"] if n["id"] == 264), None)
        if not node_to_replace or node_to_replace.get("type") != "DiscomfortPort":
            print("❌ ERROR: Could not find the input DiscomfortPort (id 264) to replace.")
            return

        node_to_replace["type"] = "LoadImage"
        # Ensure widgets_values exists before assigning
        if "widgets_values" not in node_to_replace:
            node_to_replace["widgets_values"] = []
        node_to_replace["widgets_values"] = [IMAGE_NAME]
        print(f"✅ Dynamically replaced input port with LoadImage node for '{IMAGE_NAME}'.")

    except Exception as e:
        print(f"❌ ERROR: Failed to modify the workflow for testing: {e}")
        return

    # 5. Convert the workflow to the API prompt format
    prompt_for_api = tools._build_prompt_from_workflow(clean_workflow)
    
    # 6. Queue the prompt
    data = {"prompt": prompt_for_api, "client_id": str(uuid.uuid4())}
    print("\nSubmitting modified workflow to the server...")
    try:
        response = requests.post(f"{SERVER_ADDRESS}/prompt", json=data)
        response.raise_for_status()
        result = response.json()
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            print(f"❌ ERROR: API did not return a prompt_id. Response: {result}")
            return
        print(f"✅ Workflow queued successfully! Prompt ID: {prompt_id}")
    except Exception as e:
        print(f"❌ ERROR: Failed to queue prompt: {e}")
        if 'response' in locals(): print(response.text)
        return

    # 7. Poll for completion
    print("\nWaiting for execution to complete...")
    while True:
        try:
            await asyncio.sleep(2) # Use asyncio.sleep for async functions
            history_response = requests.get(f"{SERVER_ADDRESS}/history/{prompt_id}")
            history = history_response.json()
            if prompt_id in history:
                execution_data = history[prompt_id]
                if 'status' in execution_data and execution_data['status'].get('status_str') == 'error':
                    print("❌ ERROR: Workflow execution failed on the server. Check server logs for details.")
                    break
                if 'outputs' in execution_data:
                    print("✅ Execution completed successfully! Final image has been generated.")
                    break
        except KeyboardInterrupt:
            print("\nTest interrupted by user.")
            break
        except Exception as e:
            print(f"Warning: Could not check history: {e}")
            await asyncio.sleep(5)


# This is the main function that the runner will call
async def run_test():
    if not prepare_test_files():
        sys.exit(1)
    
    try:
        await run_real_workflow_test()
    finally:
        cleanup_test_files()
        print("\n--- Test Finished ---")

# We no longer need the __main__ block here