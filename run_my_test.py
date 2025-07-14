import sys
import os
import asyncio

# This script is the single entry point for our test.
# It should be run from the ComfyUI root directory.

def main():
    """
    Sets up the full ComfyUI environment and runs the Discomfort test.
    """
    print("--- Discomfort Test Runner ---")
    
    # 1. Add the ComfyUI root to the path.
    comfyui_root = os.getcwd()
    if comfyui_root not in sys.path:
        sys.path.insert(0, comfyui_root)

    # 2. Pre-load the server instance.
    try:
        import server
        if not hasattr(server.PromptServer, 'instance') or server.PromptServer.instance is None:
             print("❌ ERROR: ComfyUI server instance not found!")
             print("   Please ensure ComfyUI is running in another terminal.")
             sys.exit(1)
        print("✅ ComfyUI server instance found.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to import and initialize server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # --- START OF THE DEFINITIVE FIX ---
    # 3. Initialize all custom nodes.
    # This is the crucial step that loads all other node packs, making them
    # available to the execution engine.
    try:
        import nodes
        print("\nInitializing all custom nodes...")
        # We set init_api_nodes=False because we don't need external API nodes for this test.
        nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False)
        print("✅ All custom nodes initialized.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to initialize extra nodes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    # --- END OF THE DEFINITIVE FIX ---
        
    # 4. Now that the environment is fully correct, import and run our test.
    from custom_nodes.discomfort.run_extender_test import run_test
    
    print("\nStarting the Discomfort workflow test...\n" + "="*40)
    asyncio.run(run_test())
    print("="*40 + "\nTest run finished.")

if __name__ == "__main__":
    main()