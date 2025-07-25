# nodes.py - Custom node definitions for Discomfort

import os
import ast
import asyncio
from typing import Dict, Any
# WorkflowTools import removed - now using Discomfort class
from .workflow_context import WorkflowContext

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

class DiscomfortPort:
    """
    A user-facing port for passthrough data flow in Discomfort workflows.
    It also serves as a placeholder for INPUT and OUTPUT ports, which are
    dynamically replaced by the run_sequential engine before execution.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unique_id": ("STRING", {"default": "port1"}),
            },
            "optional": {
                "input_data": (any_typ,),
            },
            "hidden": {
                "tags": ("STRING", {"default": "any", "multiline": True}),
            }
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("output_data",)
    FUNCTION = "passthru"
    OUTPUT_NODE = True  # Allows it to be a terminal node in the UI
    CATEGORY = "discomfort/utilities"

    def passthru(self, unique_id, input_data=None, tags=""):
        """
        In its user-facing role, this node simply passes data through.
        The run_sequential orchestrator replaces it with a specialized node
        for INPUT or OUTPUT operations at runtime.
        """
        if input_data is None:
            # Return a default empty dict to avoid errors if unconnected
            return ({},)
        return (input_data,)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")


class DiscomfortTestRunner:
    """
    A test runner node for executing arbitrary workflows containing DiscomfortPorts.
    This node tests the detection and substitution of INPUT DiscomfortPorts with DiscomfortDataLoaders,
    and the extraction of results from OUTPUT DiscomfortPorts.
    
    - Accepts a workflow JSON path as input.
    - Supports up to 5 optional inputs, each mapped to a user-specified unique_id.
    - Supports up to 5 outputs, each extracted by a user-specified unique_id from the workflow results.
    - Allows specifying max_iterations for loop testing (default: 1).
    """
    
    @classmethod
    def INPUT_TYPES(s):
        # Default workflow path relative to the ComfyUI custom_nodes/discomfort directory
        default_workflow_path = os.path.join("custom_nodes", "discomfort", "test_workflow.json")
        return {
            "required": {
                "workflow_json": ("STRING", {"default": default_workflow_path}),  # Path to the workflow JSON file
                "max_iterations": ("INT", {"default": 1, "min": 1, "max": 1000}),  # Number of iterations for run_sequential
            },
            "optional": {
                # Up to 5 optional inputs (ANY type) with corresponding unique_id strings
                "input1": (any_typ,),
                "unique_id_input1": ("STRING", {"default": ""}),  # Unique ID for input1 (leave empty if unused)
                
                "input2": (any_typ,),
                "unique_id_input2": ("STRING", {"default": ""}),
                
                "input3": (any_typ,),
                "unique_id_input3": ("STRING", {"default": ""}),
                
                "input4": (any_typ,),
                "unique_id_input4": ("STRING", {"default": ""}),
                
                "input5": (any_typ,),
                "unique_id_input5": ("STRING", {"default": ""}),
                
                # Unique IDs for up to 5 outputs to extract from results
                "unique_id_output1": ("STRING", {"default": ""}),  # Unique ID for output1 (leave empty if unused)
                "unique_id_output2": ("STRING", {"default": ""}),
                "unique_id_output3": ("STRING", {"default": ""}),
                "unique_id_output4": ("STRING", {"default": ""}),
                "unique_id_output5": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = (any_typ, any_typ, any_typ, any_typ, any_typ,)  # Fixed 5 outputs (ANY type)
    RETURN_NAMES = ("output1", "output2", "output3", "output4", "output5",)  # Named outputs for clarity
    FUNCTION = "run_test"  # The method to call for execution
    OUTPUT_NODE = True  # Marks this as an output node in ComfyUI
    CATEGORY = "discomfort/testing"  # Category in the ComfyUI node menu

    async def run_test(self, workflow_json, max_iterations,
                       input1=None, unique_id_input1="",
                       input2=None, unique_id_input2="",
                       input3=None, unique_id_input3="",
                       input4=None, unique_id_input4="",
                       input5=None, unique_id_input5="",
                       unique_id_output1="",
                       unique_id_output2="",
                       unique_id_output3="",
                       unique_id_output4="",
                       unique_id_output5=""):
        """
        Asynchronous execution method for the test runner.
        - Collects inputs mapped to their unique_ids (only if unique_id is provided).
        - Runs the workflow using run_sequential with the collected inputs and specified iterations.
        - Extracts outputs based on provided output unique_ids.
        - Returns a tuple of up to 5 outputs; unused outputs are set to None.
        """
        print("--- [DiscomfortTestRunner] Starting Test ---")
        
        from .discomfort import Discomfort
        discomfort = await Discomfort.create()  # Create Discomfort instance
        
        # Step 1: Collect inputs into a dictionary only if unique_id is provided and non-empty
        inputs_dict = {}
        if unique_id_input1 and input1 is not None:
            inputs_dict[unique_id_input1] = input1
        if unique_id_input2 and input2 is not None:
            inputs_dict[unique_id_input2] = input2
        if unique_id_input3 and input3 is not None:
            inputs_dict[unique_id_input3] = input3
        if unique_id_input4 and input4 is not None:
            inputs_dict[unique_id_input4] = input4
        if unique_id_input5 and input5 is not None:
            inputs_dict[unique_id_input5] = input5
        
        print(f"[DiscomfortTestRunner] Collected inputs: {list(inputs_dict.keys())}")
        print(f"[DiscomfortTestRunner] Workflow Path: {workflow_json}")
        print(f"[DiscomfortTestRunner] Max Iterations: {max_iterations}")
        
        # Step 2: Execute the workflow using Discomfort.run (async call)
        try:
            results = await discomfort.run(
                workflow_paths=[workflow_json],  # Single workflow as a list
                inputs=inputs_dict,  # Pass the collected inputs
                iterations=max_iterations,  # Use the specified iterations
                use_ram=True  # Default to RAM for testing; can be made configurable if needed
            )
            print(f"[DiscomfortTestRunner] Workflow execution completed. Results keys: {list(results.keys())}")
        except Exception as e:
            print(f"[DiscomfortTestRunner] Error during workflow execution: {str(e)}")
            import traceback
            traceback.print_exc()
            # On error, return None for all outputs
            return (None, None, None, None, None)
        
        # Step 3: Extract outputs based on provided unique_ids
        output_tuple = []
        output_uids = [
            unique_id_output1 if unique_id_output1 else None,
            unique_id_output2 if unique_id_output2 else None,
            unique_id_output3 if unique_id_output3 else None,
            unique_id_output4 if unique_id_output4 else None,
            unique_id_output5 if unique_id_output5 else None
        ]
        
        for uid in output_uids:
            if uid:
                output_value = results.get(uid, None)  # Extract if present, else None
                output_tuple.append(output_value)
                if output_value is not None:
                    print(f"[DiscomfortTestRunner] Extracted output for '{uid}'")
                else:
                    print(f"[DiscomfortTestRunner] No output found for '{uid}' - returning None")
            else:
                output_tuple.append(None)  # Unused output slot
        
        # Ensure exactly 5 outputs (pad with None if fewer, though loop already handles up to 5)
        while len(output_tuple) < 5:
            output_tuple.append(None)
        
        print("--- [DiscomfortTestRunner] Test Completed ---")
        return tuple(output_tuple)  # Return as tuple for ComfyUI output