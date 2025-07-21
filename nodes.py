# nodes.py - Custom node definitions for Discomfort

import os
import torch
import requests
from PIL import Image
from openai import OpenAI
import json
import tempfile
import comfy
import numpy as np
import io
import base64
import concurrent.futures
import copy
import time
import ast
import asyncio
from typing import Dict, Any
from .workflow_tools import DiscomfortWorkflowTools
from .workflow_context import WorkflowContext
import uuid
import logging

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

class DiscomfortPort:
    """
    A multi-modal port for data flow in Discomfort workflows. Its behavior is
    determined by its connections and runtime flags.
    - INPUT: When replaced by a DataLoader.
    - OUTPUT: When `is_output` is True, it saves incoming data to the WorkflowContext.
    - PASSTHRU: Passes data through without I/O operations.
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
                # `run_id` is injected by the orchestrator to link this port to a specific run's context.
                "run_id": ("STRING", {"default": "", "forceInput": True}),
                # `is_output` flags this port to save data to the context instead of passing it through.
                "is_output": ("BOOLEAN", {"default": False}),
                 "tags": ("STRING", {"default": "any", "multiline": True}),
            }
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("output_data",)
    FUNCTION = "process_port"
    OUTPUT_NODE = True  # Marked as output to allow it to be an end-point.
    CATEGORY = "discomfort/utilities"

    def _get_logger(self):
        """Initializes a logger for the node instance."""
        logger = logging.getLogger(f"DiscomfortPort_{self.unique_id}")
        logger.propagate = False
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - [%(name)s] %(levelname)s - %(message)s')
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger
        
    def process_port(self, unique_id, input_data=None, tags="", run_id="", is_output=False):
        """
        Processes data based on the port's mode (OUTPUT or PASSTHRU).
        - If `is_output` is True, it saves the data to the WorkflowContext.
        - Otherwise, it acts as a simple passthrough.
        """
        self.unique_id = unique_id
        self.logger = self._get_logger()

        # If no input data is provided, return a sensible default. This can happen
        # in a standalone run or if an upstream node fails.
        if input_data is None:
            self.logger.warning("No input data provided, returning default (empty tensor).")
            return (torch.zeros(1, 64, 64, 3),)

        # --- OUTPUT Mode ---
        # If this port is designated as an output for a sequential run.
        if is_output:
            self.logger.info(f"OUTPUT mode engaged. Attempting to save data to context (run_id: '{run_id}').")
            if not run_id:
                self.logger.error("`is_output` is True, but no run_id was provided. Cannot save data. Passing data through.")
                return (input_data,)

            try:
                # Connect to the existing context for this run to save the data.
                context = WorkflowContext(run_id=run_id, create=False)
                context.save(self.unique_id, input_data)
                self.logger.info(f"Successfully saved data to context.")
            except Exception as e:
                self.logger.critical(f"FATAL: Failed to save data to context: {e}", exc_info=True)
                # This is a critical failure, but we still pass the data through to avoid breaking the graph.
                # The exception will be logged with a full traceback.
            
            # In OUTPUT mode, we still pass the data through. This allows an output port
            # to also be a passthrough port in a complex graph. Its primary function
            # in this mode is to save state to the context.
            return (input_data,)

        # --- PASSTHRU Mode ---
        # If not an output port, just pass the data through unchanged.
        self.logger.debug(f"PASSTHRU mode engaged.")
        return (input_data,)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        """Always re-execute to ensure data is processed correctly in iterative workflows."""
        return float("NaN")

class DiscomfortLoopExecutor:
    """Main control node for executing loops and conditional branches in Discomfort workflows."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "workflow_paths": ("STRING", {"multiline": True, "default": "path/to/A.json\npath/to/B.json"}),
                "max_iterations": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "initial_inputs": ("STRING", {"multiline": True, "default": "unique_id1: value1\nunique_id2: value2"}),
            },
            "optional": {
                "loop_condition_expression": ("STRING", {"multiline": True, "default": "discomfort_loop_counter <= max_iterations"}),
                "branch_condition_expression": ("STRING", {"multiline": True, "default": ""}),
                "condition_port": ("STRING", {"default": ""}),
                "then_workflows": ("STRING", {"multiline": True, "default": ""}),
                "else_workflows": ("STRING", {"multiline": True, "default": ""}),
                "use_ram": ("BOOLEAN", {"default": True}),
                "persist_prefix": ("STRING", {"default": ""}),
            },
            "hidden": {
                "break_called": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (any_typ,)
    FUNCTION = "execute_loop"
    OUTPUT_NODE = True
    CATEGORY = "discomfort/control"

    def parse_initial_inputs(self, initial_inputs_str: str) -> Dict[str, Any]:
        """Parse the initial inputs string into a dictionary."""
        inputs = {}
        lines = initial_inputs_str.strip().split('\n')
        
        for line in lines:
            if ':' not in line:
                continue
            
            key, value_str = line.split(':', 1)
            key = key.strip()
            value_str = value_str.strip()
            
            # Try to parse the value
            try:
                # Try literal eval first for Python literals
                value = ast.literal_eval(value_str)
            except:
                # Fallback to string
                value = value_str
            
            inputs[key] = value
        
        return inputs

    def evaluate_expression(self, expression: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate a condition expression."""
        if not expression:
            return True
        
        try:
            # Use ast.literal_eval for simple expressions
            # For more complex expressions, consider using simpleeval library
            # For now, support simple comparisons
            
            # Replace variable names with their values
            for key, value in context.items():
                if key in expression:
                    expression = expression.replace(key, str(value))
            
            # Evaluate the expression
            return eval(expression, {"__builtins__": {}}, {})
        except Exception as e:
            print(f"[DiscomfortLoopExecutor] Error evaluating expression '{expression}': {e}")
            return False

    def execute_loop(self, workflow_paths, max_iterations, initial_inputs, 
                    loop_condition_expression="discomfort_loop_counter <= max_iterations", 
                    branch_condition_expression="", condition_port="", 
                    then_workflows="", else_workflows="", 
                    use_ram=True, persist_prefix="", break_called=False):
        """Execute the loop with the specified workflows and conditions."""
        
        # Parse inputs
        workflow_list = [p.strip() for p in workflow_paths.split('\n') if p.strip()]
        loop_inputs = self.parse_initial_inputs(initial_inputs)
        loop_inputs['discomfort_loop_counter'] = 1
        loop_inputs['max_iterations'] = max_iterations
        
        tools = DiscomfortWorkflowTools()
        
        # Run the loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            final_outputs = {}
            
            for i in range(1, max_iterations + 1):
                if break_called:
                    break
                
                loop_inputs['discomfort_loop_counter'] = i
                
                # Run main workflows
                result = loop.run_until_complete(
                    tools.run_sequential(
                        workflow_list, 
                        loop_inputs, 
                        iterations=1, 
                        use_ram=use_ram,
                        persist_prefix=persist_prefix
                    )
                )
                
                # Update loop inputs with results
                loop_inputs.update(result)
                final_outputs.update(result)
                
                # Check branch condition
                if branch_condition_expression:
                    if self.evaluate_expression(branch_condition_expression, loop_inputs):
                        # Execute then branch
                        if then_workflows:
                            if then_workflows.strip() == "LOOP:BREAK":
                                break
                            elif then_workflows.strip() not in ["LOOP:PASS", "LOOP:CONTINUE"]:
                                then_list = [p.strip() for p in then_workflows.split('\n') if p.strip()]
                                branch_result = loop.run_until_complete(
                                    tools.run_sequential(
                                        then_list, 
                                        loop_inputs, 
                                        iterations=1,
                                        use_ram=use_ram,
                                        persist_prefix=persist_prefix
                                    )
                                )
                                loop_inputs.update(branch_result)
                                final_outputs.update(branch_result)
                    else:
                        # Execute else branch
                        if else_workflows:
                            if else_workflows.strip() == "LOOP:BREAK":
                                break
                            elif else_workflows.strip() not in ["LOOP:PASS", "LOOP:CONTINUE"]:
                                else_list = [p.strip() for p in else_workflows.split('\n') if p.strip()]
                                branch_result = loop.run_until_complete(
                                    tools.run_sequential(
                                        else_list, 
                                        loop_inputs, 
                                        iterations=1,
                                        use_ram=use_ram,
                                        persist_prefix=persist_prefix
                                    )
                                )
                                loop_inputs.update(branch_result)
                                final_outputs.update(branch_result)
                
                # Check loop condition
                if not self.evaluate_expression(loop_condition_expression, loop_inputs):
                    break
            
            # Return the primary output based on condition_port or first output
            if condition_port and condition_port in final_outputs:
                return (final_outputs[condition_port],)
            elif final_outputs:
                # Return first output
                first_key = list(final_outputs.keys())[0]
                return (final_outputs[first_key],)
            else:
                return ("No outputs",)
                
        finally:
            loop.close()


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
        
        tools = DiscomfortWorkflowTools()  # Instantiate the workflow tools
        
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
        
        # Step 2: Execute the workflow using run_sequential (async call)
        try:
            results = await tools.run_sequential(
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


class DiscomfortExtenderWorkflowRunner:
    """
    A specific test node to run the 16:9 extender workflow via the UI.
    It takes an image, runs it through the specified workflow, and returns the result.
    """
    @classmethod
    def INPUT_TYPES(s):
        # We get the default path relative to the ComfyUI root
        default_workflow_path = os.path.join("custom_nodes", "discomfort", "discomfort_16-9_extender_with_flux.json")
        return {
            "required": {
                "input_image": ("IMAGE",),
                "workflow_path": ("STRING", {"default": default_workflow_path}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_extender_workflow"
    CATEGORY = "discomfort/testing"

    async def run_extender_workflow(self, input_image, workflow_path):
        """
        This function will be executed when the node runs in the UI.
        """
        print("--- [DiscomfortExtenderWorkflowRunner] Starting Test ---")
        tools = DiscomfortWorkflowTools()
        
        # The input DiscomfortPort in your workflow has the unique_id "port1"
        inputs_for_workflow = {
            "port1": input_image
        }
        
        print(f"Workflow Path: {workflow_path}")
        print(f"Input Image Shape: {input_image.shape}")
        
        # run_sequential is an async function, so we must 'await' it
        results = await tools.run_sequential(
            workflow_paths=[workflow_path],
            inputs=inputs_for_workflow,
            iterations=1,
            use_ram=True
        )
        
        # The output DiscomfortPort in your workflow is also named "port1"
        final_image = results.get("port1")
        
        if final_image is None:
            print("❌ ERROR: Workflow did not return an output for 'port1'. Returning original image.")
            return (input_image,) # Return original image on failure
            
        print(f"✅ SUCCESS: Workflow returned an output image with shape {final_image.shape}.")
        return (final_image,)
