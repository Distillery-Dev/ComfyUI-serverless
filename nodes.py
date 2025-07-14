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
# import server  <- THIS LINE IS THE CULPRIT AND HAS BEEN REMOVED
import concurrent.futures
import copy
import time
import ast
import asyncio
from typing import Dict, Any

from .workflow_tools import DiscomfortWorkflowTools

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

class DiscomfortPort:
    """Port node for data flow in Discomfort workflows.
    Can act as INPUT (no incoming connections), OUTPUT (no outgoing connections), 
    or PASSTHRU (both incoming and outgoing connections)."""
    
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
    FUNCTION = "process_port"
    OUTPUT_NODE = True
    CATEGORY = "discomfort/utilities"

    def __init__(self):
        self.collected = None
        self.unique_id = None
        self.tags = []

    def process_port(self, unique_id, input_data=None, tags=""):
        """Process port data. Simply passes through input data and stores it for collection."""
        try:
            self.unique_id = unique_id
            self.tags = tags.split(',') if tags else []
            
            # If we have input data, store and return it
            if input_data is not None:
                self.collected = input_data
                print(f"[DiscomfortPort] Passing data through port '{unique_id}'")
                
                # Return both the data for downstream nodes and a UI output for collection
                return (input_data,)
            
            # No input data - this might be an INPUT port that will be replaced by DiscomfortDataLoader
            # Return a safe default for validation
            print(f"[DiscomfortPort] No input data for port '{unique_id}', returning default")
            
            # Return appropriate default based on tags
            if 'string' in self.tags or 'text' in self.tags:
                return ("",)
            elif 'int' in self.tags or 'integer' in self.tags:
                return (0,)
            elif 'float' in self.tags:
                return (0.0,)
            elif 'bool' in self.tags or 'boolean' in self.tags:
                return (False,)
            elif 'image' in self.tags:
                return (torch.zeros(1, 64, 64, 3),)
            elif 'latent' in self.tags:
                return ({"samples": torch.zeros(1, 4, 8, 8)},)
            else:
                # Default to small tensor for generic cases
                return (torch.zeros(1, 64, 64, 3),)
                
        except Exception as e:
            print(f"[DiscomfortPort] Error in process_port: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a safe default on error
            return (torch.zeros(1, 64, 64, 3),)
    
    @classmethod
    def IS_CHANGED(cls, unique_id, input_data=None, tags=""):
        """Always mark as changed to ensure proper data flow."""
        return float("NaN")  # Force re-execution


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
    """Simple test runner for Discomfort workflows."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "workflow_json": ("STRING", {"default": "test_workflow.json"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test_run"
    OUTPUT_NODE = True
    CATEGORY = "discomfort/testing"

    def test_run(self, input_image, workflow_json):
        """Run a simple test with a single workflow."""
        import asyncio
        
        tools = DiscomfortWorkflowTools()
        workflows = [workflow_json]
        inputs = {"test_input": input_image}
        
        # Run the async function synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                tools.run_sequential(workflows, inputs, iterations=1, use_ram=True)
            )
            
            # Find the first image output
            for key, value in result.items():
                if isinstance(value, torch.Tensor) and value.dim() == 4:
                    return (value,)
            
            # No image found, return input
            return (input_image,)
            
        except Exception as e:
            print(f"[DiscomfortTestRunner] Error: {e}")
            import traceback
            traceback.print_exc()
            return (input_image,)
            
        finally:
            loop.close()


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