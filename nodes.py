# nodes.py - Custom node definitions for Discomfort

import os
import torch
import requests
from PIL import Image
from openai import OpenAI
import json
import tempfile
import comfy  # Assuming ComfyUI internals are accessible
import numpy as np
import io
import base64
import server  # Changed from 'from comfy.server import PromptServer'
import concurrent.futures
import copy
import time
import ast  # For safe literal_eval in conditions
# Optional: import simpleeval if added to requirements.txt for more flexible safe evals

from .workflow_tools import DiscomfortWorkflowTools, _temp_execution_data  # Assuming relative import; adjust if needed

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

class DiscomfortPort:
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
        self.injected_data = None
        self.unique_id = None

    def process_port(self, unique_id, input_data=None, tags=""):
        try:
            self.unique_id = unique_id
            self.tags = tags.split(',') if tags else []

            # NEW APPROACH: Check if there's injected data for this unique_id
            # Look through all execution contexts for data matching this port's unique_id
            injected_data_found = False
            real_data = None
            
            for execution_id, exec_context in _temp_execution_data.items():
                if isinstance(exec_context, dict) and "injected_data" in exec_context:
                    injected_data = exec_context["injected_data"]
                    if unique_id in injected_data:
                        real_data = injected_data[unique_id]
                        injected_data_found = True
                        self.collected = real_data
                        print(f"[DiscomfortPort] Using injected data for '{unique_id}' from execution {execution_id}")
                        return (real_data,)
            
            # If we have input data and no injection was found, use the input data
            if input_data is not None:
                self.collected = input_data
                if not injected_data_found:
                    print(f"[DiscomfortPort] Using passed data for '{unique_id}' (no injection found)")
                return (input_data,)

            # No data found anywhere, return a safe default
            # Return appropriate type based on tags or default to small tensor
            print(f"[DiscomfortPort] No data found for '{unique_id}', using default")
            if 'string' in self.tags or 'text' in self.tags:
                return ("",)
            elif 'int' in self.tags or 'integer' in self.tags:
                return (0,)
            elif 'float' in self.tags:
                return (0.0,)
            else:
                # Default to small tensor for IMAGE type
                return (torch.zeros(1, 64, 64, 3),)
                
        except Exception as e:
            # Log the error for debugging
            print(f"[DiscomfortPort] Error in process_port: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a safe default on error
            return (torch.zeros(1, 64, 64, 3),)

class DiscomfortLoopExecutor:
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

    def execute_loop(self, workflow_paths, max_iterations, initial_inputs, loop_condition_expression="discomfort_loop_counter <= max_iterations", branch_condition_expression="", condition_port="", then_workflows="", else_workflows="", use_ram=True, persist_prefix="", break_called=False):
        # Stub for now - full logic to be added later
        # Parse and initialize
        tools = DiscomfortWorkflowTools()
        # ... parsing logic ...
        loop_inputs = {}  # Placeholder
        # ... loop flow ...
        return ("Dummy output",)  # Temporary return

class DiscomfortTestRunner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test_run"
    OUTPUT_NODE = True
    CATEGORY = "discomfort/testing"

    def test_run(self, input_image):
        import asyncio
        tools = DiscomfortWorkflowTools()
        workflows = ["discomfort_16-9_extender_with_flux.json", "discomfort_supir_upscaler.json"]
        inputs = {"main_image": input_image}  # Assume batch=1
        
        # Run the async function synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(tools.run_sequential(workflows, inputs, iterations=1, use_ram=True))
            final_image = result.get("main_image", torch.zeros_like(input_image))  # Fallback if missing
            return (final_image,)
        finally:
            loop.close()