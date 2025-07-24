# nodes_internal.py - Internal nodes for Discomfort data passing

import logging
from typing import Any, Dict, Optional

# Import the WorkflowContext to act as the backend for data loading.
from .workflow_context import WorkflowContext

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

class DiscomfortContextLoader:
    """
    Internal node for loading data using a WorkflowContext.
    This node is used in a DiscomfortPort running in INPUT mode when a pass-by-value `unique_id` is used.
    This node is programmatically injected into workflows by the `run_sequential`
    orchestrator to provide run-specific data via a `run_id` context for a `unique_id` variable.
    It is the counterpart to a DiscomfortPort running in OUTPUT mode.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the inputs, which are provided programmatically.
        - run_id: The unique identifier for the WorkflowContext managing this run's data.
        - unique_id: The specific key for the data to be loaded from the context.
        """
        return {
            "required": {
                # The run_id is essential for connecting to the correct, existing context.
                "run_id": ("STRING", {"default": "", "forceInput": True}),
                # The unique_id identifies the specific piece of data to load.
                "unique_id": ("STRING", {"default": "", "forceInput": True}),
            }
        }
    
    RETURN_TYPES = (any_typ,)
    FUNCTION = "load_data"
    CATEGORY = "discomfort/internal"

    def _get_logger(self, unique_id):
        """Initializes a logger for the node instance."""
        logger = logging.getLogger(f"DiscomfortContextLoader_{unique_id}")
        logger.propagate = False
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - [%(name)s] %(levelname)s - %(message)s')
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def load_data(self, run_id: str, unique_id: str):
        """
        An internal node that loads data from a specific run's WorkflowContext.
        It is programmatically injected into workflows to replace a DiscomfortPort
        in INPUT mode, providing a robust mechanism for data injection.
        """
        logger = self._get_logger(unique_id)
        logger.info(f"Attempting to load data from run '{run_id}'")
        
        if not run_id or not unique_id:
            logger.error("DiscomfortContextLoader requires a valid run_id and unique_id.")
            raise ValueError("DiscomfortContextLoader requires a valid run_id and unique_id.")
            
        try:
            # Connect to the existing WorkflowContext for this run. `create=False` ensures
            # we are loading, not creating a new context.
            context = WorkflowContext(run_id=run_id, create=False)
            
            # Load the data from the context. This single call replaces all previous
            # logic for handling different storage types and deserialization.
            data = context.load(unique_id)            
            logger.info(f"Successfully loaded data from context for unique_id: '{unique_id}'")
            return (data,)
            
        except Exception as e:
            logger.error(f"Error loading data from context: {str(e)}", exc_info=True)
            # In case of an error, return a generic empty dict to prevent workflow failure.
            # Downstream nodes may need to handle this gracefully.
            logger.warning("Returning empty dictionary to prevent workflow crash.")
            return ({},)

class DiscomfortContextSaver:
    """
    Internal node for saving data to a WorkflowContext.
    This node is used in a DiscomfortPort running in OUTPUT mode when a pass-by-value `unique_id` is used.
    This node is programmatically injected into workflows by the `run_sequential`
    orchestrator to provide run-specific data to a `run_id` context for a `unique_id` pass-by-value variable.
    It is the counterpart to a DiscomfortPort running in INPUT mode.
    It is designed as a sink node, terminating the graph by declaring RETURN_TYPES = ().
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unique_id": ("STRING", {"forceInput": True}),
                "input_data": (any_typ,),
            },
            "hidden": {
                "run_id": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = () # CRITICAL: This tells ComfyUI this is a terminal sink node.
    FUNCTION = "save_output"
    OUTPUT_NODE = True
    CATEGORY = "discomfort/internal" # Hide from regular users

    def _get_logger(self, unique_id):
        # (Logger setup remains the same as your previous DiscomfortPort)
        import logging
        logger = logging.getLogger(f"DiscomfortContextSaver_{unique_id}")
        logger.propagate = False
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - [%(name)s] %(levelname)s - %(message)s')
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def save_output(self, unique_id, input_data, run_id):
        logger = self._get_logger(unique_id)
        logger.info(f"OUTPUT mode engaged. Saving data for '{unique_id}' to context (run_id: '{run_id}').")
        
        if not run_id:
            logger.error("No run_id was provided. Cannot save data.")
            return {}

        try:
            context = WorkflowContext(run_id=run_id, create=False)
            context.save(unique_id, input_data)
            logger.info(f"Successfully saved data for '{unique_id}' to context.")
        except Exception as e:
            logger.critical(f"FATAL: Failed to save data to context: {e}", exc_info=True)

        return {} # Return an empty dict to signal completion.
