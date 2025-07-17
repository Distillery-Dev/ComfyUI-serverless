# nodes_internal.py - Internal nodes for Discomfort data passing

import torch
import json
import os
from typing import Any, Dict, Optional

# Global data store for in-memory data passing
_DISCOMFORT_DATA_STORE: Dict[str, Any] = {}

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

class DiscomfortDataLoader:
    """Internal node for loading data from memory or disk storage.
    This node is programmatically inserted into workflows to provide data injection."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storage_key": ("STRING", {"default": ""}),
                "storage_type": (["memory", "disk", "inline"], {"default": "memory"}),
                "expected_type": ("STRING", {"default": "ANY"}),
            }
        }
    
    RETURN_TYPES = (any_typ,)
    FUNCTION = "load_data"
    CATEGORY = "discomfort/internal"
    
    def load_data(self, storage_key: str, storage_type: str, expected_type: str):
        """Load data from storage based on key and type."""
        try:
            if storage_type == "memory":
                if storage_key not in _DISCOMFORT_DATA_STORE:
                    raise KeyError(f"Data key '{storage_key}' not found in memory store")
                data = _DISCOMFORT_DATA_STORE[storage_key]
                print(f"[DiscomfortDataLoader] Loaded data from memory: {storage_key}")
                return (data,)
            
            elif storage_type == "disk":
                if not os.path.exists(storage_key):
                    raise FileNotFoundError(f"Data file not found: {storage_key}")
                
                # Load serialized data from disk
                with open(storage_key, 'r') as f:
                    serialized = json.load(f)
                
                # Import here to avoid circular dependency
                from .workflow_tools import DiscomfortWorkflowTools
                tools = DiscomfortWorkflowTools()
                data = tools.deserialize(serialized, expected_type)
                
                print(f"[DiscomfortDataLoader] Loaded data from disk: {storage_key}")
                return (data,)
            
            elif storage_type == "inline":
                serialized = json.loads(storage_key)
                from .workflow_tools import DiscomfortWorkflowTools
                tools = DiscomfortWorkflowTools()
                data = tools.deserialize(serialized, expected_type)
                print(f"[DiscomfortDataLoader] Loaded inline data for expected_type: {expected_type}")
                return (data,)
            
            else:
                raise ValueError(f"Unknown storage type: {storage_type}")
                
        except Exception as e:
            print(f"[DiscomfortDataLoader] Error loading data: {str(e)}")
            # Return a safe default based on expected type (unchanged)
            if expected_type == "STRING":
                return ("",)
            elif expected_type == "INT":
                return (0,)
            elif expected_type == "FLOAT":
                return (0.0,)
            elif expected_type == "BOOLEAN":
                return (False,)
            elif expected_type == "IMAGE":
                return (torch.zeros(1, 64, 64, 3),)
            elif expected_type == "LATENT":
                return ({"samples": torch.zeros(1, 4, 8, 8)},)
            else:
                # Generic default
                return ({},)

# Function to store data (called by workflow tools)
def store_data(key: str, data: Any, storage_type: str = "memory") -> str:
    """Store data and return the storage key."""
    if storage_type == "memory":
        _DISCOMFORT_DATA_STORE[key] = data
        return key
    else:
        # For disk storage, key should be the file path
        return key

# Function to clear data (for cleanup)
def clear_data(key: str, storage_type: str = "memory"):
    """Clear data from storage."""
    if storage_type == "memory" and key in _DISCOMFORT_DATA_STORE:
        del _DISCOMFORT_DATA_STORE[key]

# Function to clear all memory data
def clear_all_memory_data():
    """Clear all data from memory store."""
    global _DISCOMFORT_DATA_STORE
    _DISCOMFORT_DATA_STORE.clear()