# __init__.py for Discomfort custom nodes

from .nodes import (
    DiscomfortPort,
    DiscomfortLoopExecutor,
    DiscomfortTestRunner,
    DiscomfortExtenderWorkflowRunner
)

from .nodes_auxiliary import (
    DiscomfortFolderImageLoader,
    DiscomfortImageDescriber,
)

from .nodes_internal import (
    DiscomfortContextLoader,
    DiscomfortContextSaver,
)

NODE_CLASS_MAPPINGS = {
    "DiscomfortFolderImageLoader": DiscomfortFolderImageLoader,
    "DiscomfortImageDescriber": DiscomfortImageDescriber,
    "DiscomfortPort": DiscomfortPort,
    "DiscomfortLoopExecutor": DiscomfortLoopExecutor,
    "DiscomfortTestRunner": DiscomfortTestRunner,
    "DiscomfortContextLoader": DiscomfortContextLoader,
    "DiscomfortContextSaver": DiscomfortContextSaver,
    "DiscomfortExtenderWorkflowRunner": DiscomfortExtenderWorkflowRunner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiscomfortFolderImageLoader": "Discomfort Folder Image Loader",
    "DiscomfortImageDescriber": "Discomfort Image Describer",
    "DiscomfortPort": "Discomfort Port (Input/Output)",
    "DiscomfortLoopExecutor": "Discomfort Loop Executor",
    "DiscomfortTestRunner": "Discomfort Test Runner",
    "DiscomfortContextLoader": "Discomfort Context Loader (Internal)",
    "DiscomfortContextSaver": "Discomfort Context Saver (Internal)",
    "DiscomfortExtenderWorkflowRunner": "Discomfort Extender Runner (Test)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]