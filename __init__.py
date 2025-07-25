# __init__.py for Discomfort custom nodes

from .discomfort import Discomfort

from .nodes import (
    DiscomfortPort,
    DiscomfortTestRunner
)


from .nodes_internal import (
    DiscomfortContextLoader,
    DiscomfortContextSaver,
)

NODE_CLASS_MAPPINGS = {

    "DiscomfortPort": DiscomfortPort,
    "DiscomfortTestRunner": DiscomfortTestRunner,
    "DiscomfortContextLoader": DiscomfortContextLoader,
    "DiscomfortContextSaver": DiscomfortContextSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {

    "DiscomfortPort": "Discomfort Port (Input/Output)",
    "DiscomfortTestRunner": "Discomfort Test Runner",
    "DiscomfortContextLoader": "Discomfort Context Loader (Internal)",
    "DiscomfortContextSaver": "Discomfort Context Saver (Internal)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "Discomfort"]