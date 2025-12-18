"""
ComfyUI Firefly Services

Custom nodes for Adobe Firefly, Photoshop, and utility operations.
Provides 28 nodes total:
- 8 Firefly nodes (image/video generation)
- 6 Utils nodes (mask utilities)
- 14 Photoshop nodes (image editing and PSD creation)
"""

from .Firefly import NODE_CLASS_MAPPINGS as FIREFLY_NODES
from .Firefly import NODE_DISPLAY_NAME_MAPPINGS as FIREFLY_NAMES
from .Utils import NODE_CLASS_MAPPINGS as UTILS_NODES
from .Utils import NODE_DISPLAY_NAME_MAPPINGS as UTILS_NAMES
from .Photoshop import NODE_CLASS_MAPPINGS as PS_NODES
from .Photoshop import NODE_DISPLAY_NAME_MAPPINGS as PS_NAMES

NODE_CLASS_MAPPINGS = {**FIREFLY_NODES, **UTILS_NODES, **PS_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**FIREFLY_NAMES, **UTILS_NAMES, **PS_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
