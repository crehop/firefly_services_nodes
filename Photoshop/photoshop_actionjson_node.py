"""
Adobe Photoshop ActionJSON Node - Re-export

This module re-exports from photoshop_psd_actionjson_node for backward compatibility.
The consolidated node accepts IMAGE tensors, file paths, and URLs.
"""

from .photoshop_psd_actionjson_node import (
    PhotoshopPsdActionJsonNode as PhotoshopActionJsonNode,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["PhotoshopActionJsonNode", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
