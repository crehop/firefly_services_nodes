"""
InDesign nodes for ComfyUI.

Provides Adobe InDesign API nodes for Data Merge operations.
"""

from .indesign_load_files_node import InDesignLoadFilesNode
from .indesign_data_merge_node import InDesignDataMergeNode

NODE_CLASS_MAPPINGS = {
    "InDesignLoadFilesNode": InDesignLoadFilesNode,
    "InDesignDataMergeNode": InDesignDataMergeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InDesignLoadFilesNode": "InDesign Load Files",
    "InDesignDataMergeNode": "InDesign Data Merge",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
