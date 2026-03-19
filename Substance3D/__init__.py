"""
Substance 3D nodes for ComfyUI.

Provides 7 Adobe Substance 3D API nodes for 3D rendering, compositing, and file operations.
"""

from .substance3d_load_files_node import Substance3DLoadFilesNode
from .substance3d_render_basic_node import Substance3DRenderBasicNode
from .substance3d_render_node import Substance3DRenderNode
from .substance3d_composite_node import Substance3DCompositeNode
from .substance3d_convert_node import Substance3DConvertNode
from .substance3d_assemble_node import Substance3DAssembleNode
from .substance3d_describe_node import Substance3DDescribeNode

NODE_CLASS_MAPPINGS = {
    "Substance3DLoadFilesNode": Substance3DLoadFilesNode,
    "Substance3DRenderBasicNode": Substance3DRenderBasicNode,
    "Substance3DRenderNode": Substance3DRenderNode,
    "Substance3DCompositeNode": Substance3DCompositeNode,
    "Substance3DConvertNode": Substance3DConvertNode,
    "Substance3DAssembleNode": Substance3DAssembleNode,
    "Substance3DDescribeNode": Substance3DDescribeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Substance3DLoadFilesNode": "S3D Load Files",
    "Substance3DRenderBasicNode": "S3D Render Basic",
    "Substance3DRenderNode": "S3D Render",
    "Substance3DCompositeNode": "S3D Composite",
    "Substance3DConvertNode": "S3D Convert",
    "Substance3DAssembleNode": "S3D Assemble Scene",
    "Substance3DDescribeNode": "S3D Describe",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
