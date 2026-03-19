"""
Photoshop nodes for ComfyUI.

Provides 19 Adobe Photoshop API nodes for image editing and PSD creation.
"""

from .photoshop_easy_nodes import (
    PhotoshopRemoveBackgroundNode,
    PhotoshopRefineMaskNode,
    PhotoshopFillMaskedAreasNode,
)
from .photoshop_actions_node import PhotoshopActionsNode
from .photoshop_psd_actionjson_node import PhotoshopPsdActionJsonNode, PhotoshopActionJsonNode
from .photoshop_psd_manifest_node import PhotoshopPsdManifestNode
from .photoshop_mask_objects_node import PhotoshopMaskObjectsNode
from .photoshop_mask_body_parts_node import PhotoshopMaskBodyPartsNode
from .photoshop_product_crop_node import PhotoshopProductCropNode
from .photoshop_depth_blur_node import PhotoshopDepthBlurNode
from .photoshop_create_psd_node import PhotoshopCreatePSDNode
from .photoshop_create_psd_ez_node import PhotoshopCreatePSDEZNode
from .photoshop_preview_psd_node import PhotoshopPreviewPSDNode
from .photoshop_load_psd_node import PhotoshopLoadPSDNode
from .photoshop_document_operations_node import PhotoshopDocumentOperationsNode
from .photoshop_text_node import PhotoshopTextEditNode
from .photoshop_rendition_node import PhotoshopRenditionCreateNode
from .photoshop_smart_object_node import PhotoshopSmartObjectNode
from .photoshop_photo_restoration_node import PhotoshopPhotoRestorationNode

NODE_CLASS_MAPPINGS = {
    "PhotoshopRemoveBackgroundNode": PhotoshopRemoveBackgroundNode,
    "PhotoshopRefineMaskNode": PhotoshopRefineMaskNode,
    "PhotoshopMaskObjectsNode": PhotoshopMaskObjectsNode,
    "PhotoshopMaskBodyPartsNode": PhotoshopMaskBodyPartsNode,
    "PhotoshopFillMaskedAreasNode": PhotoshopFillMaskedAreasNode,
    "PhotoshopActionsNode": PhotoshopActionsNode,
    "PhotoshopActionJsonNode": PhotoshopActionJsonNode,
    "PhotoshopPsdActionJsonNode": PhotoshopPsdActionJsonNode,
    "PhotoshopPsdManifestNode": PhotoshopPsdManifestNode,
    "PhotoshopProductCropNode": PhotoshopProductCropNode,
    "PhotoshopDepthBlurNode": PhotoshopDepthBlurNode,
    "PhotoshopCreatePSDNode": PhotoshopCreatePSDNode,
    "PhotoshopCreatePSDEZNode": PhotoshopCreatePSDEZNode,
    "PhotoshopPreviewPSDNode": PhotoshopPreviewPSDNode,
    "PhotoshopLoadPSDNode": PhotoshopLoadPSDNode,
    "PhotoshopDocumentOperationsNode": PhotoshopDocumentOperationsNode,
    "PhotoshopTextEditNode": PhotoshopTextEditNode,
    "PhotoshopRenditionCreateNode": PhotoshopRenditionCreateNode,
    "PhotoshopSmartObjectNode": PhotoshopSmartObjectNode,
    "PhotoshopPhotoRestorationNode": PhotoshopPhotoRestorationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoshopRemoveBackgroundNode": "Remove Background",
    "PhotoshopRefineMaskNode": "Refine Mask",
    "PhotoshopMaskObjectsNode": "Mask Objects",
    "PhotoshopMaskBodyPartsNode": "Mask Body Parts",
    "PhotoshopFillMaskedAreasNode": "Fill Masked Areas",
    "PhotoshopActionsNode": "Photoshop Actions",
    "PhotoshopActionJsonNode": "Photoshop ActionJSON",
    "PhotoshopPsdActionJsonNode": "Photoshop ActionJSON",
    "PhotoshopPsdManifestNode": "Photoshop PSD Manifest",
    "PhotoshopProductCropNode": "Photoshop Product Crop",
    "PhotoshopDepthBlurNode": "Photoshop Depth Blur",
    "PhotoshopCreatePSDNode": "Create PSD",
    "PhotoshopCreatePSDEZNode": "Create PSD EZ Layers",
    "PhotoshopPreviewPSDNode": "Preview PSD",
    "PhotoshopLoadPSDNode": "Load PSD",
    "PhotoshopDocumentOperationsNode": "Document Operations",
    "PhotoshopTextEditNode": "Text Edit",
    "PhotoshopRenditionCreateNode": "Rendition Create",
    "PhotoshopSmartObjectNode": "Smart Object",
    "PhotoshopPhotoRestorationNode": "Photo Restoration",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
