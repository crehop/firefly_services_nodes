"""
Utility nodes for ComfyUI Firefly Services.

Provides 6 mask utility nodes for image processing.
"""

from .combine_masks_node import CombineMasksNode
from .cutout_mask_node import CutoutMaskNode
from .cutout_alpha_fade_node import CutoutAlphaFadeNode
from .filter_masks_node import FilterMasksNode
from .mask_morphology_node import MaskMorphologyNode
from .outline_mask_node import OutlineMaskNode

NODE_CLASS_MAPPINGS = {
    "CombineMasksNode": CombineMasksNode,
    "CutoutMaskNode": CutoutMaskNode,
    "CutoutAlphaFadeNode": CutoutAlphaFadeNode,
    "FilterMasksNode": FilterMasksNode,
    "MaskMorphologyNode": MaskMorphologyNode,
    "OutlineMaskNode": OutlineMaskNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombineMasksNode": "Combine Masks",
    "CutoutMaskNode": "Cutout Mask",
    "CutoutAlphaFadeNode": "Cutout with Alpha Fade",
    "FilterMasksNode": "Filter Masks",
    "MaskMorphologyNode": "Mask Morphology (Dilate/Erode/Blur)",
    "OutlineMaskNode": "Outline Mask",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
