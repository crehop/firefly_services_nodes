"""
Combine Masks Node

Combines multiple mask image lists into a single unified mask.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F


class CombineMasksNode:
    """
    Combine multiple mask image lists into a SINGLE unified mask.

    Features:
    - Accepts up to 11 mask inputs (1 primary + 10 additional)
    - Automatically handles different mask sizes (resizes to largest dimensions)
    - Merges all white pixels from all input masks into one combined mask
    - Outputs a SINGLE IMAGE (not a list!) with all white areas merged

    Use Cases:
    - Combine multiple body part masks (Face, Hair, Hands) into complete head mask
    - Merge filtered outputs from mask detection (e.g., all "head" category masks)
    - Create composite masks from different sources

    Example:
    - Input masks: [Face (white on face), Hair (white on hair), Hands (white on hands)]
    - Output: SINGLE mask with white on face + hair + hands combined

    Note: If you input 10 masks, you get 1 output (not 10)!
    """

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("combined_mask",)
    FUNCTION = "combine_masks"
    CATEGORY = "api node/Firefly Utils"
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "masks": ("IMAGE", {
                    "tooltip": "Primary mask image list to combine",
                }),
            },
            "optional": {},
        }

        # Add 10 optional mask inputs
        for i in range(1, 11):
            inputs["optional"][f"masks_{i}"] = ("IMAGE", {
                "tooltip": f"Additional mask list {i} (optional)",
            })

        return inputs

    @staticmethod
    def _ensure_same_size(masks: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Resize all masks to the largest dimensions found.
        Handles different sized masks by padding/resizing.
        """
        if not masks:
            return []

        # Find max dimensions across all masks
        max_h = max(m.shape[1] for m in masks)
        max_w = max(m.shape[2] for m in masks)

        resized_masks = []
        for mask in masks:
            b, h, w, c = mask.shape

            if h == max_h and w == max_w:
                resized_masks.append(mask)
            else:
                # Resize using bilinear interpolation
                # Convert from [B, H, W, C] to [B, C, H, W] for F.interpolate
                mask_hwc = mask.permute(0, 3, 1, 2)
                mask_resized = F.interpolate(
                    mask_hwc,
                    size=(max_h, max_w),
                    mode='bilinear',
                    align_corners=False
                )
                # Convert back to [B, H, W, C]
                mask_resized = mask_resized.permute(0, 2, 3, 1)
                resized_masks.append(mask_resized)

        return resized_masks

    def combine_masks(
        self,
        masks: list,
        masks_1: Optional[list] = None,
        masks_2: Optional[list] = None,
        masks_3: Optional[list] = None,
        masks_4: Optional[list] = None,
        masks_5: Optional[list] = None,
        masks_6: Optional[list] = None,
        masks_7: Optional[list] = None,
        masks_8: Optional[list] = None,
        masks_9: Optional[list] = None,
        masks_10: Optional[list] = None,
    ):
        """Combine all provided masks into a single unified mask by merging white pixels."""

        # With INPUT_IS_LIST = True, all inputs come as lists
        # But each input can be:
        # - A list with multiple images: [img1, img2, img3] (from OUTPUT_IS_LIST=True node)
        # - A list with single image: [img1] (from regular IMAGE node)

        # Collect all mask inputs that were provided
        all_mask_inputs = []

        # Add primary masks (always provided)
        if masks is not None:
            all_mask_inputs.append(masks)

        # Add optional mask inputs
        optional_masks = [
            masks_1, masks_2, masks_3, masks_4, masks_5,
            masks_6, masks_7, masks_8, masks_9, masks_10,
        ]

        for mask_input in optional_masks:
            if mask_input is not None:
                all_mask_inputs.append(mask_input)

        # Flatten all inputs into a single list of individual mask tensors
        all_masks = []
        for mask_input in all_mask_inputs:
            # Handle list inputs (from INPUT_IS_LIST)
            if isinstance(mask_input, list):
                for item in mask_input:
                    if isinstance(item, torch.Tensor):
                        # Handle batched tensors [B, H, W, C]
                        if len(item.shape) == 4:
                            for i in range(item.shape[0]):
                                all_masks.append(item[i:i+1])
                        elif len(item.shape) == 3:  # [H, W, C]
                            all_masks.append(item.unsqueeze(0))
                        else:
                            all_masks.append(item)
            # Handle direct tensor inputs (fallback)
            elif isinstance(mask_input, torch.Tensor):
                if len(mask_input.shape) == 4:
                    for i in range(mask_input.shape[0]):
                        all_masks.append(mask_input[i:i+1])
                elif len(mask_input.shape) == 3:
                    all_masks.append(mask_input.unsqueeze(0))
                else:
                    all_masks.append(mask_input)

        print(f"[COMBINE MASKS] Total individual masks collected: {len(all_masks)}")

        if not all_masks:
            # Return blank mask if no masks provided
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

        # Ensure all masks are the same size (resize to largest dimensions)
        all_masks = self._ensure_same_size(all_masks)

        # Get dimensions from first mask
        h, w = all_masks[0].shape[1], all_masks[0].shape[2]

        # Start with a black (empty) mask
        combined = torch.zeros((1, h, w, 3), dtype=torch.float32)

        # For each mask, combine by taking the maximum value at each pixel
        # This preserves alpha/grayscale values instead of binarizing to pure white
        for idx, mask in enumerate(all_masks):
            # Take maximum value at each pixel to preserve grayscale/alpha
            combined = torch.max(combined, mask)

            print(f"[COMBINE MASKS] Processed mask {idx+1}/{len(all_masks)}")

        print(f"[COMBINE MASKS] Final combined mask shape: {combined.shape}")
        print(f"[COMBINE MASKS] Returning SINGLE mask (not a list)")

        # Return SINGLE combined mask (not a list!)
        return (combined,)
