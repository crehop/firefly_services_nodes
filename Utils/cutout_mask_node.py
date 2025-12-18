"""
Cutout Mask Node

Cuts out masked pixels from an image and applies mask as alpha channel.
"""

from __future__ import annotations
from typing import Tuple
import torch


class CutoutMaskNode:
    """
    Cut out masked area from image with alpha channel.

    Takes an image and mask, returns RGBA image where:
    - Black mask pixels (0.0) = fully transparent (alpha 0)
    - White mask pixels (1.0) = fully opaque (alpha 1)
    - Grayscale values = gradient alpha transparency

    The RGB values are preserved from the original image.
    Only the alpha channel is determined by the mask.
    """

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "apply_cutout"
    CATEGORY = "api node/Firefly Utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to cut out from (RGB or RGBA)",
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Mask defining cutout area (white=keep, black=transparent)",
                }),
                "mask_image": ("IMAGE", {
                    "tooltip": "Image mask defining cutout area (alternative to mask)",
                }),
            },
        }

    @staticmethod
    def _to_grayscale(tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor to grayscale.
        Input: [B, H, W, C] or [B, H, W]
        Output: [B, H, W]
        """
        if len(tensor.shape) == 4:
            if tensor.shape[3] >= 3:
                # Use standard RGB to grayscale conversion
                gray = 0.299 * tensor[..., 0] + 0.587 * tensor[..., 1] + 0.114 * tensor[..., 2]
            else:
                # Single channel, just take it
                gray = tensor[..., 0]
        else:
            # Already grayscale
            gray = tensor
        return gray

    def apply_cutout(
        self,
        image: torch.Tensor,
        mask: torch.Tensor = None,
        mask_image: torch.Tensor = None,
    ) -> Tuple[torch.Tensor]:
        """Apply mask as alpha channel to create cutout image."""

        # Validate inputs
        if mask is None and mask_image is None:
            raise ValueError("Must provide either 'mask' or 'mask_image' input")

        # Use mask if provided, otherwise use mask_image
        if mask is not None:
            mask_input = mask
            print(f"[CUTOUT MASK] Using MASK input, shape: {mask.shape}")
        else:
            mask_input = mask_image
            print(f"[CUTOUT MASK] Using IMAGE mask input, shape: {mask_image.shape}")

        print(f"[CUTOUT MASK] Input image shape: {image.shape}")

        # Get batch size and dimensions
        batch_size = image.shape[0]
        height = image.shape[1]
        width = image.shape[2]

        # Ensure mask is grayscale [B, H, W]
        if len(mask_input.shape) == 4:
            mask_gray = self._to_grayscale(mask_input)
        else:
            mask_gray = mask_input

        # Ensure mask matches image dimensions
        if mask_gray.shape[0] != batch_size:
            if mask_gray.shape[0] == 1:
                # Broadcast single mask to all images in batch
                mask_gray = mask_gray.repeat(batch_size, 1, 1)
                print(f"[CUTOUT MASK] Broadcasted mask to batch size {batch_size}")
            else:
                raise ValueError(
                    f"Mask batch size ({mask_gray.shape[0]}) doesn't match image batch size ({batch_size})"
                )

        if mask_gray.shape[1] != height or mask_gray.shape[2] != width:
            # Resize mask to match image dimensions
            print(f"[CUTOUT MASK] Warning: Mask size ({mask_gray.shape[1]}x{mask_gray.shape[2]}) "
                  f"doesn't match image size ({height}x{width})")
            # Use interpolation to resize
            mask_gray = torch.nn.functional.interpolate(
                mask_gray.unsqueeze(1),  # Add channel dim [B, 1, H, W]
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # Remove channel dim [B, H, W]
            print(f"[CUTOUT MASK] Resized mask to {mask_gray.shape}")

        # Extract RGB channels from image (ignore alpha if present)
        if image.shape[3] >= 3:
            rgb = image[..., :3]  # [B, H, W, 3]
        else:
            # Grayscale image, convert to RGB
            rgb = image.repeat(1, 1, 1, 3)  # [B, H, W, 3]

        # Use mask as alpha channel [B, H, W] -> [B, H, W, 1]
        alpha = mask_gray.unsqueeze(-1)

        # Combine RGB + Alpha to create RGBA image [B, H, W, 4]
        rgba = torch.cat([rgb, alpha], dim=-1)

        print(f"[CUTOUT MASK] Output RGBA shape: {rgba.shape}")
        print(f"[CUTOUT MASK] Alpha range: {alpha.min().item():.3f} to {alpha.max().item():.3f}")

        # Output mask is the grayscale mask [B, H, W]
        output_mask = mask_gray

        return (rgba, output_mask)


# Node registration
NODE_CLASS_MAPPINGS = {
    "CutoutMaskNode": CutoutMaskNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CutoutMaskNode": "Cutout Mask",
}
