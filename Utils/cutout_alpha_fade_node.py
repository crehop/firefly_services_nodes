"""
Cutout with Alpha Fade Node

Applies a mask to an image with alpha channel fade where:
- White pixels (255,255,255) = 100% transparency (fully cut out)
- Black pixels (0,0,0) = 0% transparency (fully opaque)
- Gray values = partial transparency with smooth fade
"""

import torch
import numpy as np
from PIL import Image


class CutoutAlphaFadeNode:
    """
    Apply mask to image with alpha fade.

    Takes white pixels and cuts out that part of the picture,
    fading alpha channel as mask approaches true black.
    White = fully transparent, Black = fully opaque.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to apply cutout"
                }),
                "mask": ("MASK", {
                    "tooltip": "Mask where white = transparent, black = opaque"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_cutout"
    CATEGORY = "Firefly Utils"

    def apply_cutout(self, image: torch.Tensor, mask: torch.Tensor):
        """
        Apply mask to image with alpha fade.

        Args:
            image: Input image tensor [B, H, W, C]
            mask: Input mask tensor [B, H, W] where 0=opaque, 1=transparent

        Returns:
            RGBA image tensor with alpha channel applied
        """
        # Get batch size
        batch_size = image.shape[0]

        # Process each image in batch
        output_images = []

        for i in range(batch_size):
            # Get single image and mask
            img = image[i]  # [H, W, C]
            msk = mask[i] if i < mask.shape[0] else mask[0]  # [H, W]

            # Convert image to numpy (0-255 range)
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)

            # Convert mask to numpy (0-255 range)
            # In ComfyUI masks: 0 = unmasked/opaque, 1 = masked/transparent
            # We want: white (1.0) = transparent, black (0.0) = opaque
            mask_np = (msk.cpu().numpy() * 255).astype(np.uint8)

            # Ensure mask matches image dimensions
            h, w = img_np.shape[:2]
            if mask_np.shape != (h, w):
                # Resize mask to match image
                mask_pil = Image.fromarray(mask_np, mode='L')
                mask_pil = mask_pil.resize((w, h), Image.Resampling.BILINEAR)
                mask_np = np.array(mask_pil)

            # Create PIL image
            if img_np.shape[2] == 3:
                img_pil = Image.fromarray(img_np, mode='RGB')
            elif img_np.shape[2] == 4:
                img_pil = Image.fromarray(img_np, mode='RGBA')
            else:
                # Grayscale - convert to RGB
                img_pil = Image.fromarray(img_np[:, :, 0], mode='L').convert('RGB')

            # Convert to RGBA if not already
            if img_pil.mode != 'RGBA':
                img_pil = img_pil.convert('RGBA')

            # Get image data as numpy array
            img_rgba = np.array(img_pil)

            # Apply mask as alpha channel
            # mask_np is 0-255 where 255 should be transparent
            # Alpha channel should be inverted: 255=opaque, 0=transparent
            # So we need to invert the mask: alpha = 255 - mask_np
            alpha_channel = 255 - mask_np

            # Replace alpha channel
            img_rgba[:, :, 3] = alpha_channel

            # Convert back to PIL
            output_pil = Image.fromarray(img_rgba, mode='RGBA')

            # Convert to tensor (0-1 range)
            output_np = np.array(output_pil).astype(np.float32) / 255.0
            output_tensor = torch.from_numpy(output_np)

            output_images.append(output_tensor)

        # Stack batch
        output_batch = torch.stack(output_images, dim=0)

        return (output_batch,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "CutoutAlphaFadeNode": CutoutAlphaFadeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CutoutAlphaFadeNode": "Cutout with Alpha Fade",
}
