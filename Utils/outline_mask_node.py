"""
Outline Mask Node

Creates an outline effect around mask boundaries with inbleed and outbleed controls.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


class OutlineMaskNode:
    """
    Create an outline around mask boundaries.

    Features:
    - Creates an outline/border of the mask edge
    - Inbleed: Expand outline inward (inside the mask)
    - Outbleed: Expand outline outward (outside the mask)
    - Blur: Distance-based gradient fade from edge
    - Feather: Random ice crystal-like dropout from edge
    - Accepts MASK or IMAGE inputs
    - Outputs both MASK and IMAGE formats

    Parameters:
    - inbleed: Expand outline inward into the mask (pixels)
    - outbleed: Expand outline outward from the mask (pixels)
    - blur_amount: Distance for gradient fade (white at edge → black outward)
    - feather_amount: Distance for random pixel dropout (0% at edge → 95% outward)

    Example: inbleed=5, outbleed=5
    - Creates a 10-pixel wide outline centered on the mask edge
    - 5 pixels inside, 5 pixels outside
    """

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "image")
    FUNCTION = "create_outline"
    CATEGORY = "api node/Firefly Utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Input mask to create outline from",
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image to create outline from (alternative to mask)",
                }),
                "inbleed": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Expand outline inward into the mask (pixels)",
                }),
                "outbleed": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Expand outline outward from the mask (pixels)",
                }),
                "blur_amount": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Distance-based gradient fade from edge line (white→black). Distance in pixels from edge where fade reaches black.",
                }),
                "feather_amount": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Random pixel dropout creating ice crystal patterns. Distance in pixels from edge where dropout reaches 95% probability.",
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
                gray = 0.299 * tensor[..., 0] + 0.587 * tensor[..., 1] + 0.114 * tensor[..., 2]
            else:
                gray = tensor[..., 0]
        else:
            gray = tensor
        return gray

    @staticmethod
    def _dilate(mask: torch.Tensor, amount: float) -> torch.Tensor:
        """
        Dilate mask (expand white areas).
        Input: [B, H, W] grayscale mask
        Output: [B, H, W] dilated mask
        """
        if amount <= 0:
            return mask

        mask_4d = mask.unsqueeze(1)
        kernel_size = int(amount * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        dilated = F.max_pool2d(
            mask_4d,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )

        return dilated.squeeze(1)

    @staticmethod
    def _erode(mask: torch.Tensor, amount: float) -> torch.Tensor:
        """
        Erode mask (shrink white areas).
        Input: [B, H, W] grayscale mask
        Output: [B, H, W] eroded mask
        """
        if amount <= 0:
            return mask

        # Invert, dilate, invert back
        inverted = 1.0 - mask
        dilated = OutlineMaskNode._dilate(inverted, amount)
        eroded = 1.0 - dilated

        return eroded

    @staticmethod
    def _detect_edge(mask: torch.Tensor) -> torch.Tensor:
        """
        Detect the edge/boundary of the mask.
        Input: [B, H, W] grayscale mask
        Output: [B, H, W] edge mask (1-pixel wide boundary)
        """
        # Threshold input mask to binary
        binary_mask = (mask > 0.5).float()

        # Dilate and erode by 1 pixel, then take the difference
        dilated = OutlineMaskNode._dilate(binary_mask, 1.0)
        eroded = OutlineMaskNode._erode(binary_mask, 1.0)

        # Edge is the difference between dilated and eroded
        edge = dilated - eroded

        # Threshold to pure binary (only 0.0 or 1.0)
        edge = (edge > 0.5).float()

        return edge

    @staticmethod
    def _calculate_distance_from_edge(mask: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate distance of each pixel from the original edge line.
        Input: [B, H, W] mask and edge_mask
        Output: [B, H, W] distance map
        """
        # Use dilate operations to calculate distance
        # Each dilation step is 1 pixel distance
        distance_map = torch.zeros_like(mask)
        current_edge = edge_mask.clone()

        max_distance = 100  # Maximum distance to calculate
        for dist in range(1, max_distance + 1):
            # Dilate the edge by 1 pixel
            dilated = OutlineMaskNode._dilate(current_edge, 1.0)
            # New pixels are those in dilated but not in current_edge
            new_pixels = (dilated > 0.5) & (current_edge <= 0.5) & (mask > 0.5)
            # Assign distance to new pixels
            distance_map = torch.where(new_pixels, torch.tensor(float(dist), device=mask.device), distance_map)
            # Update current edge
            current_edge = dilated
            # Stop if no more mask pixels to process
            if not new_pixels.any():
                break

        return distance_map

    @staticmethod
    def _apply_distance_blur(mask: torch.Tensor, distance_map: torch.Tensor, blur_amount: float) -> torch.Tensor:
        """
        Apply distance-based gradient blur from edge outward.
        Respects pixels already removed by feather.
        Input: [B, H, W] grayscale mask and pre-calculated distance map
        Output: [B, H, W] blurred mask
        """
        if blur_amount <= 0:
            return mask

        # Create gradient: 1.0 at edge, 0.0 at blur_amount distance
        # distance / blur_amount gives 0.0 at edge, 1.0 at blur_amount
        # 1.0 - that gives 1.0 at edge, 0.0 at blur_amount
        gradient = torch.clamp(1.0 - (distance_map / blur_amount), 0.0, 1.0)

        # Apply gradient only where mask still has pixels (respects feather)
        # Use min instead of multiply to preserve feathered pixels at 0
        return torch.min(mask, gradient)

    @staticmethod
    def _apply_random_feather(mask: torch.Tensor, distance_map: torch.Tensor, feather_amount: float) -> torch.Tensor:
        """
        Apply random pixel dropout based on distance from edge.
        Creates ice crystal-like patterns around edges.
        Only removes pixels, never adds them.
        Input: [B, H, W] grayscale mask and pre-calculated distance map
        Output: [B, H, W] feathered mask
        """
        if feather_amount <= 0:
            return mask

        # Only operate on pixels that exist in the mask
        existing_pixels = mask > 0.5

        # Calculate dropout probability: 0% at edge, 95% at feather_amount distance
        # Only calculate for existing pixels
        dropout_probability = torch.clamp((distance_map / feather_amount) * 0.95, 0.0, 0.95)
        # Zero out dropout probability for non-existing pixels
        dropout_probability = dropout_probability * existing_pixels.float()

        # Generate random values [0, 1] for each pixel
        random_values = torch.rand_like(mask)

        # Keep pixel if random value > dropout probability
        # This means lower dropout_probability = more likely to keep
        keep_mask = random_values > dropout_probability

        # Apply dropout - only to existing pixels
        # Ensure we never add pixels (use min of original mask and keep_mask)
        result = mask * keep_mask.float()

        # Double-check: ensure result never has pixels where mask didn't
        result = result * existing_pixels.float()

        return result

    def create_outline(
        self,
        mask: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        inbleed: float = 0.0,
        outbleed: float = 0.0,
        blur_amount: float = 0.0,
        feather_amount: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create outline effect around mask boundaries."""

        # Validate inputs
        if mask is None and image is None:
            raise ValueError("Must provide either 'mask' or 'image' input")

        # Use mask if provided, otherwise use image
        if mask is not None:
            input_tensor = mask
            print(f"[OUTLINE MASK] Processing MASK input, shape: {mask.shape}")
        else:
            input_tensor = image
            print(f"[OUTLINE MASK] Processing IMAGE input, shape: {image.shape}")

        # Convert to grayscale [B, H, W]
        gray_mask = self._to_grayscale(input_tensor)
        print(f"[OUTLINE MASK] Grayscale mask shape: {gray_mask.shape}")

        # Step 1: Detect edge (1-pixel boundary)
        print("[OUTLINE MASK] Step 1: Detecting edge boundary")
        edge = self._detect_edge(gray_mask)

        # Store original edge for distance calculations
        original_edge = edge.clone()

        # Step 2: Expand edge inward (inbleed)
        if inbleed > 0:
            print(f"[OUTLINE MASK] Step 2: Applying inbleed: {inbleed:.1f} pixels")
            # Dilate the edge, but only inside the original mask
            edge_dilated = self._dilate(edge, inbleed)
            # Mask it to only keep areas inside the original mask
            binary_gray_mask = (gray_mask > 0.5).float()
            edge = torch.min(edge_dilated, binary_gray_mask)
            # Threshold to binary
            edge = (edge > 0.5).float()
        else:
            print("[OUTLINE MASK] Step 2: No inbleed applied")

        # Step 3: Expand edge outward (outbleed)
        if outbleed > 0:
            print(f"[OUTLINE MASK] Step 3: Applying outbleed: {outbleed:.1f} pixels")
            # Dilate the edge outward
            edge = self._dilate(edge, outbleed)
            # Threshold to binary
            edge = (edge > 0.5).float()
        else:
            print("[OUTLINE MASK] Step 3: No outbleed applied")

        # Step 3.5: Calculate distance map ONCE for both feather and blur
        # This ensures both effects use the same distance reference
        if feather_amount > 0 or blur_amount > 0:
            print("[OUTLINE MASK] Step 3.5: Calculating distance map from original edge")
            distance_map = self._calculate_distance_from_edge(edge, original_edge)
        else:
            distance_map = None

        # Step 4: Apply random feather (ice crystal dropout) - binary operation
        if feather_amount > 0:
            print(f"[OUTLINE MASK] Step 4: Applying random feather: {feather_amount:.1f}")
            edge = self._apply_random_feather(edge, distance_map, feather_amount)
        else:
            print("[OUTLINE MASK] Step 4: No feather applied")

        # Step 5: Apply distance-based blur gradient - RUNS LAST to preserve gradient
        if blur_amount > 0:
            print(f"[OUTLINE MASK] Step 5: Applying distance blur: {blur_amount:.1f}")
            edge = self._apply_distance_blur(edge, distance_map, blur_amount)
        else:
            # No blur = ensure pure binary output (only 0.0 or 1.0)
            print("[OUTLINE MASK] Step 5: No blur applied, ensuring binary output")
            edge = (edge > 0.5).float()

        # Clamp values to [0, 1]
        edge = torch.clamp(edge, 0.0, 1.0)

        # Prepare outputs
        mask_output = edge
        image_output = edge.unsqueeze(-1).repeat(1, 1, 1, 3)

        print(f"[OUTLINE MASK] Output shapes - MASK: {mask_output.shape}, IMAGE: {image_output.shape}")

        return (mask_output, image_output)


# Node registration
NODE_CLASS_MAPPINGS = {
    "OutlineMaskNode": OutlineMaskNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OutlineMaskNode": "Outline Mask",
}
