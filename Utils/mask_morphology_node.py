"""
Mask Morphology Node

Applies morphological operations (dilate/erode) and blur/feather to masks.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


class MaskMorphologyNode:
    """
    Apply morphological operations and blur to masks or images.

    Features:
    - Dilate (expand) or Erode (shrink) mask areas
    - Separate control for X axis, Y axis, and combined XY
    - Blur for smooth edge transitions
    - Random feather for irregular, natural-looking edges
    - Mask fade for overall opacity control
    - Directional erode for gradient-based edge removal (with fade)
    - Linear erode for hard-cutoff edge removal (no fade)
    - Accepts MASK or IMAGE inputs
    - Outputs both MASK and IMAGE formats

    Parameters:
    - xy_amount: Apply to both X and Y axes simultaneously (applied first)
    - horizontal: Horizontal stretch - apply to X axis only (applied after xy_amount)
    - vertical: Vertical stretch - apply to Y axis only (applied after xy_amount)
    - blur_amount: Gaussian blur for edge smoothing
    - feather_amount: Random pixel removal from edges - creates irregular edges
    - mask_fade: Overall mask opacity/brightness control
    - directional_erode_x: Erode X axis with gradient fade (left/right direction)
    - directional_erode_y: Erode Y axis with gradient fade (top/bottom direction)
    - linear_erode_x: Erode X axis with hard cutoff line (left/right direction)
    - linear_erode_y: Erode Y axis with hard cutoff line (top/bottom direction)

    Example: xy_amount=1, horizontal=1, vertical=0
    1. Dilate by 1 in both X and Y (xy_amount=1)
    2. Dilate by 1 more in X only (horizontal=1) - doubles X expansion
    3. No additional Y dilation (vertical=0)
    """

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "image")
    FUNCTION = "process_mask"
    CATEGORY = "api node/Firefly Utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Input mask to process",
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image to process (alternative to mask)",
                }),
                "xy_amount": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Dilate/erode both X and Y axes (applied first). Positive = dilate, Negative = erode",
                }),
                "horizontal": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Horizontal stretch (applied after xy_amount). Positive = expand, Negative = shrink",
                }),
                "vertical": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Vertical stretch (applied after xy_amount). Positive = expand, Negative = shrink",
                }),
                "blur_amount": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Gaussian blur amount for edge smoothing (0 = no blur)",
                }),
                "feather_amount": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Random feather - randomly removes pixels from outside edge inward. Larger = removes more pixels, creates irregular natural-looking edges (0 = no feather)",
                }),
                "mask_fade": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Overall mask opacity/fade. Positive = brighten/fade in, Negative = darken/fade out",
                }),
                "directional_erode_x": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Directional X erode with gradient. Positive = erode left-to-right, Negative = right-to-left. 50 = black left 50%, fade 50-100%. 100 = fully black.",
                }),
                "directional_erode_y": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Directional Y erode with gradient. Positive = erode top-to-bottom, Negative = bottom-to-top. 50 = black top 50%, fade 50-100%. 100 = fully black.",
                }),
                "linear_erode_x": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Linear X erode - hard cutoff line. Positive = delete left side, Negative = delete right side. Value is % of width.",
                }),
                "linear_erode_y": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Linear Y erode - hard cutoff line. Positive = delete top side, Negative = delete bottom side. Value is % of height.",
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
            # RGB/RGBA to grayscale
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

    @staticmethod
    def _dilate_directional(mask: torch.Tensor, x_amount: float, y_amount: float) -> torch.Tensor:
        """
        Dilate mask with directional control (expand white areas).
        Input: [B, H, W] grayscale mask
        Output: [B, H, W] dilated mask
        """
        if x_amount <= 0 and y_amount <= 0:
            return mask

        # Convert to [B, 1, H, W] for max pooling
        mask_4d = mask.unsqueeze(1)

        # Calculate kernel sizes based on amounts
        x_kernel = int(x_amount * 2) + 1 if x_amount > 0 else 1
        y_kernel = int(y_amount * 2) + 1 if y_amount > 0 else 1

        # Ensure odd kernel sizes
        if x_kernel % 2 == 0:
            x_kernel += 1
        if y_kernel % 2 == 0:
            y_kernel += 1

        # Use max pooling to dilate with directional kernel
        dilated = F.max_pool2d(
            mask_4d,
            kernel_size=(y_kernel, x_kernel),  # (height, width)
            stride=1,
            padding=(y_kernel // 2, x_kernel // 2)
        )

        return dilated.squeeze(1)

    @staticmethod
    def _erode_directional(mask: torch.Tensor, x_amount: float, y_amount: float) -> torch.Tensor:
        """
        Erode mask with directional control (shrink white areas).
        Input: [B, H, W] grayscale mask
        Output: [B, H, W] eroded mask
        """
        if x_amount <= 0 and y_amount <= 0:
            return mask

        # Invert mask, dilate, then invert back
        inverted = 1.0 - mask
        dilated = MaskMorphologyNode._dilate_directional(inverted, x_amount, y_amount)
        eroded = 1.0 - dilated

        return eroded

    @staticmethod
    def _gaussian_blur(mask: torch.Tensor, blur_amount: float) -> torch.Tensor:
        """
        Apply Gaussian blur to mask.
        Input: [B, H, W] grayscale mask
        Output: [B, H, W] blurred mask
        """
        if blur_amount <= 0:
            return mask

        # Convert to [B, 1, H, W] for convolution
        mask_4d = mask.unsqueeze(1)

        # Calculate kernel size and sigma based on blur amount
        kernel_size = int(blur_amount * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size

        sigma = blur_amount / 3.0  # Rule of thumb: sigma ≈ kernel_size / 6

        # Create Gaussian kernel
        kernel_range = torch.arange(kernel_size, dtype=mask.dtype, device=mask.device)
        kernel_range = kernel_range - kernel_size // 2
        gauss_1d = torch.exp(-0.5 * (kernel_range / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()

        # Create 2D Gaussian kernel
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d / gauss_2d.sum()

        # Reshape for conv2d: [out_channels, in_channels, H, W]
        kernel = gauss_2d.unsqueeze(0).unsqueeze(0)

        # Apply convolution with padding
        padding = kernel_size // 2
        blurred = F.conv2d(mask_4d, kernel, padding=padding)

        return blurred.squeeze(1)

    @staticmethod
    def _gradient_feather(mask: torch.Tensor, feather_amount: float) -> torch.Tensor:
        """
        Apply dithered feather to mask edges - creates halftone/dithered pattern.
        Creates a pixelated, dithered transition like halftone printing.
        Larger feather_amount = wider dithered transition zone.

        Input: [B, H, W] grayscale mask
        Output: [B, H, W] feathered mask with dithered edges
        """
        if feather_amount <= 0:
            return mask

        B, H, W = mask.shape
        device = mask.device
        dtype = mask.dtype

        # Convert to numpy for easier processing
        result = mask.clone()

        # Calculate maximum distance for feather zone
        max_distance = int(feather_amount * 10) + 1

        # For each batch
        for b in range(B):
            mask_np = result[b].cpu().numpy()

            # Binary mask: 1 where mask > 0.5, 0 elsewhere
            binary_mask = (mask_np > 0.5).astype(np.uint8)

            if binary_mask.sum() == 0:
                # Empty mask, skip
                continue

            # Compute distance transform (distance from edge)
            distance_from_edge = ndimage.distance_transform_edt(binary_mask)

            # Normalize distance to 0-1 range based on max_distance
            # This creates our gradient from edge (0) to interior (1)
            normalized_distance = np.clip(distance_from_edge / max_distance, 0.0, 1.0)

            # Create a dithered pattern using ordered dithering (Bayer matrix approach)
            # This creates the halftone/pixelated effect shown in the image

            # Create threshold matrix - use a simple pattern for dithering
            # Larger patterns create more visible dithering effect
            bayer_size = 8  # 8x8 Bayer matrix for nice dithering

            # Create a tiled Bayer-like pattern across the image
            # Use random thresholds in each cell for organic look
            threshold_map = np.random.random((H, W))

            # Apply ordered dithering: compare gradient value to threshold
            # If gradient > threshold, keep pixel as 1; otherwise set to 0
            # This creates the characteristic dithered/halftone pattern
            dithered = (normalized_distance > threshold_map).astype(np.float32)

            # Apply the dithered pattern to the mask
            # Pixels inside the mask stay 1, pixels in feather zone get dithered,
            # pixels outside stay 0
            mask_np = mask_np * dithered

            # Convert back to tensor
            result[b] = torch.from_numpy(mask_np).to(device=device, dtype=dtype)

        return result

    @staticmethod
    def _mask_fade(mask: torch.Tensor, fade_amount: float) -> torch.Tensor:
        """
        Apply overall fade/opacity to mask.
        Positive values = brighten/fade in
        Negative values = darken/fade out

        Input: [B, H, W] grayscale mask
        Output: [B, H, W] faded mask
        """
        if fade_amount == 0:
            return mask

        # Convert fade amount to multiplier
        # fade_amount range: -100 to 100
        # We'll map this to a reasonable opacity range
        if fade_amount > 0:
            # Positive: brighten (multiply by value > 1, up to 2x)
            multiplier = 1.0 + (fade_amount / 100.0)
        else:
            # Negative: darken (multiply by value < 1, down to 0)
            multiplier = 1.0 + (fade_amount / 100.0)  # This will be < 1

        result = mask * multiplier
        return torch.clamp(result, 0.0, 1.0)

    @staticmethod
    def _directional_erode_gradient(mask: torch.Tensor, x_percent: float, y_percent: float) -> torch.Tensor:
        """
        Apply directional erosion as a gradient based on percentage.

        X axis:
        - Positive x_percent: erode from left to right
        - Negative x_percent: erode from right to left
        - 50 = black left 50%, fade 50-100%
        - 100 = fully black across entire image
        - -50 = black right 50%, fade 0-50%
        - -100 = fully black across entire image

        Y axis:
        - Positive y_percent: erode from top to bottom
        - Negative y_percent: erode from bottom to top
        - Same percentage behavior as X axis

        Input: [B, H, W] grayscale mask
        Output: [B, H, W] eroded mask
        """
        if x_percent == 0 and y_percent == 0:
            return mask

        B, H, W = mask.shape
        device = mask.device
        dtype = mask.dtype

        # Create fresh result tensor (don't modify input)
        result = mask.clone()

        # Apply X-axis gradient
        if x_percent != 0:
            x_coords = torch.linspace(0, 1, W, device=device, dtype=dtype)
            abs_percent = abs(x_percent) / 100.0

            if x_percent < 0:
                # Negative: erode from right to left
                # At -50%: black from x=0.5 to x=1.0, fade from x=0 to x=0.5
                # At -100%: black from x=0 to x=1.0 (fully black)
                fade_end = 1.0 - abs_percent             # At -50%: 0.5, At -100%: 0.0
                fade_start = 1.0 - (abs_percent * 2.0)   # At -50%: 0.0, At -100%: -1.0
                fade_start = max(fade_start, 0.0)        # Clamp to 0
                if fade_end <= fade_start:
                    x_gradient = torch.zeros_like(x_coords)  # Fully black
                else:
                    x_gradient = torch.clamp((fade_end - x_coords) / (fade_end - fade_start + 1e-6), 0.0, 1.0)
            else:
                # Positive: erode from left to right
                # At 50%: black from x=0 to x=0.5, fade from x=0.5 to x=1.0
                # At 100%: black from x=0 to x=1.0 (fully black)
                fade_start = abs_percent                 # At 50%: 0.5, At 100%: 1.0
                fade_end = abs_percent * 2.0             # At 50%: 1.0, At 100%: 2.0
                fade_end = min(fade_end, 1.0)            # Clamp to 1
                if fade_end <= fade_start:
                    x_gradient = torch.zeros_like(x_coords)  # Fully black
                else:
                    x_gradient = torch.clamp((x_coords - fade_start) / (fade_end - fade_start + 1e-6), 0.0, 1.0)

            # Broadcast and apply to result
            x_gradient = x_gradient.view(1, 1, W).expand(B, H, W)
            result = result * x_gradient

        # Apply Y-axis gradient
        if y_percent != 0:
            y_coords = torch.linspace(0, 1, H, device=device, dtype=dtype)
            abs_percent = abs(y_percent) / 100.0

            if y_percent < 0:
                # Negative: erode from bottom to top
                # At -50%: black from y=0.5 to y=1.0, fade from y=0 to y=0.5
                # At -100%: black from y=0 to y=1.0 (fully black)
                fade_end = 1.0 - abs_percent             # At -50%: 0.5, At -100%: 0.0
                fade_start = 1.0 - (abs_percent * 2.0)   # At -50%: 0.0, At -100%: -1.0
                fade_start = max(fade_start, 0.0)        # Clamp to 0
                if fade_end <= fade_start:
                    y_gradient = torch.zeros_like(y_coords)  # Fully black
                else:
                    y_gradient = torch.clamp((fade_end - y_coords) / (fade_end - fade_start + 1e-6), 0.0, 1.0)
            else:
                # Positive: erode from top to bottom
                # At 50%: black from y=0 to y=0.5, fade from y=0.5 to y=1.0
                # At 100%: black from y=0 to y=1.0 (fully black)
                fade_start = abs_percent                 # At 50%: 0.5, At 100%: 1.0
                fade_end = abs_percent * 2.0             # At 50%: 1.0, At 100%: 2.0
                fade_end = min(fade_end, 1.0)            # Clamp to 1
                if fade_end <= fade_start:
                    y_gradient = torch.zeros_like(y_coords)  # Fully black
                else:
                    y_gradient = torch.clamp((y_coords - fade_start) / (fade_end - fade_start + 1e-6), 0.0, 1.0)

            # Broadcast and apply to result
            y_gradient = y_gradient.view(1, H, 1).expand(B, H, W)
            result = result * y_gradient

        return result

    @staticmethod
    def _linear_erode(mask: torch.Tensor, x_percent: float, y_percent: float) -> torch.Tensor:
        """
        Apply linear erosion with hard cutoff line based on percentage.
        All pixels beyond the erode line are set to 0 (black/deleted).
        All pixels inside the line are unchanged (keep original value).

        X axis:
        - Positive x_percent: delete left side (0 to x_percent)
        - Negative x_percent: delete right side (x_percent to 100)
        - 50 = delete left 50%, -50 = delete right 50%

        Y axis:
        - Positive y_percent: delete top side (0 to y_percent)
        - Negative y_percent: delete bottom side (y_percent to 100)
        - 50 = delete top 50%, -50 = delete bottom 50%

        Input: [B, H, W] grayscale mask
        Output: [B, H, W] eroded mask
        """
        if x_percent == 0 and y_percent == 0:
            return mask

        B, H, W = mask.shape
        device = mask.device
        dtype = mask.dtype

        # Create fresh result tensor (don't modify input)
        result = mask.clone()

        # Apply X-axis linear cutoff
        if x_percent != 0:
            x_coords = torch.linspace(0, 1, W, device=device, dtype=dtype)
            abs_percent = abs(x_percent) / 100.0

            if x_percent < 0:
                # Negative: delete right side
                # Create mask where x > (1.0 - abs_percent) = 0, else = 1
                cutoff_line = 1.0 - abs_percent
                x_mask = (x_coords < cutoff_line).float()
            else:
                # Positive: delete left side
                # Create mask where x < abs_percent = 0, else = 1
                x_mask = (x_coords >= abs_percent).float()

            # Broadcast and apply to result
            x_mask = x_mask.view(1, 1, W).expand(B, H, W)
            result = result * x_mask

        # Apply Y-axis linear cutoff
        if y_percent != 0:
            y_coords = torch.linspace(0, 1, H, device=device, dtype=dtype)
            abs_percent = abs(y_percent) / 100.0

            if y_percent < 0:
                # Negative: delete bottom side
                # Create mask where y > (1.0 - abs_percent) = 0, else = 1
                cutoff_line = 1.0 - abs_percent
                y_mask = (y_coords < cutoff_line).float()
            else:
                # Positive: delete top side
                # Create mask where y < abs_percent = 0, else = 1
                y_mask = (y_coords >= abs_percent).float()

            # Broadcast and apply to result
            y_mask = y_mask.view(1, H, 1).expand(B, H, W)
            result = result * y_mask

        return result

    def process_mask(
        self,
        mask: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        xy_amount: float = 0.0,
        horizontal: float = 0.0,
        vertical: float = 0.0,
        blur_amount: float = 0.0,
        feather_amount: float = 0.0,
        mask_fade: float = 0.0,
        directional_erode_x: float = 0.0,
        directional_erode_y: float = 0.0,
        linear_erode_x: float = 0.0,
        linear_erode_y: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process mask with directional morphology, blur, and gradient feather operations."""

        # Validate inputs
        if mask is None and image is None:
            raise ValueError("Must provide either 'mask' or 'image' input")

        # Use mask if provided, otherwise use image
        if mask is not None:
            input_tensor = mask
            print(f"[MASK MORPHOLOGY] Processing MASK input, shape: {mask.shape}")
        else:
            input_tensor = image
            print(f"[MASK MORPHOLOGY] Processing IMAGE input, shape: {image.shape}")

        # Convert to grayscale [B, H, W]
        gray_mask = self._to_grayscale(input_tensor)
        print(f"[MASK MORPHOLOGY] Grayscale mask shape: {gray_mask.shape}")

        # Step 1: Apply XY morphology (both axes simultaneously)
        if xy_amount != 0:
            if xy_amount > 0:
                print(f"[MASK MORPHOLOGY] Step 1: Applying XY dilation: {xy_amount:.1f}")
                gray_mask = self._dilate_directional(gray_mask, xy_amount, xy_amount)
            else:
                print(f"[MASK MORPHOLOGY] Step 1: Applying XY erosion: {abs(xy_amount):.1f}")
                gray_mask = self._erode_directional(gray_mask, abs(xy_amount), abs(xy_amount))
        else:
            print("[MASK MORPHOLOGY] Step 1: No XY morphology applied")

        # Step 2: Apply horizontal stretch (after XY)
        if horizontal != 0:
            if horizontal > 0:
                print(f"[MASK MORPHOLOGY] Step 2: Applying horizontal expansion: {horizontal:.1f}")
                gray_mask = self._dilate_directional(gray_mask, horizontal, 0)
            else:
                print(f"[MASK MORPHOLOGY] Step 2: Applying horizontal shrink: {abs(horizontal):.1f}")
                gray_mask = self._erode_directional(gray_mask, abs(horizontal), 0)
        else:
            print("[MASK MORPHOLOGY] Step 2: No horizontal stretch applied")

        # Step 3: Apply vertical stretch (after XY and horizontal)
        if vertical != 0:
            if vertical > 0:
                print(f"[MASK MORPHOLOGY] Step 3: Applying vertical expansion: {vertical:.1f}")
                gray_mask = self._dilate_directional(gray_mask, 0, vertical)
            else:
                print(f"[MASK MORPHOLOGY] Step 3: Applying vertical shrink: {abs(vertical):.1f}")
                gray_mask = self._erode_directional(gray_mask, 0, abs(vertical))
        else:
            print("[MASK MORPHOLOGY] Step 3: No vertical stretch applied")

        # Step 4: Apply gradient feather (spray paint effect)
        if feather_amount > 0:
            print(f"[MASK MORPHOLOGY] Step 4: Applying gradient feather: {feather_amount:.1f}")
            gray_mask = self._gradient_feather(gray_mask, feather_amount)
        else:
            print("[MASK MORPHOLOGY] Step 4: No gradient feather applied")

        # Step 5: Apply Gaussian blur
        if blur_amount > 0:
            print(f"[MASK MORPHOLOGY] Step 5: Applying Gaussian blur: {blur_amount:.1f}")
            gray_mask = self._gaussian_blur(gray_mask, blur_amount)
        else:
            print("[MASK MORPHOLOGY] Step 5: No blur applied")

        # Step 6: Apply mask fade
        if mask_fade != 0:
            print(f"[MASK MORPHOLOGY] Step 6: Applying mask fade: {mask_fade:.1f}")
            gray_mask = self._mask_fade(gray_mask, mask_fade)
        else:
            print("[MASK MORPHOLOGY] Step 6: No mask fade applied")

        # Step 7: Apply directional erode
        if directional_erode_x != 0 or directional_erode_y != 0:
            print(f"[MASK MORPHOLOGY] Step 7: Applying directional erode - X: {directional_erode_x:.1f}, Y: {directional_erode_y:.1f}")
            gray_mask = self._directional_erode_gradient(gray_mask, directional_erode_x, directional_erode_y)
        else:
            print("[MASK MORPHOLOGY] Step 7: No directional erode applied")

        # Step 8: Apply linear erode (hard cutoff)
        if linear_erode_x != 0 or linear_erode_y != 0:
            print(f"[MASK MORPHOLOGY] Step 8: Applying linear erode - X: {linear_erode_x:.1f}, Y: {linear_erode_y:.1f}")
            gray_mask = self._linear_erode(gray_mask, linear_erode_x, linear_erode_y)
        else:
            print("[MASK MORPHOLOGY] Step 8: No linear erode applied")

        # Clamp values to [0, 1]
        gray_mask = torch.clamp(gray_mask, 0.0, 1.0)

        # Prepare outputs
        # MASK output: [B, H, W]
        mask_output = gray_mask

        # IMAGE output: [B, H, W, 3] (convert grayscale to RGB for visualization)
        image_output = gray_mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        print(f"[MASK MORPHOLOGY] Output shapes - MASK: {mask_output.shape}, IMAGE: {image_output.shape}")
        print(f"[MASK MORPHOLOGY] Mask value range: {mask_output.min().item():.3f} to {mask_output.max().item():.3f}")
        print(f"[MASK MORPHOLOGY] Image value range: {image_output.min().item():.3f} to {image_output.max().item():.3f}")
        print(f"[MASK MORPHOLOGY] Non-zero pixels in mask: {(mask_output > 0).sum().item()}/{mask_output.numel()}")

        return (mask_output, image_output)
