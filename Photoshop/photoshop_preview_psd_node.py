"""
Adobe Photoshop Preview PSD Node

Render PSD files to flattened preview images using psd-tools.
"""

from __future__ import annotations
from typing import Optional
import torch
import numpy as np
import tempfile
import os
import hashlib
import mimetypes
from PIL import Image

# Register PSD MIME type so it's recognized as an image type
if not mimetypes.guess_type('test.psd')[0]:
    mimetypes.add_type('image/vnd.adobe.photoshop', '.psd')

# Import psd-tools for PSD rendering
try:
    from psd_tools import PSDImage
    PSD_TOOLS_AVAILABLE = True
except ImportError:
    PSD_TOOLS_AVAILABLE = False

# Import utility functions
from ..apinode_utils import download_url_to_bytesio

# Import ComfyUI folder utilities
import folder_paths


class PhotoshopPreviewPSDNode:
    """
    Render PSD files to flattened preview images.

    Features:
    - Accept PSD from URL or local file path
    - Render flattened composite using psd-tools
    - Display layer count and dimensions
    - MIT licensed (commercial-friendly)
    """

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "preview_psd"
    CATEGORY = "api node/photoshop"

    @classmethod
    def INPUT_TYPES(cls):
        # Use same pattern as Load Image - filter by content type
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        # Filter for image content types (includes PSDs now)
        files = folder_paths.filter_files_content_types(files, ["image"])
        # Further filter to only PSD files
        files = [f for f in files if f.lower().endswith('.psd')]

        return {
            "required": {},
            "optional": {
                "psd_file": (sorted(files), {"image_upload": True}),
                "psd_source": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "URL or local file path to PSD file (overrides psd_file if provided)",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, psd_file="", psd_source=""):
        # If using psd_source (URL or custom path), always re-execute
        if psd_source and psd_source.strip():
            return float("nan")

        # If using psd_file from picker, check file hash
        if psd_file and psd_file.strip():
            psd_path = folder_paths.get_annotated_filepath(psd_file)
            m = hashlib.sha256()
            with open(psd_path, 'rb') as f:
                m.update(f.read())
            return m.digest().hex()

        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, psd_file="", psd_source=""):
        # At least one input must be provided
        if (not psd_file or psd_file.strip() == "") and (not psd_source or psd_source.strip() == ""):
            return "Must provide either psd_file or psd_source"

        # If psd_file is provided, validate it exists
        if psd_file and psd_file.strip():
            if not folder_paths.exists_annotated_filepath(psd_file):
                return "Invalid PSD file: {}".format(psd_file)
            if not psd_file.lower().endswith('.psd'):
                return "File must be a .psd file"

        return True

    async def preview_psd(
        self,
        psd_file: str = "",
        psd_source: str = "",
    ):
        """Render PSD file to flattened preview image."""

        # Check if psd-tools is available
        if not PSD_TOOLS_AVAILABLE:
            raise ImportError(
                "psd-tools library is required but is not installed.\n"
                "Please install it with: pip install psd-tools"
            )

        # Determine which input to use
        # Priority: psd_source (if provided) > psd_file (from picker)
        if psd_source and psd_source.strip():
            # Use the manually entered URL or path
            source = psd_source.strip()
        elif psd_file and psd_file.strip():
            # Use file from input folder
            source = folder_paths.get_annotated_filepath(psd_file)
        else:
            raise ValueError("Must provide either psd_file (from picker) or psd_source (URL or path)")

        info_log = "=" * 55 + "\n"
        info_log += "Photoshop PSD Preview\n"
        info_log += "-" * 55 + "\n"

        tmp_psd_path = None
        is_temp_file = False

        try:
            # Determine if source is URL or local path
            if source.startswith("http://") or source.startswith("https://"):
                # Download from URL
                info_log += f"Source: URL\n"
                info_log += f"  {source[:80]}{'...' if len(source) > 80 else ''}\n"
                info_log += "\nDownloading PSD file...\n"

                # Use utility function to download
                psd_bytesio = await download_url_to_bytesio(source)
                psd_bytes = psd_bytesio.getvalue()

                info_log += "[OK] Downloaded PSD file\n"
                info_log += f"  Size: {len(psd_bytes)} bytes\n"

                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".psd") as tmp_file:
                    tmp_file.write(psd_bytes)
                    tmp_psd_path = tmp_file.name

                is_temp_file = True

            else:
                # Use local file path
                info_log += f"Source: Local file\n"
                info_log += f"  {source}\n"

                # Check if file exists
                if not os.path.exists(source):
                    raise FileNotFoundError(f"PSD file not found: {source}")

                tmp_psd_path = source
                is_temp_file = False

                # Get file size
                file_size = os.path.getsize(tmp_psd_path)
                info_log += f"  Size: {file_size} bytes\n"

            # Render PSD to flattened image using psd-tools
            info_log += "\nOpening PSD file...\n"

            psd = PSDImage.open(tmp_psd_path)

            info_log += "[OK] PSD opened successfully\n"
            info_log += f"  Dimensions: {psd.width}x{psd.height}\n"
            info_log += f"  Color mode: {psd.mode}\n"
            info_log += f"  Channels: {psd.channels}\n"
            info_log += f"  Depth: {psd.depth} bits\n"

            # Count layers
            layer_count = len(list(psd))
            info_log += f"  Layer count: {layer_count}\n"

            # List layer names with detailed info
            if layer_count > 0:
                info_log += "\nLayers (detailed):\n"
                for idx, layer in enumerate(psd, start=1):
                    layer_name = layer.name if hasattr(layer, 'name') else f"Layer {idx}"
                    layer_visible = "visible" if (hasattr(layer, 'visible') and layer.visible) else "hidden"
                    layer_opacity = layer.opacity if hasattr(layer, 'opacity') else "N/A"
                    layer_blend_mode = layer.blend_mode if hasattr(layer, 'blend_mode') else "N/A"

                    info_log += f"  {idx}. {layer_name}\n"
                    info_log += f"      Visible: {layer_visible}\n"
                    info_log += f"      Opacity: {layer_opacity}\n"
                    info_log += f"      Blend mode: {layer_blend_mode}\n"

                    # Check if layer has actual pixel data
                    if hasattr(layer, 'bbox'):
                        bbox = layer.bbox
                        info_log += f"      Bounding box: {bbox}\n"

                    if hasattr(layer, 'has_pixels') and callable(layer.has_pixels):
                        has_pixels = layer.has_pixels()
                        info_log += f"      Has pixels: {has_pixels}\n"

                    # Check layer transparency by converting to PIL
                    try:
                        if hasattr(layer, 'topil') and layer.kind == 'pixel':
                            layer_pil = layer.topil()
                            if layer_pil:
                                info_log += f"      Layer mode: {layer_pil.mode}\n"
                                if layer_pil.mode == 'RGBA':
                                    info_log += f"      ✓ Has alpha channel (transparency)\n"
                                else:
                                    info_log += f"      ✗ No alpha channel\n"
                    except Exception as e:
                        info_log += f"      Could not check transparency: {str(e)}\n"

            # Render composite
            info_log += "\nRendering flattened composite...\n"
            info_log += "  Note: composite() flattens all layers and may not preserve\n"
            info_log += "  per-layer transparency (known psd-tools limitation)\n"

            # Try to get RGBA composite with transparent backdrop
            pil_image = psd.composite(color=1.0, alpha=0.0)

            info_log += "[OK] Rendered PSD composite\n"
            info_log += f"  Output size: {pil_image.width}x{pil_image.height}\n"
            info_log += f"  Output mode: {pil_image.mode}\n"
            if pil_image.mode == 'RGBA':
                info_log += "  ✓ Composite has alpha channel\n"
            else:
                info_log += "  ✗ Composite does not have alpha channel\n"

            # Convert PIL Image to tensor, handling alpha channel properly
            if pil_image.mode == 'RGBA':
                info_log += f"  Compositing RGBA onto white background\n"
                # Create a white background image
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                # Composite the RGBA image onto the white background
                background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha as mask
                pil_image = background
            elif pil_image.mode not in ['RGB', 'RGBA']:
                info_log += f"  Converting {pil_image.mode} to RGB\n"
                pil_image = pil_image.convert('RGB')

            img_array = np.array(pil_image).astype(np.float32) / 255.0

            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)

            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            info_log += f"\n{'='*55}\n"
            info_log += "Preview rendered successfully\n"
            info_log += f"{'='*55}\n"

            return (img_tensor, info_log)

        except Exception as e:
            info_log += f"\n{'='*55}\n"
            info_log += f"ERROR: {str(e)}\n"
            info_log += f"{'='*55}\n"
            print(info_log)
            raise

        finally:
            # Clean up temporary file if downloaded
            if is_temp_file and tmp_psd_path:
                try:
                    os.unlink(tmp_psd_path)
                except:
                    pass


# Node registration
NODE_CLASS_MAPPINGS = {
    "PhotoshopPreviewPSDNode": PhotoshopPreviewPSDNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoshopPreviewPSDNode": "Preview PSD",
}
