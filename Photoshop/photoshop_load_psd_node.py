"""
Adobe Photoshop Load PSD Node

Load PSD files, show preview, and output PSD as pre-signed URL.
"""

from __future__ import annotations
from typing import Optional
import torch
import numpy as np
import tempfile
import os
import uuid
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
from .photoshop_storage import generate_download_url

# Import ComfyUI folder utilities
import folder_paths


class PhotoshopLoadPSDNode:
    """
    Load PSD files, show preview, and output PSD URL.

    Features:
    - Select PSD from input folder via file picker
    - Render flattened composite preview using psd-tools
    - Output the actual PSD as a pre-signed URL (for passing to other nodes)
    - MIT licensed (commercial-friendly)
    """

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "psd_url")
    FUNCTION = "load_psd"
    CATEGORY = "api node/photoshop"

    @classmethod
    def INPUT_TYPES(cls):
        # Simple approach: just list PSD files from input directory (like LayerUtility LoadPSD)
        input_dir = folder_paths.get_input_directory()
        all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        # Filter to only PSD files
        psd_files = [f for f in all_files if f.lower().endswith('.psd')]

        return {
            "required": {
                "psd_file": (sorted(psd_files), {
                    "tooltip": "Select a PSD file from the ComfyUI/input folder. Place your .psd files in the input directory to see them here."
                }),
            },
            "optional": {
                "file_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: Full path to a PSD file. Overrides psd_file if provided."
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, psd_file):
        print(f"[Load PSD] IS_CHANGED called with psd_file: {psd_file}")
        psd_path = folder_paths.get_annotated_filepath(psd_file)
        print(f"[Load PSD] Annotated filepath: {psd_path}")
        m = hashlib.sha256()
        with open(psd_path, 'rb') as f:
            m.update(f.read())
        hash_value = m.digest().hex()
        print(f"[Load PSD] File hash: {hash_value}")
        return hash_value

    @classmethod
    def VALIDATE_INPUTS(cls, psd_file):
        print(f"[Load PSD] VALIDATE_INPUTS called with psd_file: '{psd_file}'")

        # Check for undefined or empty value
        if not psd_file or psd_file == "undefined" or psd_file.strip() == "":
            print(f"[Load PSD] Validation failed: No file selected")
            return "No PSD file selected. Please select a PSD file from the dropdown or upload one."

        if not folder_paths.exists_annotated_filepath(psd_file):
            print(f"[Load PSD] Validation failed: File does not exist")
            return "Invalid PSD file: {}".format(psd_file)

        if not psd_file.lower().endswith('.psd'):
            print(f"[Load PSD] Validation failed: Not a .psd file")
            return "File must be a .psd file"

        print(f"[Load PSD] Validation passed")
        return True

    async def load_psd(
        self,
        psd_file: str,
        file_path: Optional[str] = "",
    ):
        """Load PSD file from input folder or custom path, render preview, and output PSD URL."""

        print(f"\n[Load PSD] ====== LOAD_PSD FUNCTION CALLED ======")
        print(f"[Load PSD] psd_file parameter: '{psd_file}'")
        print(f"[Load PSD] file_path parameter: '{file_path}'")

        # Check if psd-tools is available
        if not PSD_TOOLS_AVAILABLE:
            print(f"[Load PSD] ERROR: psd-tools not available")
            raise ImportError(
                "psd-tools library is required but is not installed.\n"
                "Please install it with: pip install psd-tools"
            )

        print(f"[Load PSD] psd-tools is available")

        try:
            # Determine which path to use
            if file_path and file_path.strip():
                print(f"[Load PSD] Using custom file_path")
                psd_path = file_path.strip()
            else:
                print(f"[Load PSD] Using psd_file from input folder")
                print(f"[Load PSD] Getting annotated filepath...")
                psd_path = folder_paths.get_annotated_filepath(psd_file)

            print(f"[Load PSD] Full PSD path: {psd_path}")

            # Upload local PSD to S3 and generate pre-signed URL
            print(f"[Load PSD] Reading PSD file...")
            with open(psd_path, 'rb') as f:
                psd_bytes = f.read()
            print(f"[Load PSD] PSD file size: {len(psd_bytes)} bytes")

            # Upload to S3 using the storage helper
            print(f"[Load PSD] Importing storage modules...")
            from comfy_api_nodes.apis.photoshop_storage import _upload_to_s3_sync, _load_firefly_config
            import asyncio

            print(f"[Load PSD] Loading Firefly config...")
            config = _load_firefly_config()
            aws_config = {
                'aws_access_key_id': config.get("aws_access_key_id"),
                'aws_secret_access_key': config.get("aws_secret_access_key"),
                'aws_region': config.get("aws_region", "us-east-1"),
                'aws_bucket': config.get("aws_bucket"),
            }

            # Generate unique filename for PSD
            filename = f"psd-upload-{uuid.uuid4()}.psd"
            print(f"[Load PSD] Generated filename: {filename}")

            # Upload to S3
            print(f"[Load PSD] Uploading to S3...")
            loop = asyncio.get_running_loop()
            output_psd_url = await loop.run_in_executor(
                None,
                _upload_to_s3_sync,
                psd_bytes,
                aws_config,
                filename
            )
            print(f"[Load PSD] S3 upload complete: {output_psd_url[:80]}...")

            # Open and render PSD preview
            print(f"[Load PSD] Opening PSD with psd-tools...")
            psd = PSDImage.open(psd_path)
            print(f"[Load PSD] PSD opened successfully: {psd.width}x{psd.height}")

            # Composite with transparent backdrop to try preserving alpha
            print(f"[Load PSD] Rendering composite...")
            pil_image = psd.composite(color=1.0, alpha=0.0)
            print(f"[Load PSD] Composite rendered: {pil_image.width}x{pil_image.height}, mode: {pil_image.mode}")

            # Convert PIL Image to tensor, handling alpha channel properly
            if pil_image.mode == 'RGBA':
                print(f"[Load PSD] Converting RGBA to RGB with white background")
                # Create a white background image
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                # Composite the RGBA image onto the white background
                background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha as mask
                pil_image = background
            elif pil_image.mode not in ['RGB', 'RGBA']:
                print(f"[Load PSD] Converting {pil_image.mode} to RGB")
                pil_image = pil_image.convert('RGB')

            print(f"[Load PSD] Converting PIL to numpy array...")
            img_array = np.array(pil_image).astype(np.float32) / 255.0

            if len(img_array.shape) == 2:  # Grayscale
                print(f"[Load PSD] Converting grayscale to RGB")
                img_array = np.stack([img_array] * 3, axis=-1)

            print(f"[Load PSD] Converting numpy to tensor...")
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            print(f"[Load PSD] Final tensor shape: {img_tensor.shape}")

            print(f"[Load PSD] ====== LOAD_PSD COMPLETE ======\n")
            return (img_tensor, output_psd_url)

        except Exception as e:
            print(f"[Load PSD] ====== ERROR IN LOAD_PSD ======")
            print(f"[Load PSD] Exception: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"[Load PSD] ===================================\n")
            raise Exception(f"Failed to load PSD: {str(e)}")


# Node registration
NODE_CLASS_MAPPINGS = {
    "PhotoshopLoadPSDNode": PhotoshopLoadPSDNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoshopLoadPSDNode": "Load PSD",
}
