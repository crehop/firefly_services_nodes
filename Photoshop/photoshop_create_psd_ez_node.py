"""
Adobe Photoshop Create PSD EZ Layers Node

Create multi-layer PSD files with simple layer stacking using Adobe Photoshop API.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import aiohttp
import numpy as np
import io
import time
import json
import tempfile
import os
from urllib.parse import urlparse
from PIL import Image

# Import psd-tools for PSD rendering
try:
    from psd_tools import PSDImage
    PSD_TOOLS_AVAILABLE = True
except ImportError:
    PSD_TOOLS_AVAILABLE = False

# Import utility functions
from ..apinode_utils import download_url_to_bytesio

from .photoshop_api import (
    PhotoshopJobStatusEnum,
    PhotoshopActionsInput,
    DocumentSettings,
    DocumentLayerInput,
    DocumentCreateOptions,
    DocumentCreateOutput,
    DocumentCreateRequest,
    DocumentCreateResponse,
    DocumentCreateJobStatus,
)
from .photoshop_storage import upload_image_to_s3, generate_output_presigned_url, generate_download_url
from ..Firefly.firefly_storage import create_adobe_client
from ..client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)


class PhotoshopCreatePSDEZNode:
    """
    Create multi-layer PSD files with simple layer stacking.

    Features:
    - Up to 10 image layers
    - Auto-generated layer names (Layer1, Layer2, etc.)
    - Layer 1 = bottom layer, higher numbers = upper layers
    - Auto-detect document dimensions from largest image
    - Manual document dimension override
    - Output as JPEG, PNG, or PSD
    - PSD outputs rendered as flattened preview
    - Comprehensive debug logging
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "output_url", "debug_log")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/photoshop"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                # Image layers (1-10)
                "layer_1_image": ("IMAGE", {"tooltip": "Layer 1 (bottom layer)"}),
                "layer_1_mask": ("MASK", {"tooltip": "Optional mask for layer 1 transparency"}),
                "layer_2_image": ("IMAGE", {"tooltip": "Layer 2"}),
                "layer_2_mask": ("MASK", {"tooltip": "Optional mask for layer 2 transparency"}),
                "layer_3_image": ("IMAGE", {"tooltip": "Layer 3"}),
                "layer_3_mask": ("MASK", {"tooltip": "Optional mask for layer 3 transparency"}),
                "layer_4_image": ("IMAGE", {"tooltip": "Layer 4"}),
                "layer_4_mask": ("MASK", {"tooltip": "Optional mask for layer 4 transparency"}),
                "layer_5_image": ("IMAGE", {"tooltip": "Layer 5"}),
                "layer_5_mask": ("MASK", {"tooltip": "Optional mask for layer 5 transparency"}),
                "layer_6_image": ("IMAGE", {"tooltip": "Layer 6"}),
                "layer_6_mask": ("MASK", {"tooltip": "Optional mask for layer 6 transparency"}),
                "layer_7_image": ("IMAGE", {"tooltip": "Layer 7"}),
                "layer_7_mask": ("MASK", {"tooltip": "Optional mask for layer 7 transparency"}),
                "layer_8_image": ("IMAGE", {"tooltip": "Layer 8"}),
                "layer_8_mask": ("MASK", {"tooltip": "Optional mask for layer 8 transparency"}),
                "layer_9_image": ("IMAGE", {"tooltip": "Layer 9"}),
                "layer_9_mask": ("MASK", {"tooltip": "Optional mask for layer 9 transparency"}),
                "layer_10_image": ("IMAGE", {"tooltip": "Layer 10 (top layer)"}),
                "layer_10_mask": ("MASK", {"tooltip": "Optional mask for layer 10 transparency"}),
            },
            # "hidden": {
            #     "unique_id": "UNIQUE_ID",  # Disabled - no task tracking
            # },
        }

    def _get_image_dimensions(self, img_tensor: torch.Tensor) -> Tuple[int, int]:
        """Get image dimensions (width, height) from tensor."""
        h, w, c = img_tensor.shape
        return (w, h)

    def _auto_detect_document_size(self, images: list) -> Tuple[int, int]:
        """Auto-detect document size from largest image dimensions."""
        max_width = 0
        max_height = 0

        for img in images:
            if img is not None:
                w, h = self._get_image_dimensions(img[0])
                max_width = max(max_width, w)
                max_height = max(max_height, h)

        return (max_width, max_height)

    def _join_image_with_alpha(self, img_tensor: torch.Tensor, mask_tensor: Optional[torch.Tensor]) -> Tuple[torch.Tensor, str]:
        """Join RGB image with mask to create RGBA.

        Args:
            img_tensor: RGB image tensor [H, W, 3]
            mask_tensor: Optional mask tensor [H, W] or [1, H, W]

        Returns:
            Tuple of (RGBA tensor, debug message)
        """
        # Check if already RGBA (4 channels)
        if img_tensor.shape[-1] == 4:
            return img_tensor, "Already RGBA"

        # If no mask provided, return original
        if mask_tensor is None:
            return img_tensor, "No mask provided"

        # Only process RGB (3 channels)
        if img_tensor.shape[-1] != 3:
            return img_tensor, f"Unsupported channel count: {img_tensor.shape[-1]}"

        # Get mask and ensure it's 2D [H, W]
        if len(mask_tensor.shape) == 3:
            # Mask is [1, H, W], squeeze to [H, W]
            alpha = mask_tensor.squeeze(0)
        else:
            alpha = mask_tensor

        # Ensure mask dimensions match image
        if alpha.shape != img_tensor.shape[:2]:
            return img_tensor, f"Mask size mismatch: {alpha.shape} vs {img_tensor.shape[:2]}"

        # Invert mask: ComfyUI masks are inverted for alpha channel
        # In mask: white = masked area, black = unmasked
        # In alpha: white = opaque, black = transparent
        # So we need to invert: alpha = 1.0 - mask
        alpha = 1.0 - alpha

        # Join image with mask to create RGBA
        # Stack the RGB channels with the inverted mask as alpha channel
        rgba_tensor = torch.cat([img_tensor, alpha.unsqueeze(-1)], dim=-1)

        return rgba_tensor, "Joined with inverted mask"

    def _build_debug_log(
        self,
        layer_count: int,
        doc_width: int,
        doc_height: int,
        doc_resolution: int,
        doc_fill: str,
        doc_mode: str,
        output_type: str,
        output_quality: int,
        compression: str,
    ) -> str:
        """Build formatted debug log showing request details."""
        log = "=" * 55 + "\n"
        log += "POST /pie/psdService/documentCreate\n"
        log += "-" * 55 + "\n"
        log += "Headers:\n"
        log += "  Authorization: Bearer [TOKEN]\n"
        log += "  x-api-key: [API_KEY]\n"
        log += "  Content-Type: application/json\n"
        log += "\nRequest Body:\n"

        # Options
        log += "  options:\n"
        log += "    document:\n"
        log += f"      width: {doc_width}\n"
        log += f"      height: {doc_height}\n"
        log += f"      resolution: {doc_resolution}\n"
        log += f"      fill: {doc_fill}\n"
        log += f"      mode: {doc_mode}\n"
        log += f"      depth: 8\n"
        log += "    layers:\n"
        for i in range(layer_count):
            log += f"      - name: Layer{i+1}\n"
            log += f"        type: layer\n"
            log += f"        input:\n"
            log += f"          href: [S3_PRESIGNED_URL_{i+1}]\n"
            log += f"          storage: external\n"
            log += f"        bounds:\n"
            log += f"          top: 0\n"
            log += f"          left: 0\n"
            log += f"          width: {doc_width}\n"
            log += f"          height: {doc_height}\n"

        # Output
        log += "  outputs:\n"
        log += "    - href: [S3_PRESIGNED_URL]\n"
        log += "      storage: external\n"
        log += f"      type: {output_type}\n"
        log += "      overwrite: true\n"
        if output_type == "image/jpeg":
            log += f"      quality: {output_quality}\n"
        elif output_type == "image/png":
            log += f"      compression: {compression}\n"

        return log

    async def api_call(
        self,
        layer_1_image: Optional[torch.Tensor] = None,
        layer_1_mask: Optional[torch.Tensor] = None,
        layer_2_image: Optional[torch.Tensor] = None,
        layer_2_mask: Optional[torch.Tensor] = None,
        layer_3_image: Optional[torch.Tensor] = None,
        layer_3_mask: Optional[torch.Tensor] = None,
        layer_4_image: Optional[torch.Tensor] = None,
        layer_4_mask: Optional[torch.Tensor] = None,
        layer_5_image: Optional[torch.Tensor] = None,
        layer_5_mask: Optional[torch.Tensor] = None,
        layer_6_image: Optional[torch.Tensor] = None,
        layer_6_mask: Optional[torch.Tensor] = None,
        layer_7_image: Optional[torch.Tensor] = None,
        layer_7_mask: Optional[torch.Tensor] = None,
        layer_8_image: Optional[torch.Tensor] = None,
        layer_8_mask: Optional[torch.Tensor] = None,
        layer_9_image: Optional[torch.Tensor] = None,
        layer_9_mask: Optional[torch.Tensor] = None,
        layer_10_image: Optional[torch.Tensor] = None,
        layer_10_mask: Optional[torch.Tensor] = None,
        # unique_id: Optional[str] = None,  # Disabled - no task tracking
    ):
        """Create multi-layer PSD file using Photoshop API."""

        # Hardcoded settings
        doc_resolution = 72
        doc_fill = "transparent"
        doc_mode = "rgb"
        depth = 8
        output_type = "image/vnd.adobe.photoshop"

        # Check if psd-tools is available
        if not PSD_TOOLS_AVAILABLE:
            raise ImportError(
                "psd-tools library is required but is not installed.\n"
                "Please install it with: pip install psd-tools"
            )

        # Collect all provided images and masks as tuples
        layers_data = [
            (layer_1_image, layer_1_mask),
            (layer_2_image, layer_2_mask),
            (layer_3_image, layer_3_mask),
            (layer_4_image, layer_4_mask),
            (layer_5_image, layer_5_mask),
            (layer_6_image, layer_6_mask),
            (layer_7_image, layer_7_mask),
            (layer_8_image, layer_8_mask),
            (layer_9_image, layer_9_mask),
            (layer_10_image, layer_10_mask),
        ]

        # Filter out None images (keep mask even if None)
        valid_layers = [(img, mask) for img, mask in layers_data if img is not None]

        # Validate at least one image
        if len(valid_layers) == 0:
            raise ValueError("Must provide at least one image layer (layer_1_image through layer_10_image)")

        # Extract just images for document size detection
        valid_images = [img for img, _ in valid_layers]

        # Auto-detect document size from largest image
        doc_width, doc_height = self._auto_detect_document_size(valid_images)

        # Validate document size
        if doc_width == 0 or doc_height == 0:
            raise ValueError(f"Could not determine document dimensions from images.")

        # Build initial debug log
        console_log = "=" * 55 + "\n"
        console_log += "POST /pie/psdService/documentCreate\n"
        console_log += "-" * 55 + "\n"
        console_log += f"Layers: {len(valid_images)}\n"
        console_log += f"Document: {doc_width}x{doc_height}, {doc_resolution}dpi, {doc_mode}, {doc_fill}\n"
        console_log += f"Output: PSD\n"

        client = await create_adobe_client()

        try:
            # Upload all images to S3
            console_log += f"\n{'='*55}\n"
            console_log += f"Uploading {len(valid_images)} layer(s) to S3...\n"

            layer_urls = []
            for idx, (img, mask) in enumerate(valid_layers, start=1):
                console_log += f"\nLayer {idx}:\n"

                # Get image tensor
                img_tensor = img[0]
                h, w, c = img_tensor.shape
                console_log += f"  Original: {w}x{h} ({c} channels)\n"

                # Use provided mask (e.g., from Load Image's mask output)
                mask_to_use = None

                if mask is not None:
                    # Mask provided - extract from Load Image's mask output
                    mask_to_use = mask[0] if len(mask.shape) == 4 else mask
                    console_log += f"  Mask provided: {mask_to_use.shape}\n"
                else:
                    console_log += f"  No mask provided\n"

                # Join image with mask to create RGBA
                img_tensor, join_msg = self._join_image_with_alpha(img_tensor, mask_to_use)
                h, w, c = img_tensor.shape
                console_log += f"  After join: {w}x{h} ({c} channels) - {join_msg}\n"

                if c == 4:
                    console_log += f"  ✓ Has alpha channel for transparency\n"
                else:
                    console_log += f"  ✗ No alpha channel (solid image)\n"

                upload_start = time.time()
                layer_url = await upload_image_to_s3(img_tensor)
                upload_duration = time.time() - upload_start

                console_log += f"  [OK] Upload complete ({upload_duration:.2f}s)\n"
                console_log += f"  URL: {layer_url[:80]}...\n"

                layer_urls.append(layer_url)

            # Create output URL (S3 pre-signed PUT URL)
            console_log += f"\n{'='*55}\n"
            console_log += "Generating output pre-signed PUT URL for S3...\n"

            # Generate pre-signed PUT URL for Adobe to upload the result
            url_gen_start = time.time()
            output_url_presigned, output_filename = await generate_output_presigned_url(file_extension="psd")
            url_gen_duration = time.time() - url_gen_start

            console_log += f"[OK] Generated pre-signed PUT URL ({url_gen_duration:.2f}s)\n"
            console_log += f"  URL: {output_url_presigned[:80]}...\n"
            console_log += f"  Adobe will upload result to this URL\n"
            console_log += f"  Output filename: {output_filename}\n"

            # Build layers array
            # NOTE: Layers are added in reverse order (top to bottom) for Photoshop API
            # So layer_1_image (bottom) is added last, layer_10_image (top) is added first
            layers = []
            for idx, layer_url in enumerate(reversed(layer_urls), start=1):
                # Calculate the actual layer number (reversed)
                actual_layer_num = len(layer_urls) - idx + 1

                # Build layer with name "Layer1", "Layer2", etc.
                layers.append(
                    DocumentLayerInput(
                        type="layer",
                        input=PhotoshopActionsInput(href=layer_url),
                        name=f"Layer{actual_layer_num}",
                    )
                )

            # Build request matching the exact schema
            request = DocumentCreateRequest(
                options=DocumentCreateOptions(
                    document=DocumentSettings(
                        width=doc_width,
                        height=doc_height,
                        resolution=doc_resolution,
                        mode=doc_mode,
                        fill=doc_fill,
                        depth=depth,
                    ),
                    layers=layers,
                ),
                outputs=[
                    DocumentCreateOutput(
                        href=output_url_presigned,
                        storage="external",
                        type=output_type,
                    )
                ],
            )

            # Log the actual request JSON being sent
            console_log += f"\nRequest JSON being sent to API:\n"
            console_log += f"{json.dumps(request.model_dump(exclude_unset=True, exclude_none=True), indent=2)}\n"

            # Submit job
            submit_endpoint = ApiEndpoint(
                path="/pie/psdService/documentCreate",
                method=HttpMethod.POST,
                request_model=DocumentCreateRequest,
                response_model=DocumentCreateResponse,
            )

            submit_op = SynchronousOperation(
                endpoint=submit_endpoint,
                request=request,
                api_base="https://image.adobe.io",
            )

            # Update client's base_url
            client.base_url = "https://image.adobe.io"
            submit_response = await submit_op.execute(client=client)

            # Log submit response
            console_log += f"\nResponse: 202 Accepted\n"

            # Extract job info from response
            job_id = submit_response.jobId
            status_url = submit_response.statusUrl

            if not job_id or not status_url:
                console_log += f"ERROR: Failed to extract jobId or statusUrl from response\n"
                console_log += f"Response _links: {submit_response.links}\n"
                raise Exception("Failed to extract job information from API response")

            console_log += f"  jobId: {job_id}\n"
            console_log += f"  statusUrl: {status_url}\n"

            # Log raw response for debugging
            console_log += f"\nRaw Submit Response:\n"
            console_log += f"{json.dumps(submit_response.model_dump(), indent=2)}\n"

            # Parse statusUrl to get correct polling endpoint
            parsed_status_url = urlparse(status_url)
            status_base_url = f"{parsed_status_url.scheme}://{parsed_status_url.netloc}"
            status_path = parsed_status_url.path

            # Log polling start
            console_log += f"\n{'='*55}\n"
            console_log += f"GET {status_path}\n"
            console_log += f"{'-'*55}\n"
            console_log += "Polling for job completion...\n"
            console_log += f"  Status URL: {status_url}\n"
            console_log += f"  Interval: {2.0}s\n"
            console_log += f"  Max attempts: {15} (timeout: {15 * 2.0}s = 30s)\n"

            # Poll for completion
            poll_endpoint = ApiEndpoint(
                path=status_path,
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=DocumentCreateJobStatus,
            )

            poll_op = PollingOperation(
                poll_endpoint=poll_endpoint,
                request=EmptyRequest(),
                completed_statuses=["succeeded"],
                failed_statuses=["failed"],
                status_extractor=lambda x: x.status,
                api_base=status_base_url,
                poll_interval=2.0,
                max_poll_attempts=15,  # 30s timeout
                estimated_duration=10.0,  # Estimated 10s for Document Create
                # node_id=unique_id,  # Disabled - no progress bar
            )

            # Track polling time
            poll_start_time = time.time()
            console_log += f"\nStarting polling at {time.strftime('%H:%M:%S', time.localtime(poll_start_time))}...\n"

            result = await poll_op.execute(client=client)

            # Calculate polling duration
            poll_end_time = time.time()
            poll_duration = poll_end_time - poll_start_time
            estimated_attempts = int(poll_duration / 2.0) + 1

            # Log result with detailed timing
            console_log += f"\nPolling completed at {time.strftime('%H:%M:%S', time.localtime(poll_end_time))}\n"
            console_log += f"  Duration: {poll_duration:.1f}s\n"
            console_log += f"  Estimated attempts: ~{estimated_attempts}\n"
            console_log += f"\nResponse: 200 OK\n"
            console_log += f"  status: {result.status}\n"
            if result.jobId:
                console_log += f"  jobId: {result.jobId}\n"
            try:
                output_count = len(result.result.outputs) if result.result and result.result.outputs else 0
                console_log += f"  outputs: {output_count} file(s)\n"
            except Exception as e:
                console_log += f"  outputs: ERROR accessing outputs - {e}\n"

            # Log raw polling response for debugging
            console_log += f"\nRaw Poll Response:\n"
            try:
                console_log += f"{json.dumps(result.model_dump(), indent=2)}\n"
            except Exception as e:
                console_log += f"ERROR dumping result: {e}\n"

            # Validate outputs
            if not result.result or not result.result.outputs:
                console_log += f"\n{'='*55}\n"
                console_log += "ERROR: No outputs in response\n"
                console_log += f"  status: {result.status}\n"
                console_log += f"\n**CONNECT debug_log OUTPUT TO ShowText TO SEE FULL RESPONSE**\n"
                print(console_log)
                raise Exception(f"No outputs returned from Photoshop Document Create API. Check console logs or connect debug_log output for details.")

            # Generate GET URL for downloading the result
            console_log += f"\n{'='*55}\n"
            console_log += "Generating download URL...\n"

            download_url = await generate_download_url(output_filename)
            console_log += f"[OK] Generated download URL\n"
            console_log += f"  URL: {download_url[:80]}...\n"

            # Download and process result based on output type
            if output_type == "image/vnd.adobe.photoshop":
                # Download PSD and render using psd-tools
                console_log += "\nDownloading PSD file...\n"

                # Use utility function to download
                psd_bytesio = await download_url_to_bytesio(download_url)
                psd_bytes = psd_bytesio.getvalue()

                console_log += "[OK] Downloaded PSD file\n"
                console_log += f"  Size: {len(psd_bytes)} bytes\n"

                # Save PSD to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".psd") as tmp_file:
                    tmp_file.write(psd_bytes)
                    tmp_psd_path = tmp_file.name

                try:
                    # Render PSD to flattened image using psd-tools
                    console_log += "\nRendering PSD to flattened preview image...\n"

                    psd = PSDImage.open(tmp_psd_path)

                    console_log += "[OK] Opened PSD file\n"
                    console_log += f"  PSD size: {psd.width}x{psd.height}\n"
                    layer_list = list(psd)
                    console_log += f"  Layer count: {len(layer_list)}\n"

                    # Check each layer for transparency
                    console_log += "\nChecking layer transparency:\n"
                    for idx, layer in enumerate(layer_list, start=1):
                        layer_name = layer.name if hasattr(layer, 'name') else f"Layer {idx}"
                        console_log += f"  {idx}. {layer_name}\n"
                        try:
                            if hasattr(layer, 'topil') and layer.kind == 'pixel':
                                layer_pil = layer.topil()
                                if layer_pil:
                                    console_log += f"      Mode: {layer_pil.mode}\n"
                                    if layer_pil.mode == 'RGBA':
                                        console_log += f"      ✓ Has alpha channel\n"
                                    else:
                                        console_log += f"      ✗ No alpha channel\n"
                        except Exception as e:
                            console_log += f"      Error checking: {str(e)}\n"

                    # Composite with transparent backdrop
                    console_log += "\nRendering flattened composite...\n"
                    pil_image = psd.composite(color=1.0, alpha=0.0)
                    console_log += f"[OK] Composite mode: {pil_image.mode}\n"

                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_psd_path)
                    except:
                        pass

                # Convert PIL Image to tensor, handling alpha channel properly
                if pil_image.mode == 'RGBA':
                    console_log += "  Compositing RGBA preview onto white background\n"
                    # Create a white background image
                    background = Image.new('RGB', pil_image.size, (255, 255, 255))
                    # Composite the RGBA image onto the white background
                    background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha as mask
                    pil_image = background
                elif pil_image.mode not in ['RGB', 'RGBA']:
                    pil_image = pil_image.convert('RGB')

                img_array = np.array(pil_image).astype(np.float32) / 255.0

                if len(img_array.shape) == 2:  # Grayscale
                    img_array = np.stack([img_array] * 3, axis=-1)

                img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            else:
                # Download JPEG/PNG directly
                console_log += "\nDownloading output image...\n"

                # Use utility function to download
                image_bytesio = await download_url_to_bytesio(download_url)

                console_log += "[OK] Downloaded image\n"

                # Convert to tensor
                img_pil = Image.open(image_bytesio)

                if img_pil.mode not in ['RGB', 'RGBA']:
                    img_pil = img_pil.convert('RGB')

                img_array = np.array(img_pil).astype(np.float32) / 255.0

                if len(img_array.shape) == 2:  # Grayscale
                    img_array = np.stack([img_array] * 3, axis=-1)

                img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            console_log += f"\nOutput URL (valid for 24 hours):\n"
            console_log += f"  {download_url}\n"
            console_log += f"{'='*55}\n"

            return (img_tensor, download_url, console_log)

        except Exception as e:
            console_log += f"\n{'='*55}\n"
            console_log += f"ERROR: {str(e)}\n"
            console_log += f"{'='*55}\n"
            print(console_log)

            try:
                await client.close()
            except:
                pass

            raise
        finally:
            try:
                await client.close()
            except:
                pass


# Node registration
NODE_CLASS_MAPPINGS = {
    "PhotoshopCreatePSDEZNode": PhotoshopCreatePSDEZNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoshopCreatePSDEZNode": "Create PSD EZ Layers",
}
