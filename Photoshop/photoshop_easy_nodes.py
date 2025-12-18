"""
Adobe Photoshop API Nodes

Implements Photoshop API nodes using the same pattern as Firefly nodes.
"""

from __future__ import annotations
from typing import Optional
import torch
import aiohttp
import numpy as np
import io
import time
import json
from urllib.parse import urlparse
from PIL import Image

from .photoshop_api import (
    PhotoshopRemoveBgMode,
    PhotoshopOutputMediaType,
    PhotoshopJobStatusEnum,
    PhotoshopStorageType,
    PhotoshopImageSource,
    PhotoshopImageInput,
    PhotoshopBackgroundColor,
    PhotoshopOutputOptions,
    RemoveBackgroundRequest,
    RemoveBackgroundResponse,
    RefineMaskRequest,
    PhotoshopJobStatus,
    RefineMaskStatusResponse,
    MaskObjectsRequest,
    MaskObjectsStatusResponse,
    FillMaskedAreasRequest,
    FillMaskedAreasStatusResponse,
)
from .photoshop_storage import upload_image_to_s3
from ..Firefly.firefly_storage import create_adobe_client
from ..client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)
# from comfy.utils import ProgressBar  # Disabled - no task tracking


class PhotoshopRemoveBackgroundNode:
    """
    Remove background from images using Adobe Photoshop API.

    Features:
    - Dual input support (tensor or pre-signed URL)
    - Full parameter control
    - Comprehensive debug logging
    - Async polling for results
    """

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "output_url", "debug_log")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/photoshop"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["mask", "cutout", "psd"], {
                    "default": "mask",
                    "tooltip": "Output mode: cutout (transparent BG), mask (B&W), or psd (layered)",
                }),
            },
            "optional": {
                # Dual input: tensor OR reference URL (mutually exclusive)
                "image": ("IMAGE", {
                    "tooltip": "Input image tensor to process",
                }),
                "image_reference": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Pre-signed URL to input image (alternative to tensor)",
                }),

                # Remove background parameters
                "output_media_type": (
                    ["image/png", "image/jpeg", "image/webp", "image/vnd.adobe.photoshop"],
                    {
                        "default": "image/png",
                        "tooltip": "Output image format",
                    }
                ),
                "trim": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Crop to cutout border (removes transparent pixels)",
                }),

                # Background color (only used if not trim)
                "background_color_red": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "tooltip": "Background color red component (0-255)",
                }),
                "background_color_green": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "tooltip": "Background color green component (0-255)",
                }),
                "background_color_blue": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "tooltip": "Background color blue component (0-255)",
                }),
                "background_color_alpha": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Background color alpha (transparency, 0=transparent, 1=opaque)",
                }),

                # Color decontamination
                "color_decontamination": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Remove colored reflections from background (0=none, 1=full)",
                }),
            },
            # "hidden": {
            #     "unique_id": "UNIQUE_ID",  # Disabled - no task tracking
            # },
        }

    def _build_debug_log(
        self,
        mode: str,
        output_media_type: str,
        trim: bool,
        bg_red: int,
        bg_green: int,
        bg_blue: int,
        bg_alpha: float,
        color_decontamination: float,
        image: Optional[torch.Tensor],
        image_reference: str,
    ) -> str:
        """Build formatted debug log showing request details."""
        log = "=" * 55 + "\n"
        log += "POST /v2/remove-background\n"
        log += "-" * 55 + "\n"
        log += "Headers:\n"
        log += "  Authorization: Bearer [TOKEN]\n"
        log += "  x-api-key: [API_KEY]\n"
        log += "  Content-Type: application/json\n"
        log += "\nRequest Body:\n"

        # Image source
        if image is not None:
            log += "  image:\n"
            log += "    source:\n"
            log += "      url: [S3_PRESIGNED_URL]\n"
        else:
            log += "  image:\n"
            log += "    source:\n"
            log += f"      url: {image_reference}\n"

        log += f"  mode: {mode}\n"

        log += "  output:\n"
        log += f"    mediaType: {output_media_type}\n"
        log += f"    trim: {trim}\n"

        if not trim:
            log += "    backgroundColor:\n"
            log += f"      red: {bg_red}\n"
            log += f"      green: {bg_green}\n"
            log += f"      blue: {bg_blue}\n"
            log += f"      alpha: {bg_alpha}\n"

        log += f"    colorDecontamination: {color_decontamination}\n"

        return log

    async def api_call(
        self,
        mode: str,
        image: Optional[torch.Tensor] = None,
        image_reference: str = "",
        output_media_type: str = "image/png",
        trim: bool = False,
        background_color_red: int = 255,
        background_color_green: int = 255,
        background_color_blue: int = 255,
        background_color_alpha: float = 0.0,
        color_decontamination: float = 0.0,
        # unique_id: Optional[str] = None,  # Disabled - no task tracking
    ):
        """Remove background from image using Photoshop API."""

        # Validate inputs
        if image is None and not image_reference:
            raise ValueError("Must provide either 'image' or 'image_reference'")
        if image is not None and image_reference:
            raise ValueError("Cannot provide both 'image' and 'image_reference' - choose only one")

        # Build initial debug log
        console_log = self._build_debug_log(
            mode=mode,
            output_media_type=output_media_type,
            trim=trim,
            bg_red=background_color_red,
            bg_green=background_color_green,
            bg_blue=background_color_blue,
            bg_alpha=background_color_alpha,
            color_decontamination=color_decontamination,
            image=image,
            image_reference=image_reference,
        )

        client = await create_adobe_client()

        try:
            # Get input URL
            if image is not None:
                console_log += f"\n{'='*55}\n"
                console_log += "Uploading image to S3...\n"

                # Get image info
                img_tensor = image[0]
                h, w, c = img_tensor.shape
                console_log += f"  Image size: {w}x{h} ({c} channels)\n"

                # Upload and measure time
                upload_start = time.time()
                input_url = await upload_image_to_s3(img_tensor)
                upload_duration = time.time() - upload_start

                console_log += f"[OK] Upload complete ({upload_duration:.2f}s)\n"
                console_log += f"  Pre-signed URL generated (valid 24h)\n"
                console_log += f"  URL: {input_url[:100]}...\n"
            else:
                input_url = image_reference
                console_log += f"\n{'='*55}\n"
                console_log += "Using provided image reference URL\n"
                console_log += f"  URL: {input_url[:80]}{'...' if len(input_url) > 80 else ''}\n"

            # Log the input URL being used
            console_log += f"\nInput URL for Photoshop API:\n"
            console_log += f"  {input_url}\n"

            # Build request
            request = RemoveBackgroundRequest(
                image=PhotoshopImageInput(
                    source=PhotoshopImageSource(
                        url=input_url,
                        storage=PhotoshopStorageType.EXTERNAL  # Required for S3/external URLs
                    )
                ),
                mode=mode,
                output=PhotoshopOutputOptions(
                    mediaType=output_media_type,
                    trim=trim,
                    backgroundColor=PhotoshopBackgroundColor(
                        red=background_color_red,
                        green=background_color_green,
                        blue=background_color_blue,
                        alpha=background_color_alpha,
                    ) if not trim else None,
                    colorDecontamination=color_decontamination,
                )
            )

            # Store the mode for later mask extraction
            _mode = mode

            # Log the actual request JSON being sent
            console_log += f"\nRequest JSON being sent to API:\n"
            console_log += f"{json.dumps(request.model_dump(exclude_none=True), indent=2)}\n"

            # Submit job
            submit_endpoint = ApiEndpoint(
                path="/v2/remove-background",
                method=HttpMethod.POST,
                request_model=RemoveBackgroundRequest,
                response_model=RemoveBackgroundResponse,
            )

            submit_op = SynchronousOperation(
                endpoint=submit_endpoint,
                request=request,
                api_base="https://image.adobe.io",
            )

            # Don't pass client - let operation create its own with correct base_url
            # But we need the auth headers, so temporarily update client's base_url
            client.base_url = "https://image.adobe.io"
            submit_response = await submit_op.execute(client=client)

            # Log submit response
            console_log += f"\nResponse: 200 OK\n"
            console_log += f"  jobId: {submit_response.jobId}\n"
            console_log += f"  statusUrl: {submit_response.statusUrl}\n"
            if hasattr(submit_response, 'status') and submit_response.status:
                console_log += f"  status: {submit_response.status} (job submitted, not yet processed)\n"

            # Log raw response for debugging
            console_log += f"\nRaw Submit Response:\n"
            console_log += f"{json.dumps(submit_response.model_dump(), indent=2)}\n"

            # Log polling start
            console_log += f"\n{'='*55}\n"
            job_id_short = submit_response.jobId.split(':')[-1][:8] if ':' in submit_response.jobId else submit_response.jobId[:8]
            console_log += f"GET /v2/status/{job_id_short}...\n"
            console_log += f"{'-'*55}\n"
            console_log += "Polling for job completion...\n"
            console_log += f"  Interval: {2.0}s\n"
            console_log += f"  Max attempts: {150} (timeout: {150 * 2.0}s)\n"

            # Poll for completion
            poll_endpoint = ApiEndpoint(
                path=f"/v2/status/{submit_response.jobId}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=PhotoshopJobStatus,
            )

            poll_op = PollingOperation(
                poll_endpoint=poll_endpoint,
                request=EmptyRequest(),
                completed_statuses=["succeeded"],
                failed_statuses=["failed"],
                status_extractor=lambda x: x.status,
                api_base="https://image.adobe.io",
                poll_interval=2.0,
                max_poll_attempts=150,  # 5 min timeout
                # node_id=unique_id,  # Disabled - no task tracking
            )

            # Track polling time
            poll_start_time = time.time()
            console_log += f"\nStarting polling at {time.strftime('%H:%M:%S', time.localtime(poll_start_time))}...\n"

            result = await poll_op.execute(client=client)

            # Calculate polling duration
            poll_end_time = time.time()
            poll_duration = poll_end_time - poll_start_time
            estimated_attempts = int(poll_duration / 2.0) + 1  # Based on 2s interval

            # Log result with detailed timing
            console_log += f"\nPolling completed at {time.strftime('%H:%M:%S', time.localtime(poll_end_time))}\n"
            console_log += f"  Duration: {poll_duration:.1f}s\n"
            console_log += f"  Estimated attempts: ~{estimated_attempts}\n"
            console_log += f"\nResponse: 200 OK\n"
            console_log += f"  status: {result.status}\n"
            console_log += f"  jobId: {result.jobId}\n"
            try:
                output_count = len(result.result.outputs) if result.result and result.result.outputs else 0
                console_log += f"  outputs: {output_count} image(s)\n"
            except Exception as e:
                console_log += f"  outputs: ERROR accessing outputs - {e}\n"

            # Log raw polling response for debugging
            console_log += f"\nRaw Poll Response:\n"
            try:
                console_log += f"{json.dumps(result.model_dump(), indent=2)}\n"
            except Exception as e:
                console_log += f"ERROR dumping result: {e}\n"
                console_log += f"result type: {type(result)}\n"
                console_log += f"result: {result}\n"

            # Validate outputs
            if not result.result or not result.result.outputs:
                console_log += f"\n{'='*55}\n"
                console_log += "ERROR: No outputs in response\n"
                console_log += f"  status: {result.status}\n"
                console_log += f"  jobId: {result.jobId}\n"
                console_log += f"  result exists: {result.result is not None}\n"
                if result.result:
                    console_log += f"  outputs exists: {hasattr(result.result, 'outputs')}\n"
                    if hasattr(result.result, 'outputs'):
                        console_log += f"  outputs value: {result.result.outputs}\n"
                console_log += f"\n**CONNECT debug_log OUTPUT TO ShowText TO SEE FULL RESPONSE**\n"
                print(console_log)  # Print to console for debugging
                raise Exception(f"No outputs returned from Photoshop API. Check console logs or connect debug_log output for details.")

            # Get output URL
            output_url = result.result.outputs[0].destination.url

            # Download result
            console_log += f"\n{'='*55}\n"
            console_log += "Downloading output...\n"

            # Download image
            async with aiohttp.ClientSession() as session:
                async with session.get(output_url) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to download result: HTTP {resp.status}")
                    image_bytes = await resp.read()

            console_log += "[OK] Downloaded image\n"

            # Convert to tensor
            img_pil = Image.open(io.BytesIO(image_bytes))

            # Extract alpha channel as mask (like LoadImage does)
            if 'A' in img_pil.getbands():
                # Extract alpha channel and invert it (1. - mask)
                mask = np.array(img_pil.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
                console_log += f"  Extracted alpha channel as mask\n"
            else:
                # No alpha channel - create empty mask
                mask = torch.zeros((img_pil.size[1], img_pil.size[0]), dtype=torch.float32, device="cpu")
                console_log += f"  No alpha channel found, created empty mask\n"

            # Convert image to RGB (remove alpha channel for image output)
            if img_pil.mode == 'RGBA':
                img_rgb = img_pil.convert('RGB')
            elif img_pil.mode not in ['RGB']:
                img_rgb = img_pil.convert('RGB')
            else:
                img_rgb = img_pil

            img_array = np.array(img_rgb).astype(np.float32) / 255.0

            # Ensure 3 channels (RGB)
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)

            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            console_log += f"\nOutput URL (valid for 24 hours):\n"
            console_log += f"  {output_url}\n"
            console_log += f"{'='*55}\n"

            return (img_tensor, mask, output_url, console_log)

        except Exception as e:
            console_log += f"\n{'='*55}\n"
            console_log += f"ERROR: {str(e)}\n"
            console_log += f"{'='*55}\n"
            print(console_log)  # Print to console even on error

            # Close client
            try:
                await client.close()
            except:
                pass  # Already closed or error closing

            # Re-raise the exception so the node fails properly
            raise
        finally:
            # Ensure client is closed
            try:
                await client.close()
            except:
                pass  # Already closed or error closing


class PhotoshopRefineMaskNode:
    """
    Refine and improve mask quality using Adobe Photoshop API.

    Features:
    - Dual input support for image and mask (tensor or pre-signed URL)
    - Color decontamination option
    - Comprehensive debug logging
    - Async polling for results
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
                # Image inputs (mutually exclusive)
                "image": ("IMAGE", {
                    "tooltip": "Input image tensor",
                }),
                "image_reference": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Pre-signed URL to input image (alternative to tensor)",
                }),

                # Mask inputs (mutually exclusive)
                "mask": ("MASK", {
                    "tooltip": "Mask tensor to refine",
                }),
                "mask_image": ("IMAGE", {
                    "tooltip": "Mask as IMAGE tensor to refine (alternative to MASK)",
                }),
                "mask_reference": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Pre-signed URL to mask image (alternative to tensor)",
                }),

                # Refine parameters
                "color_decontamination": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Return RGBA image with refined mask (true) or just refined mask (false)",
                }),
            },
            # "hidden": {
            #     "unique_id": "UNIQUE_ID",  # Disabled - no task tracking
            # },
        }

    def _build_debug_log(
        self,
        color_decontamination: bool,
        image: Optional[torch.Tensor],
        image_reference: str,
        mask: Optional[torch.Tensor],
        mask_image: Optional[torch.Tensor],
        mask_reference: str,
    ) -> str:
        """Build formatted debug log showing request details."""
        log = "=" * 55 + "\n"
        log += "POST /v2/refine-mask\n"
        log += "-" * 55 + "\n"
        log += "Headers:\n"
        log += "  Authorization: Bearer [TOKEN]\n"
        log += "  x-api-key: [API_KEY]\n"
        log += "  Content-Type: application/json\n"
        log += "\nRequest Body:\n"

        # Image source
        if image is not None:
            log += "  image:\n"
            log += "    source:\n"
            log += "      url: [S3_PRESIGNED_URL]\n"
        else:
            log += "  image:\n"
            log += "    source:\n"
            log += f"      url: {image_reference}\n"

        # Mask source
        if mask is not None:
            log += "  mask:\n"
            log += "    source:\n"
            log += "      url: [S3_PRESIGNED_URL] (MASK tensor)\n"
        elif mask_image is not None:
            log += "  mask:\n"
            log += "    source:\n"
            log += "      url: [S3_PRESIGNED_URL] (IMAGE tensor)\n"
        else:
            log += "  mask:\n"
            log += "    source:\n"
            log += f"      url: {mask_reference}\n"

        log += f"  colorDecontamination: {color_decontamination}\n"

        return log

    async def api_call(
        self,
        image: Optional[torch.Tensor] = None,
        image_reference: str = "",
        mask: Optional[torch.Tensor] = None,
        mask_image: Optional[torch.Tensor] = None,
        mask_reference: str = "",
        color_decontamination: bool = False,
        # unique_id: Optional[str] = None,  # Disabled - no task tracking
    ):
        """Refine mask using Photoshop API."""

        # Validate image inputs
        image_inputs_provided = sum([image is not None, bool(image_reference)])
        if image_inputs_provided == 0:
            raise ValueError("Must provide one of: 'image' or 'image_reference'")
        if image_inputs_provided > 1:
            raise ValueError("Cannot provide both 'image' and 'image_reference' - choose only one")

        # Validate mask inputs
        mask_inputs_provided = sum([mask is not None, mask_image is not None, bool(mask_reference)])
        if mask_inputs_provided == 0:
            raise ValueError("Must provide one of: 'mask', 'mask_image', or 'mask_reference'")
        if mask_inputs_provided > 1:
            raise ValueError("Cannot provide multiple mask inputs - choose only one: 'mask', 'mask_image', or 'mask_reference'")

        # Build initial debug log
        console_log = self._build_debug_log(
            color_decontamination=color_decontamination,
            image=image,
            image_reference=image_reference,
            mask=mask,
            mask_image=mask_image,
            mask_reference=mask_reference,
        )

        client = await create_adobe_client()

        try:
            # Convert mask to RGB format if provided as MASK tensor
            if mask is not None:
                if len(mask.shape) == 3:  # [B, H, W]
                    mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3)  # Convert to [B, H, W, 3]
                elif len(mask.shape) == 2:  # [H, W]
                    mask = mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)  # Convert to [1, H, W, 3]
                console_log += f"\n{'='*55}\n"
                console_log += "Converting mask to RGB format...\n"
                console_log += f"  Mask shape: {mask.shape}\n"

            # Get image URL
            if image is not None:
                console_log += f"\n{'='*55}\n"
                console_log += "Uploading image to S3...\n"

                # Get image info
                img_tensor = image[0]
                h, w, c = img_tensor.shape
                console_log += f"  Image size: {w}x{h} ({c} channels)\n"

                # Upload and measure time
                upload_start = time.time()
                image_url = await upload_image_to_s3(img_tensor)
                upload_duration = time.time() - upload_start

                console_log += f"[OK] Upload complete ({upload_duration:.2f}s)\n"
                console_log += f"  Pre-signed URL generated (valid 24h)\n"
                console_log += f"  URL: {image_url[:100]}...\n"
            else:
                image_url = image_reference
                console_log += f"\n{'='*55}\n"
                console_log += "Using provided image reference URL\n"
                console_log += f"  URL: {image_url[:80]}{'...' if len(image_url) > 80 else ''}\n"

            # Get mask URL
            if mask is not None:
                console_log += f"\n{'='*55}\n"
                console_log += "Uploading mask to S3...\n"

                # Get mask info
                mask_tensor = mask[0]
                h, w, c = mask_tensor.shape
                console_log += f"  Mask size: {w}x{h} ({c} channels)\n"

                # Upload and measure time
                upload_start = time.time()
                mask_url = await upload_image_to_s3(mask_tensor)
                upload_duration = time.time() - upload_start

                console_log += f"[OK] Upload complete ({upload_duration:.2f}s)\n"
                console_log += f"  Pre-signed URL generated (valid 24h)\n"
                console_log += f"  URL: {mask_url[:100]}...\n"
            elif mask_image is not None:
                console_log += f"\n{'='*55}\n"
                console_log += "Uploading mask image to S3...\n"

                # Get mask image info
                mask_tensor = mask_image[0]
                h, w, c = mask_tensor.shape
                console_log += f"  Mask image size: {w}x{h} ({c} channels)\n"

                # Upload and measure time
                upload_start = time.time()
                mask_url = await upload_image_to_s3(mask_tensor)
                upload_duration = time.time() - upload_start

                console_log += f"[OK] Upload complete ({upload_duration:.2f}s)\n"
                console_log += f"  Pre-signed URL generated (valid 24h)\n"
                console_log += f"  URL: {mask_url[:100]}...\n"
            else:
                mask_url = mask_reference
                console_log += f"\n{'='*55}\n"
                console_log += "Using provided mask reference URL\n"
                console_log += f"  URL: {mask_url[:80]}{'...' if len(mask_url) > 80 else ''}\n"

            # Log the URLs being used
            console_log += f"\nURLs for Photoshop API:\n"
            console_log += f"  Image: {image_url}\n"
            console_log += f"  Mask: {mask_url}\n"

            # Build request
            request = RefineMaskRequest(
                image=PhotoshopImageInput(
                    source=PhotoshopImageSource(
                        url=image_url
                        # Note: storage field is NOT included for v1/refine-mask endpoint
                    )
                ),
                mask=PhotoshopImageInput(
                    source=PhotoshopImageSource(
                        url=mask_url
                        # Note: storage field is NOT included for v1/refine-mask endpoint
                    )
                ),
                colorDecontamination=color_decontamination,
            )

            # Log the actual request JSON being sent
            console_log += f"\nRequest JSON being sent to API:\n"
            console_log += f"{json.dumps(request.model_dump(exclude_none=True), indent=2)}\n"

            # Submit job
            submit_endpoint = ApiEndpoint(
                path="/v1/refine-mask",
                method=HttpMethod.POST,
                request_model=RefineMaskRequest,
                response_model=RemoveBackgroundResponse,  # Reuse same response model
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
            console_log += f"  jobId: {submit_response.jobId}\n"
            console_log += f"  statusUrl: {submit_response.statusUrl}\n"
            if hasattr(submit_response, 'status') and submit_response.status:
                console_log += f"  status: {submit_response.status} (job submitted, not yet processed)\n"

            # Log raw response for debugging
            console_log += f"\nRaw Submit Response:\n"
            console_log += f"{json.dumps(submit_response.model_dump(), indent=2)}\n"

            # Parse statusUrl to get correct polling endpoint
            parsed_status_url = urlparse(submit_response.statusUrl)
            status_base_url = f"{parsed_status_url.scheme}://{parsed_status_url.netloc}"
            status_path = parsed_status_url.path

            # Log polling start
            console_log += f"\n{'='*55}\n"
            console_log += f"GET {status_path}\n"
            console_log += f"{'-'*55}\n"
            console_log += "Polling for job completion...\n"
            console_log += f"  Status URL: {submit_response.statusUrl}\n"
            console_log += f"  Interval: {2.0}s\n"
            console_log += f"  Max attempts: {150} (timeout: {150 * 2.0}s)\n"

            # Poll for completion using the statusUrl from the API
            # v1 API returns different structure with semanticMasks/backgroundMasks
            poll_endpoint = ApiEndpoint(
                path=status_path,
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=RefineMaskStatusResponse,
            )

            poll_op = PollingOperation(
                poll_endpoint=poll_endpoint,
                request=EmptyRequest(),
                completed_statuses=["succeeded"],
                failed_statuses=["failed"],
                status_extractor=lambda x: x.status,
                api_base=status_base_url,
                poll_interval=2.0,
                max_poll_attempts=150,  # 5 min timeout
                # node_id=unique_id,  # Disabled - no task tracking
            )

            # Track polling time
            poll_start_time = time.time()
            console_log += f"\nStarting polling at {time.strftime('%H:%M:%S', time.localtime(poll_start_time))}...\n"

            result = await poll_op.execute(client=client)

            # Calculate polling duration
            poll_end_time = time.time()
            poll_duration = poll_end_time - poll_start_time
            estimated_attempts = int(poll_duration / 2.0) + 1  # Based on 2s interval

            # Log result with detailed timing
            console_log += f"\nPolling completed at {time.strftime('%H:%M:%S', time.localtime(poll_end_time))}\n"
            console_log += f"  Duration: {poll_duration:.1f}s\n"
            console_log += f"  Estimated attempts: ~{estimated_attempts}\n"
            console_log += f"\nResponse: 200 OK\n"
            console_log += f"  status: {result.status}\n"
            console_log += f"  jobId: {result.jobId}\n"

            # Check for image or mask result from v1 API response
            # API returns 'image' when colorDecontamination=true, 'mask' when false
            has_image = result.image is not None
            has_mask = result.mask is not None
            console_log += f"  image result: {'Yes' if has_image else 'No'}\n"
            console_log += f"  mask result: {'Yes' if has_mask else 'No'}\n"

            if has_image:
                console_log += f"    mediaType: {result.image.mediaType}\n"
                console_log += f"    boundingBox: {result.image.boundingBox.width}x{result.image.boundingBox.height} at ({result.image.boundingBox.x},{result.image.boundingBox.y})\n"
            elif has_mask:
                console_log += f"    mediaType: {result.mask.mediaType}\n"
                console_log += f"    boundingBox: {result.mask.boundingBox.width}x{result.mask.boundingBox.height} at ({result.mask.boundingBox.x},{result.mask.boundingBox.y})\n"

            # Log complete polling response as JSON for debugging
            console_log += f"\n{'='*55}\n"
            console_log += "FULL POLLING RESPONSE (JSON):\n"
            console_log += f"{'-'*55}\n"
            console_log += f"URL: {submit_response.statusUrl}\n"
            console_log += f"{'-'*55}\n"
            try:
                # Use mode='json' and exclude_none=False to get complete representation
                response_dict = result.model_dump(mode='json', exclude_none=False)
                console_log += f"{json.dumps(response_dict, indent=2)}\n"
            except Exception as e:
                console_log += f"ERROR serializing response to JSON: {e}\n"
                try:
                    # Fallback to regular model_dump
                    console_log += f"{json.dumps(result.model_dump(), indent=2)}\n"
                except Exception as e2:
                    console_log += f"ERROR with fallback: {e2}\n"
                    console_log += f"result type: {type(result)}\n"
                    console_log += f"result repr: {repr(result)}\n"
            console_log += f"{'='*55}\n"

            # Validate outputs - check for image or mask result
            if not result.image and not result.mask:
                console_log += f"\n{'='*55}\n"
                console_log += "ERROR: No image or mask in response\n"
                console_log += f"  status: {result.status}\n"
                console_log += f"  jobId: {result.jobId}\n"
                console_log += f"  image: {result.image}\n"
                console_log += f"  mask: {result.mask}\n"
                console_log += f"\n**CONNECT debug_log OUTPUT TO ShowText TO SEE FULL RESPONSE**\n"
                print(console_log)  # Print to console for debugging
                raise Exception(f"No image or mask returned from Photoshop API. Check console logs or connect debug_log output for details.")

            # Get output URL from image or mask result
            if result.image:
                output_url = result.image.destination.url
                console_log += f"\nRefined image URL obtained (colorDecontamination=true)\n"
                console_log += f"  mediaType: {result.image.mediaType}\n"
                console_log += f"  boundingBox: {result.image.boundingBox.width}x{result.image.boundingBox.height}\n"
            else:
                output_url = result.mask.destination.url
                console_log += f"\nRefined mask URL obtained (colorDecontamination=false)\n"
                console_log += f"  mediaType: {result.mask.mediaType}\n"
                console_log += f"  boundingBox: {result.mask.boundingBox.width}x{result.mask.boundingBox.height}\n"

            # Download result
            console_log += f"\n{'='*55}\n"
            console_log += "Downloading output...\n"

            # Download image
            async with aiohttp.ClientSession() as session:
                async with session.get(output_url) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to download result: HTTP {resp.status}")
                    image_bytes = await resp.read()

            console_log += "[OK] Downloaded image\n"

            # Convert to tensor
            img_pil = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if needed (handle RGBA, L, etc.)
            if img_pil.mode not in ['RGB', 'RGBA']:
                img_pil = img_pil.convert('RGB')

            img_array = np.array(img_pil).astype(np.float32) / 255.0

            # Ensure 3 channels (RGB)
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:  # RGBA - keep alpha for now
                # ComfyUI expects RGB, but we'll keep RGBA and let it handle it
                pass

            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            console_log += f"\nOutput URL (valid for 24 hours):\n"
            console_log += f"  {output_url}\n"
            console_log += f"{'='*55}\n"

            return (img_tensor, output_url, console_log)

        except Exception as e:
            console_log += f"\n{'='*55}\n"
            console_log += f"ERROR: {str(e)}\n"
            console_log += f"{'='*55}\n"
            print(console_log)  # Print to console even on error

            # Close client
            try:
                await client.close()
            except:
                pass  # Already closed or error closing

            # Re-raise the exception so the node fails properly
            raise
        finally:
            # Ensure client is closed
            try:
                await client.close()
            except:
                pass  # Already closed or error closing


class PhotoshopFillMaskedAreasNode:
    """
    Fill masked areas in images using Adobe Photoshop API.

    Features:
    - Dual input support for image and mask (tensor or pre-signed URL)
    - Comprehensive debug logging
    - Async polling for results
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
                # Image inputs (mutually exclusive)
                "image": ("IMAGE", {
                    "tooltip": "Input image tensor",
                }),
                "image_reference": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Pre-signed URL to input image (alternative to tensor)",
                }),

                # Mask inputs (mutually exclusive)
                "mask": ("MASK", {
                    "tooltip": "Mask tensor indicating areas to fill",
                }),
                "mask_image": ("IMAGE", {
                    "tooltip": "Mask as IMAGE tensor (alternative to MASK)",
                }),
                "mask_reference": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Pre-signed URL to mask image (alternative to tensor)",
                }),
            },
            # "hidden": {
            #     "unique_id": "UNIQUE_ID",  # Disabled - no task tracking
            # },
        }

    def _build_debug_log(
        self,
        image: Optional[torch.Tensor],
        image_reference: str,
        mask: Optional[torch.Tensor],
        mask_image: Optional[torch.Tensor],
        mask_reference: str,
    ) -> str:
        """Build formatted debug log showing request details."""
        log = "=" * 55 + "\n"
        log += "POST /v1/fill-masked-areas\n"
        log += "-" * 55 + "\n"
        log += "Headers:\n"
        log += "  Authorization: Bearer [TOKEN]\n"
        log += "  x-api-key: [API_KEY]\n"
        log += "  Content-Type: application/json\n"
        log += "\nRequest Body:\n"

        # Image source
        if image is not None:
            log += "  image:\n"
            log += "    source:\n"
            log += "      url: [S3_PRESIGNED_URL]\n"
        else:
            log += "  image:\n"
            log += "    source:\n"
            log += f"      url: {image_reference}\n"

        # Mask source
        if mask is not None:
            log += "  masks:\n"
            log += "    - source:\n"
            log += "        url: [S3_PRESIGNED_URL] (MASK tensor)\n"
        elif mask_image is not None:
            log += "  masks:\n"
            log += "    - source:\n"
            log += "        url: [S3_PRESIGNED_URL] (IMAGE tensor)\n"
        else:
            log += "  masks:\n"
            log += "    - source:\n"
            log += f"        url: {mask_reference}\n"

        return log

    async def api_call(
        self,
        image: Optional[torch.Tensor] = None,
        image_reference: str = "",
        mask: Optional[torch.Tensor] = None,
        mask_image: Optional[torch.Tensor] = None,
        mask_reference: str = "",
        # unique_id: Optional[str] = None,  # Disabled - no task tracking
    ):
        """Fill masked areas in image using Photoshop API."""

        # Validate image inputs
        image_inputs_provided = sum([image is not None, bool(image_reference)])
        if image_inputs_provided == 0:
            raise ValueError("Must provide one of: 'image' or 'image_reference'")
        if image_inputs_provided > 1:
            raise ValueError("Cannot provide both 'image' and 'image_reference' - choose only one")

        # Validate mask inputs
        mask_inputs_provided = sum([mask is not None, mask_image is not None, bool(mask_reference)])
        if mask_inputs_provided == 0:
            raise ValueError("Must provide one of: 'mask', 'mask_image', or 'mask_reference'")
        if mask_inputs_provided > 1:
            raise ValueError("Cannot provide multiple mask inputs - choose only one: 'mask', 'mask_image', or 'mask_reference'")

        # Build initial debug log
        console_log = self._build_debug_log(
            image=image,
            image_reference=image_reference,
            mask=mask,
            mask_image=mask_image,
            mask_reference=mask_reference,
        )

        client = await create_adobe_client()

        try:
            # Convert mask to RGB format if provided as MASK tensor
            if mask is not None:
                if len(mask.shape) == 3:  # [B, H, W]
                    mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3)  # Convert to [B, H, W, 3]
                elif len(mask.shape) == 2:  # [H, W]
                    mask = mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)  # Convert to [1, H, W, 3]
                console_log += f"\n{'='*55}\n"
                console_log += "Converting mask to RGB format...\n"
                console_log += f"  Mask shape: {mask.shape}\n"

            # Get image URL
            if image is not None:
                console_log += f"\n{'='*55}\n"
                console_log += "Uploading image to S3...\n"

                # Get image info
                img_tensor = image[0]
                h, w, c = img_tensor.shape
                console_log += f"  Image size: {w}x{h} ({c} channels)\n"

                # Upload and measure time
                upload_start = time.time()
                image_url = await upload_image_to_s3(img_tensor)
                upload_duration = time.time() - upload_start

                console_log += f"[OK] Upload complete ({upload_duration:.2f}s)\n"
                console_log += f"  Pre-signed URL generated (valid 24h)\n"
                console_log += f"  URL: {image_url[:100]}...\n"
            else:
                image_url = image_reference
                console_log += f"\n{'='*55}\n"
                console_log += "Using provided image reference URL\n"
                console_log += f"  URL: {image_url[:80]}{'...' if len(image_url) > 80 else ''}\n"

            # Get mask URL
            if mask is not None:
                console_log += f"\n{'='*55}\n"
                console_log += "Uploading mask to S3...\n"

                # Get mask info
                mask_tensor = mask[0]
                h, w, c = mask_tensor.shape
                console_log += f"  Mask size: {w}x{h} ({c} channels)\n"

                # Upload and measure time
                upload_start = time.time()
                mask_url = await upload_image_to_s3(mask_tensor)
                upload_duration = time.time() - upload_start

                console_log += f"[OK] Upload complete ({upload_duration:.2f}s)\n"
                console_log += f"  Pre-signed URL generated (valid 24h)\n"
                console_log += f"  URL: {mask_url[:100]}...\n"
            elif mask_image is not None:
                console_log += f"\n{'='*55}\n"
                console_log += "Uploading mask image to S3...\n"

                # Get mask image info
                mask_tensor = mask_image[0]
                h, w, c = mask_tensor.shape
                console_log += f"  Mask image size: {w}x{h} ({c} channels)\n"

                # Upload and measure time
                upload_start = time.time()
                mask_url = await upload_image_to_s3(mask_tensor)
                upload_duration = time.time() - upload_start

                console_log += f"[OK] Upload complete ({upload_duration:.2f}s)\n"
                console_log += f"  Pre-signed URL generated (valid 24h)\n"
                console_log += f"  URL: {mask_url[:100]}...\n"
            else:
                mask_url = mask_reference
                console_log += f"\n{'='*55}\n"
                console_log += "Using provided mask reference URL\n"
                console_log += f"  URL: {mask_url[:80]}{'...' if len(mask_url) > 80 else ''}\n"

            # Log the URLs being used
            console_log += f"\nURLs for Photoshop API:\n"
            console_log += f"  Image: {image_url}\n"
            console_log += f"  Mask: {mask_url}\n"

            # Build request
            request = FillMaskedAreasRequest(
                image=PhotoshopImageInput(
                    source=PhotoshopImageSource(
                        url=image_url
                    )
                ),
                masks=[
                    PhotoshopImageInput(
                        source=PhotoshopImageSource(
                            url=mask_url
                        )
                    )
                ],
            )

            # Log the actual request JSON being sent
            console_log += f"\nRequest JSON being sent to API:\n"
            console_log += f"{json.dumps(request.model_dump(exclude_none=True), indent=2)}\n"

            # Submit job
            submit_endpoint = ApiEndpoint(
                path="/v1/fill-masked-areas",
                method=HttpMethod.POST,
                request_model=FillMaskedAreasRequest,
                response_model=RemoveBackgroundResponse,  # Reuse same response model
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
            console_log += f"  jobId: {submit_response.jobId}\n"
            console_log += f"  statusUrl: {submit_response.statusUrl}\n"
            if hasattr(submit_response, 'status') and submit_response.status:
                console_log += f"  status: {submit_response.status} (job submitted, not yet processed)\n"

            # Log raw response for debugging
            console_log += f"\nRaw Submit Response:\n"
            console_log += f"{json.dumps(submit_response.model_dump(), indent=2)}\n"

            # Parse statusUrl to get correct polling endpoint
            parsed_status_url = urlparse(submit_response.statusUrl)
            status_base_url = f"{parsed_status_url.scheme}://{parsed_status_url.netloc}"
            status_path = parsed_status_url.path

            # Log polling start
            console_log += f"\n{'='*55}\n"
            console_log += f"GET {status_path}\n"
            console_log += f"{'-'*55}\n"
            console_log += "Polling for job completion...\n"
            console_log += f"  Status URL: {submit_response.statusUrl}\n"
            console_log += f"  Interval: {2.0}s\n"
            console_log += f"  Max attempts: {150} (timeout: {150 * 2.0}s)\n"

            # Poll for completion using the statusUrl from the API
            # v1 API returns different structure with image field
            poll_endpoint = ApiEndpoint(
                path=status_path,
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=FillMaskedAreasStatusResponse,
            )

            poll_op = PollingOperation(
                poll_endpoint=poll_endpoint,
                request=EmptyRequest(),
                completed_statuses=["succeeded"],
                failed_statuses=["failed"],
                status_extractor=lambda x: x.status,
                api_base=status_base_url,
                poll_interval=2.0,
                max_poll_attempts=150,  # 5 min timeout
                # node_id=unique_id,  # Disabled - no task tracking
            )

            # Track polling time
            poll_start_time = time.time()
            console_log += f"\nStarting polling at {time.strftime('%H:%M:%S', time.localtime(poll_start_time))}...\n"

            result = await poll_op.execute(client=client)

            # Calculate polling duration
            poll_end_time = time.time()
            poll_duration = poll_end_time - poll_start_time
            estimated_attempts = int(poll_duration / 2.0) + 1  # Based on 2s interval

            # Log result with detailed timing
            console_log += f"\nPolling completed at {time.strftime('%H:%M:%S', time.localtime(poll_end_time))}\n"
            console_log += f"  Duration: {poll_duration:.1f}s\n"
            console_log += f"  Estimated attempts: ~{estimated_attempts}\n"
            console_log += f"\nResponse: 200 OK\n"
            console_log += f"  status: {result.status}\n"
            console_log += f"  jobId: {result.jobId}\n"

            # Check for image result from v1 API response
            has_image = result.image is not None
            console_log += f"  image result: {'Yes' if has_image else 'No'}\n"

            if has_image:
                console_log += f"    mediaType: {result.image.mediaType}\n"

            # Log complete polling response as JSON for debugging
            console_log += f"\n{'='*55}\n"
            console_log += "FULL POLLING RESPONSE (JSON):\n"
            console_log += f"{'-'*55}\n"
            console_log += f"URL: {submit_response.statusUrl}\n"
            console_log += f"{'-'*55}\n"
            try:
                # Use mode='json' and exclude_none=False to get complete representation
                response_dict = result.model_dump(mode='json', exclude_none=False)
                console_log += f"{json.dumps(response_dict, indent=2)}\n"
            except Exception as e:
                console_log += f"ERROR serializing response to JSON: {e}\n"
                try:
                    # Fallback to regular model_dump
                    console_log += f"{json.dumps(result.model_dump(), indent=2)}\n"
                except Exception as e2:
                    console_log += f"ERROR with fallback: {e2}\n"
                    console_log += f"result type: {type(result)}\n"
                    console_log += f"result repr: {repr(result)}\n"
            console_log += f"{'='*55}\n"

            # Validate outputs - check for image result
            if not result.image:
                console_log += f"\n{'='*55}\n"
                console_log += "ERROR: No image in response\n"
                console_log += f"  status: {result.status}\n"
                console_log += f"  jobId: {result.jobId}\n"
                console_log += f"  image: {result.image}\n"
                console_log += f"\n**CONNECT debug_log OUTPUT TO ShowText TO SEE FULL RESPONSE**\n"
                print(console_log)  # Print to console for debugging
                raise Exception(f"No image returned from Photoshop API. Check console logs or connect debug_log output for details.")

            # Get output URL from image result
            output_url = result.image.destination.url
            console_log += f"\nFilled image URL obtained\n"
            console_log += f"  mediaType: {result.image.mediaType}\n"

            # Download result
            console_log += f"\n{'='*55}\n"
            console_log += "Downloading output...\n"

            # Download image
            async with aiohttp.ClientSession() as session:
                async with session.get(output_url) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to download result: HTTP {resp.status}")
                    image_bytes = await resp.read()

            console_log += "[OK] Downloaded image\n"

            # Convert to tensor
            img_pil = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if needed (handle RGBA, L, etc.)
            if img_pil.mode not in ['RGB', 'RGBA']:
                img_pil = img_pil.convert('RGB')

            img_array = np.array(img_pil).astype(np.float32) / 255.0

            # Ensure 3 channels (RGB)
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:  # RGBA - keep alpha for now
                # ComfyUI expects RGB, but we'll keep RGBA and let it handle it
                pass

            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            console_log += f"\nOutput URL (valid for 24 hours):\n"
            console_log += f"  {output_url}\n"
            console_log += f"{'='*55}\n"

            return (img_tensor, output_url, console_log)

        except Exception as e:
            console_log += f"\n{'='*55}\n"
            console_log += f"ERROR: {str(e)}\n"
            console_log += f"{'='*55}\n"
            print(console_log)  # Print to console even on error

            # Close client
            try:
                await client.close()
            except:
                pass  # Already closed or error closing

            # Re-raise the exception so the node fails properly
            raise
        finally:
            # Ensure client is closed
            try:
                await client.close()
            except:
                pass  # Already closed or error closing
