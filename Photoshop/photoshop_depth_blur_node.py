"""
Adobe Photoshop Depth Blur Node

Apply depth blur effects to images using Adobe Photoshop API.
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
    PhotoshopJobStatusEnum,
    PhotoshopActionsInput,
    PhotoshopActionsOutput,
    FocalSelector,
    DepthBlurOptions,
    DepthBlurRequest,
    DepthBlurResponse,
    DepthBlurJobStatus,
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
from comfy.utils import ProgressBar


class PhotoshopDepthBlurNode:
    """
    Apply depth blur effects to images using Adobe Photoshop API.

    Features:
    - Automatic subject focus or manual focal point selection
    - Adjustable blur strength, haze, and grain
    - Color adjustments (temperature, tint, saturation, brightness)
    - Dual input support (tensor or pre-signed URL)
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
                # Input image
                "image": ("IMAGE", {
                    "tooltip": "Input image tensor to process",
                }),
                "image_reference": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Pre-signed URL to input image (alternative to tensor)",
                }),

                # Focus settings
                "focus_subject": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Automatically select prominent subject for focus",
                }),
                "focal_distance": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Distance of the point to be in focus (0=nearest, 100=furthest)",
                }),
                "focal_range": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Range of the focal point",
                }),
                "focal_x": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "X coordinate of focal point (0.0-1.0, normalized)",
                }),
                "focal_y": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Y coordinate of focal point (0.0-1.0, normalized)",
                }),

                # Blur effects
                "blur_strength": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Amount of blur to apply",
                }),
                "haze": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Amount of haze to apply",
                }),
                "grain": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Amount of graining to add to the image",
                }),

                # Color adjustments
                "temperature": ("INT", {
                    "default": 0,
                    "min": -50,
                    "max": 50,
                    "tooltip": "Temperature adjustment (-50=coldest, 50=warmest)",
                }),
                "tint": ("INT", {
                    "default": 0,
                    "min": -50,
                    "max": 50,
                    "tooltip": "Amount of tint to apply",
                }),
                "saturation": ("INT", {
                    "default": 0,
                    "min": -50,
                    "max": 50,
                    "tooltip": "Saturation adjustment (-50=unsaturated, 50=fully saturated)",
                }),
                "brightness": ("INT", {
                    "default": 0,
                    "min": -50,
                    "max": 50,
                    "tooltip": "Brightness adjustment",
                }),

                # Output settings
                "output_type": (
                    ["image/vnd.adobe.photoshop", "image/jpeg", "image/png", "image/tiff"],
                    {
                        "default": "image/png",
                        "tooltip": "Output file format",
                    }
                ),
                "output_quality": ("INT", {
                    "default": 7,
                    "min": 1,
                    "max": 12,
                    "tooltip": "JPEG quality level (1-12, only for JPEG output)",
                }),
                "compression": (["small", "medium", "large"], {
                    "default": "small",
                    "tooltip": "PNG compression level",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        image: Optional[torch.Tensor] = None,
        image_reference: str = "",
        focus_subject: bool = False,
        focal_distance: int = 0,
        focal_range: int = 0,
        focal_x: float = 0.0,
        focal_y: float = 0.0,
        blur_strength: int = 50,
        haze: int = 0,
        grain: int = 0,
        temperature: int = 0,
        tint: int = 0,
        saturation: int = 0,
        brightness: int = 0,
        output_type: str = "image/png",
        output_quality: int = 7,
        compression: str = "small",
        unique_id: Optional[str] = None,
    ):
        """Execute Photoshop Depth Blur on image using Photoshop API."""

        # Validate inputs
        if image is None and not image_reference:
            raise ValueError("Must provide either 'image' or 'image_reference'")
        if image is not None and image_reference:
            raise ValueError("Cannot provide both 'image' and 'image_reference' - choose only one")

        # Validate output parameters
        if output_type == "image/jpeg" and compression != "small":
            raise ValueError(
                f"The 'compression' parameter only applies to PNG output.\n"
                f"Current output type: {output_type}\n"
                f"Either change output_type to 'image/png' or leave compression at default."
            )
        if output_type == "image/png" and output_quality != 7:
            raise ValueError(
                f"The 'output_quality' parameter only applies to JPEG output.\n"
                f"Current output type: {output_type}\n"
                f"Either change output_type to 'image/jpeg' or leave output_quality at default (7)."
            )
        if output_type in ["image/vnd.adobe.photoshop", "image/tiff"]:
            if output_quality != 7:
                raise ValueError(
                    f"The 'output_quality' parameter only applies to JPEG output.\n"
                    f"Current output type: {output_type}\n"
                    f"Leave output_quality at default (7) for {output_type} output."
                )
            if compression != "small":
                raise ValueError(
                    f"The 'compression' parameter only applies to PNG output.\n"
                    f"Current output type: {output_type}\n"
                    f"Leave compression at default for {output_type} output."
                )

        console_log = "=" * 55 + "\n"
        console_log += "POST /pie/psdService/depthBlur\n"
        console_log += "-" * 55 + "\n"

        client = await create_adobe_client()

        try:
            # Get input URL
            if image is not None:
                console_log += f"\n{'='*55}\n"
                console_log += "Uploading input image to S3...\n"

                img_tensor = image[0]
                h, w, c = img_tensor.shape
                console_log += f"  Image size: {w}x{h} ({c} channels)\n"

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

            # Create output URL
            console_log += f"\n{'='*55}\n"
            console_log += "Generating output pre-signed PUT URL for S3...\n"

            extension_map = {
                "image/vnd.adobe.photoshop": "psd",
                "vnd.adobe.photoshop": "psd",
                "image/jpeg": "jpg",
                "image/png": "png",
                "image/tiff": "tiff",
            }
            file_extension = extension_map.get(output_type, "png")

            url_gen_start = time.time()
            output_url_presigned, output_filename = await generate_output_presigned_url(file_extension=file_extension)
            url_gen_duration = time.time() - url_gen_start

            console_log += f"[OK] Generated pre-signed PUT URL ({url_gen_duration:.2f}s)\n"
            console_log += f"  URL: {output_url_presigned[:80]}...\n"
            console_log += f"  Output filename: {output_filename}\n"

            output_params = {
                "href": output_url_presigned,
                "storage": "external",
                "type": output_type,
                "overwrite": True,
            }

            if output_type == "image/jpeg":
                output_params["quality"] = output_quality
            elif output_type == "image/png":
                output_params["compression"] = compression

            # Build depth blur options
            options_dict = {}

            if focus_subject:
                options_dict["focusSubject"] = True
            if focal_distance > 0:
                options_dict["focalDistance"] = focal_distance
            if focal_range > 0:
                options_dict["focalRange"] = focal_range
            if focal_x > 0 or focal_y > 0:
                options_dict["focalSelector"] = FocalSelector(x=focal_x, y=focal_y)
            if blur_strength != 50:
                options_dict["blurStrength"] = blur_strength
            if haze > 0:
                options_dict["haze"] = haze
            if grain > 0:
                options_dict["grain"] = grain
            if temperature != 0:
                options_dict["temp"] = temperature
            if tint != 0:
                options_dict["tint"] = tint
            if saturation != 0:
                options_dict["saturation"] = saturation
            if brightness != 0:
                options_dict["brightness"] = brightness

            depth_blur_options = DepthBlurOptions(**options_dict) if options_dict else None

            # Build request
            request = DepthBlurRequest(
                inputs=[
                    PhotoshopActionsInput(
                        href=input_url,
                        storage="external"
                    )
                ],
                outputs=[
                    PhotoshopActionsOutput(**output_params)
                ],
                options=depth_blur_options,
            )

            console_log += f"\nRequest JSON being sent to API:\n"
            console_log += f"{json.dumps(request.model_dump(exclude_none=True), indent=2)}\n"

            # Submit job
            submit_endpoint = ApiEndpoint(
                path="/pie/psdService/depthBlur",
                method=HttpMethod.POST,
                request_model=DepthBlurRequest,
                response_model=DepthBlurResponse,
            )

            submit_op = SynchronousOperation(
                endpoint=submit_endpoint,
                request=request,
                api_base="https://image.adobe.io",
            )

            client.base_url = "https://image.adobe.io"
            submit_response = await submit_op.execute(client=client)

            console_log += f"\nResponse: 202 Accepted\n"

            job_id = submit_response.jobId
            status_url = submit_response.statusUrl

            if not job_id or not status_url:
                console_log += f"ERROR: Failed to extract jobId or statusUrl from response\n"
                raise Exception("Failed to extract job information from API response")

            console_log += f"  jobId: {job_id}\n"
            console_log += f"  statusUrl: {status_url}\n"

            # Parse statusUrl
            parsed_status_url = urlparse(status_url)
            status_base_url = f"{parsed_status_url.scheme}://{parsed_status_url.netloc}"
            status_path = parsed_status_url.path

            console_log += f"\n{'='*55}\n"
            console_log += f"GET {status_path}\n"
            console_log += f"{'-'*55}\n"
            console_log += "Polling for job completion...\n"

            # Poll for completion
            poll_endpoint = ApiEndpoint(
                path=status_path,
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=DepthBlurJobStatus,
            )

            poll_op = PollingOperation(
                poll_endpoint=poll_endpoint,
                request=EmptyRequest(),
                completed_statuses=["succeeded"],
                failed_statuses=["failed"],
                status_extractor=lambda x: x.status,
                api_base=status_base_url,
                poll_interval=2.0,
                max_poll_attempts=300,
                estimated_duration=30.0,  # Estimated 30s for Depth Blur
                node_id=unique_id,
            )

            poll_start_time = time.time()
            result = await poll_op.execute(client=client)

            poll_duration = time.time() - poll_start_time
            console_log += f"\n[OK] Polling completed in {poll_duration:.1f}s\n"

            # Validate outputs
            if not result.result or not result.result.outputs:
                console_log += f"\n{'='*55}\n"
                console_log += "ERROR: No outputs in response\n"
                print(console_log)
                raise Exception(f"No outputs returned from Photoshop Depth Blur API")

            # Generate GET URL for downloading
            console_log += f"\n{'='*55}\n"
            console_log += "Generating download URL...\n"

            download_url = await generate_download_url(output_filename)
            console_log += f"[OK] Generated download URL\n"

            # Download result
            console_log += "\nDownloading output image...\n"

            async with aiohttp.ClientSession() as session:
                async with session.get(download_url) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to download result: HTTP {resp.status}")
                    image_bytes = await resp.read()

            console_log += "[OK] Downloaded image\n"

            # Convert to tensor
            img_pil = Image.open(io.BytesIO(image_bytes))

            if img_pil.mode not in ['RGB', 'RGBA']:
                img_pil = img_pil.convert('RGB')

            img_array = np.array(img_pil).astype(np.float32) / 255.0

            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                pass

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
    "PhotoshopDepthBlurNode": PhotoshopDepthBlurNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoshopDepthBlurNode": "Photoshop Depth Blur",
}
