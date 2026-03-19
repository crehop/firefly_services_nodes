"""
Adobe Photoshop Photo Restoration Node

Apply Photo Restoration neural filter to images using Adobe Photoshop API.
Uses the neuralGalleryFilters ActionJSON with 7 adjustable sliders.
"""

from __future__ import annotations
from typing import Optional
import torch
import aiohttp
import numpy as np
import io
import time
import json
import os
import copy
from urllib.parse import urlparse
from PIL import Image

from .photoshop_api import (
    PhotoshopActionsInput,
    PhotoshopActionsOutput,
    ActionJsonOptions,
    ActionJsonRequest,
    ActionJsonResponse,
    ActionJsonJobStatus,
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


# Load the Photo Restoration template JSON at module level
_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "photo_restoration_template.json")
with open(_TEMPLATE_PATH, "r", encoding="utf-8") as _f:
    _PHOTO_RESTORATION_TEMPLATE = json.load(_f)


def _set_node_param(nodes, index, param_key, value):
    """Set a parameter value on a graph node."""
    node = nodes[index]
    params = node.get("spl::params", {})
    if param_key in params and isinstance(params[param_key], dict):
        params[param_key]["spl::value"] = value
    else:
        params[param_key] = {"spl::ID": 0, "spl::type": "scalar", "spl::value": value}


def _build_photo_restoration_action_json(
    photo_enhancement: int = 50,
    face_enhancement: int = 60,
    scratch_reduction: int = 0,
    noise_reduction: int = 0,
    color_noise_reduction: int = 0,
    halftone_artifacts_reduction: int = 0,
    jpeg_artifacts_reduction: int = 0,
) -> list:
    """Build the neuralGalleryFilters ActionJSON with slider values injected.

    Uses the max (100%) template as the base graph and modifies specific
    node parameters based on slider values. Mappings derived by diffing
    Photoshop exports at slider=1, slider=mid, and slider=100.
    """
    payload = copy.deepcopy(_PHOTO_RESTORATION_TEMPLATE)
    nodes = payload[0]["NF_SPL_GRAPH"]["spl::nodes"]

    # --- Update UI metadata (spl::values) ---
    filter_stack = payload[0]["NF_UI_DATA"]["spl::filterStack"]
    for stack_entry in filter_stack:
        crop_states = stack_entry.get("spl::cropStates", [])
        for crop_state in crop_states:
            values = crop_state.get("spl::values", {})
            values["spl::enhance"] = photo_enhancement
            values["spl::faceEnhance"] = face_enhancement
            values["spl::scratches"] = scratch_reduction
            values["spl::luminanceNoise"] = noise_reduction
            values["spl::colorNoise"] = color_noise_reduction
            values["spl::halftoning"] = halftone_artifacts_reduction
            values["spl::removeJpeg"] = jpeg_artifacts_reduction

    # --- Update graph node parameters ---
    # Clamp all sliders to 1-100 range (0 would need passthrough graph)
    enhance = max(1, min(100, photo_enhancement))
    scratches = max(1, min(100, scratch_reduction))
    lum_noise = max(1, min(100, noise_reduction))
    col_noise = max(1, min(100, color_noise_reduction))
    halftone = max(1, min(100, halftone_artifacts_reduction))
    jpeg = max(1, min(100, jpeg_artifacts_reduction))

    # ENHANCE blend: Node[41] = 1 - enhance/100, Node[160] = enhance/100
    _set_node_param(nodes, 41, "spl::value", 1.0 - enhance / 100.0)
    _set_node_param(nodes, 160, "spl::value", enhance / 100.0)

    # HALFTONE blend: Node[162] = 1 - halftone/100, Node[176] = halftone/100
    _set_node_param(nodes, 162, "spl::value", 1.0 - halftone / 100.0)
    _set_node_param(nodes, 176, "spl::value", halftone / 100.0)

    # JPEG REMOVAL blend: Node[178] = 1 - jpeg/100, Node[182] = jpeg/100
    _set_node_param(nodes, 178, "spl::value", 1.0 - jpeg / 100.0)
    _set_node_param(nodes, 182, "spl::value", jpeg / 100.0)

    # COLOR NOISE blend: Node[219] = col_noise/100 (in 244-node max template)
    _set_node_param(nodes, 219, "spl::value", col_noise / 100.0)

    # SCRATCH DETECTION:
    #   Node[14] threshold = 0.5 - 0.0045 * scratches
    #   Node[18] dilate h/w (approximation: scales with scratches)
    threshold = 0.5 - 0.0045 * scratches
    _set_node_param(nodes, 14, "spl::thresholdValue", threshold)
    dilate_size = max(1.0, round(scratches * 7.0 / 100.0))
    _set_node_param(nodes, 18, "spl::height", dilate_size)
    _set_node_param(nodes, 18, "spl::width", dilate_size)

    # LUMINANCE NOISE:
    #   Node[198] sigma (quadratic): sigma = -0.0992 - 0.0000315*s + 0.0000514*s^2
    #   Node[202] model selection: soft (<20), normal (20-60), strong (>60)
    sigma = -0.09920073 + (-0.0000315009) * lum_noise + 0.00005143508 * lum_noise * lum_noise
    _set_node_param(nodes, 198, "spl::value", sigma)

    if lum_noise <= 20:
        model_id = "denoise_lab_luminance_soft"
    elif lum_noise <= 60:
        model_id = "denoise_lab_luminance"
    else:
        model_id = "denoise_lab_luminance_strong"
    _set_node_param(nodes, 202, "spl::modelID", model_id)
    # Also update the second denoise_luminance pass at Node[211] (max template only)
    if len(nodes) > 211:
        n211_op = nodes[211].get("spl::operation", "")
        if n211_op == "senseiModel":
            _set_node_param(nodes, 211, "spl::modelID", model_id)

    return payload


class PhotoshopPhotoRestorationNode:
    """
    Apply Photo Restoration neural filter to images using Adobe Photoshop API.

    Sliders:
    - Photo enhancement (0-100)
    - Face enhancement (0-100)
    - Scratch reduction (0-100)
    - Noise reduction (0-100)
    - Color noise reduction (0-100)
    - Halftone artifacts reduction (0-100)
    - JPEG artifacts reduction (0-100)
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "action_json", "debug_log")
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
                    "tooltip": "Pre-signed URL or local file path to input image (alternative to tensor)",
                }),

                # Photo Restoration sliders
                "photo_enhancement": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Overall photo enhancement strength (0-100)",
                }),
                "face_enhancement": ("INT", {
                    "default": 60,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Face enhancement strength (0-100)",
                }),
                "scratch_reduction": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Scratch reduction strength (0-100)",
                }),
                "noise_reduction": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Luminance noise reduction strength (0-100)",
                }),
                "color_noise_reduction": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Color noise reduction strength (0-100)",
                }),
                "halftone_artifacts_reduction": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Halftone artifacts reduction strength (0-100)",
                }),
                "jpeg_artifacts_reduction": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "JPEG artifacts reduction strength (0-100)",
                }),

                # Output settings
                "output_type": (
                    ["image/png", "image/jpeg", "image/vnd.adobe.photoshop", "image/tiff"],
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
        photo_enhancement: int = 50,
        face_enhancement: int = 60,
        scratch_reduction: int = 0,
        noise_reduction: int = 0,
        color_noise_reduction: int = 0,
        halftone_artifacts_reduction: int = 0,
        jpeg_artifacts_reduction: int = 0,
        output_type: str = "image/png",
        output_quality: int = 7,
        compression: str = "small",
        unique_id: Optional[str] = None,
    ):
        """Execute Photo Restoration neural filter via Photoshop ActionJSON API."""

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

        # Build the ActionJSON payload with slider values
        action_json_payload = _build_photo_restoration_action_json(
            photo_enhancement=photo_enhancement,
            face_enhancement=face_enhancement,
            scratch_reduction=scratch_reduction,
            noise_reduction=noise_reduction,
            color_noise_reduction=color_noise_reduction,
            halftone_artifacts_reduction=halftone_artifacts_reduction,
            jpeg_artifacts_reduction=jpeg_artifacts_reduction,
        )

        action_json_text = json.dumps(action_json_payload, indent=2)

        console_log = "=" * 55 + "\n"
        console_log += "POST /pie/psdService/actionJSON\n"
        console_log += "  (Photo Restoration Neural Filter)\n"
        console_log += "-" * 55 + "\n"
        console_log += "Slider Values:\n"
        console_log += f"  Photo Enhancement:          {photo_enhancement}\n"
        console_log += f"  Face Enhancement:            {face_enhancement}\n"
        console_log += f"  Scratch Reduction:           {scratch_reduction}\n"
        console_log += f"  Noise Reduction:             {noise_reduction}\n"
        console_log += f"  Color Noise Reduction:       {color_noise_reduction}\n"
        console_log += f"  Halftone Artifacts Reduction: {halftone_artifacts_reduction}\n"
        console_log += f"  JPEG Artifacts Reduction:    {jpeg_artifacts_reduction}\n"
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

            # Build request
            request = ActionJsonRequest(
                inputs=[
                    PhotoshopActionsInput(
                        href=input_url,
                        storage="external"
                    )
                ],
                outputs=[
                    PhotoshopActionsOutput(**output_params)
                ],
                options=ActionJsonOptions(
                    actionJSON=action_json_payload,
                )
            )

            console_log += f"\nRequest submitted (ActionJSON payload: {len(action_json_text)} chars)\n"

            # Submit job
            submit_endpoint = ApiEndpoint(
                path="/pie/psdService/actionJSON",
                method=HttpMethod.POST,
                request_model=ActionJsonRequest,
                response_model=ActionJsonResponse,
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
            console_log += "  Note: Neural filters can take 30-120 seconds\n"

            # Poll for completion
            poll_endpoint = ApiEndpoint(
                path=status_path,
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=ActionJsonJobStatus,
            )

            poll_op = PollingOperation(
                poll_endpoint=poll_endpoint,
                request=EmptyRequest(),
                completed_statuses=["succeeded"],
                failed_statuses=["failed"],
                status_extractor=lambda x: x.status,
                api_base=status_base_url,
                poll_interval=5.0,
                max_poll_attempts=120,
                estimated_duration=60.0,
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
                raise Exception(f"No outputs returned from Photoshop Photo Restoration API")

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

            console_log += f"[OK] Downloaded image ({len(image_bytes) / (1024*1024):.2f} MB)\n"

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

            return (img_tensor, action_json_text, console_log)

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
    "PhotoshopPhotoRestorationNode": PhotoshopPhotoRestorationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoshopPhotoRestorationNode": "Photo Restoration",
}
