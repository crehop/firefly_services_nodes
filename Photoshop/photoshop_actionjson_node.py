"""
Adobe Photoshop ActionJSON Node

Execute Photoshop actions defined in JSON format using Adobe Photoshop API.
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
    PhotoshopAsset,
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
# from comfy.utils import ProgressBar  # Not used - progress bar disabled


class PhotoshopActionJsonNode:
    """
    Execute Photoshop actions defined in JSON format using Adobe Photoshop API.

    Features:
    - Execute actions defined in JSON format (no .atn file needed)
    - Support for patterns, fonts, brushes, and additional images
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

                # ActionJSON
                "action_json": ("STRING", {
                    "default": """[
  {
    "_obj": "invert",
    "_isReference": true
  }
]""",
                    "multiline": True,
                    "tooltip": "JSON array of Photoshop actions to execute. Default: Invert colors",
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

                # Optional assets
                "pattern_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "URL to pattern file (.pat) - optional",
                }),
                "font_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "URL to font file - optional",
                }),
                "brush_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "URL to brush file (.abr) - optional",
                }),
                "additional_image_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "URL to additional image for actionJSON commands - optional",
                }),
            },
            # "hidden": {
            #     "unique_id": "UNIQUE_ID",  # Disabled - no task tracking
            # },
        }

    def _build_debug_log(
        self,
        action_json_str: str,
        output_type: str,
        output_quality: int,
        compression: str,
        image: Optional[torch.Tensor],
        image_reference: str,
        pattern_url: str,
        font_url: str,
        brush_url: str,
        additional_image_url: str,
    ) -> str:
        """Build formatted debug log showing request details."""
        log = "=" * 55 + "\n"
        log += "POST /pie/psdService/actionJSON\n"
        log += "-" * 55 + "\n"
        log += "Headers:\n"
        log += "  Authorization: Bearer [TOKEN]\n"
        log += "  x-api-key: [API_KEY]\n"
        log += "  Content-Type: application/json\n"
        log += "\nRequest Body:\n"

        # Input
        log += "  inputs:\n"
        if image is not None:
            log += "    - href: [S3_PRESIGNED_URL]\n"
        else:
            log += f"    - href: {image_reference}\n"
        log += "      storage: external\n"

        # Output
        log += "  outputs:\n"
        log += "    - href: [S3_PRESIGNED_URL]\n"
        log += "      storage: external\n"
        log += f"      type: {output_type}\n"
        log += "      overwrite: true\n"
        log += f"      quality: {output_quality}\n"
        log += f"      compression: {compression}\n"

        # Options
        log += "  options:\n"
        log += "    actionJSON:\n"
        # Parse and show action JSON
        try:
            actions = json.loads(action_json_str) if action_json_str else []
            log += f"      {json.dumps(actions, indent=6)}\n"
        except json.JSONDecodeError as e:
            log += f"      [ERROR parsing JSON: {e}]\n"
            log += f"      Raw: {action_json_str[:200]}...\n"

        if pattern_url:
            log += "    patterns:\n"
            log += f"      - href: {pattern_url}\n"
            log += "        storage: external\n"

        if font_url:
            log += "    fonts:\n"
            log += f"      - href: {font_url}\n"
            log += "        storage: external\n"

        if brush_url:
            log += "    brushes:\n"
            log += f"      - href: {brush_url}\n"
            log += "        storage: external\n"

        if additional_image_url:
            log += "    additionalImages:\n"
            log += f"      - href: {additional_image_url}\n"
            log += "        storage: external\n"

        return log

    async def api_call(
        self,
        image: Optional[torch.Tensor] = None,
        image_reference: str = "",
        action_json: str = "[]",
        output_type: str = "image/png",
        output_quality: int = 7,
        compression: str = "small",
        pattern_url: str = "",
        font_url: str = "",
        brush_url: str = "",
        additional_image_url: str = "",
        # unique_id: Optional[str] = None,  # Disabled - no task tracking
    ):
        """Execute Photoshop actionJSON on image using Photoshop API."""

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

        # Parse actionJSON
        try:
            action_json_parsed = json.loads(action_json) if action_json else []
            if not isinstance(action_json_parsed, list):
                raise ValueError("action_json must be a JSON array")
            if len(action_json_parsed) == 0:
                raise ValueError("action_json must contain at least one action")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in action_json: {e}")

        # Build initial debug log
        console_log = self._build_debug_log(
            action_json_str=action_json,
            output_type=output_type,
            output_quality=output_quality,
            compression=compression,
            image=image,
            image_reference=image_reference,
            pattern_url=pattern_url,
            font_url=font_url,
            brush_url=brush_url,
            additional_image_url=additional_image_url,
        )

        client = await create_adobe_client()

        try:
            # Get input URL
            if image is not None:
                console_log += f"\n{'='*55}\n"
                console_log += "Uploading input image to S3...\n"

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

            # Create output URL (S3 pre-signed PUT URL)
            console_log += f"\n{'='*55}\n"
            console_log += "Generating output pre-signed PUT URL for S3...\n"

            # Determine file extension from output_type
            # e.g., "image/vnd.adobe.photoshop" -> "psd", "image/png" -> "png"
            extension_map = {
                "image/vnd.adobe.photoshop": "psd",
                "vnd.adobe.photoshop": "psd",
                "image/jpeg": "jpg",
                "image/png": "png",
                "image/tiff": "tiff",
            }
            file_extension = extension_map.get(output_type, "png")

            # Generate pre-signed PUT URL for Adobe to upload the result
            url_gen_start = time.time()
            output_url_presigned, output_filename = await generate_output_presigned_url(file_extension=file_extension)
            url_gen_duration = time.time() - url_gen_start

            console_log += f"[OK] Generated pre-signed PUT URL ({url_gen_duration:.2f}s)\n"
            console_log += f"  URL: {output_url_presigned[:80]}...\n"
            console_log += f"  Adobe will upload result to this URL\n"
            console_log += f"  Output filename: {output_filename}\n"

            # Build optional assets
            patterns = None
            if pattern_url:
                patterns = [PhotoshopAsset(href=pattern_url, storage="external")]

            fonts = None
            if font_url:
                fonts = [PhotoshopAsset(href=font_url, storage="external")]

            brushes = None
            if brush_url:
                brushes = [PhotoshopAsset(href=brush_url, storage="external")]

            additional_images = None
            if additional_image_url:
                additional_images = [PhotoshopAsset(href=additional_image_url, storage="external")]

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
                    actionJSON=action_json_parsed,
                    patterns=patterns,
                    fonts=fonts,
                    brushes=brushes,
                    additionalImages=additional_images,
                )
            )

            # Log the actual request JSON being sent
            console_log += f"\nRequest JSON being sent to API:\n"
            console_log += f"{json.dumps(request.model_dump(exclude_none=True), indent=2)}\n"

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
                console_log += f"Response _links: {submit_response._links}\n"
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
            console_log += f"  Max attempts: {300} (timeout: {300 * 2.0}s = 10 min)\n"
            console_log += "  Note: ActionJSON can take several minutes\n"

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
                poll_interval=2.0,
                max_poll_attempts=300,  # 10 min timeout
                estimated_duration=30.0,  # Estimated 30s for ActionJSON processing
                # node_id=unique_id,  # Disabled - no progress bar
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
                console_log += f"result type: {type(result)}\n"
                console_log += f"result: {result}\n"

            # Validate outputs
            if not result.result or not result.result.outputs:
                console_log += f"\n{'='*55}\n"
                console_log += "ERROR: No outputs in response\n"
                console_log += f"  status: {result.status}\n"
                console_log += f"  result exists: {result.result is not None}\n"
                if result.result:
                    console_log += f"  outputs exists: {hasattr(result.result, 'outputs')}\n"
                    if hasattr(result.result, 'outputs'):
                        console_log += f"  outputs value: {result.result.outputs}\n"
                console_log += f"\n**CONNECT debug_log OUTPUT TO ShowText TO SEE FULL RESPONSE**\n"
                print(console_log)  # Print to console for debugging
                raise Exception(f"No outputs returned from Photoshop ActionJSON API. Check console logs or connect debug_log output for details.")

            # Generate GET URL for downloading the result
            console_log += f"\n{'='*55}\n"
            console_log += "Generating download URL...\n"

            download_url = await generate_download_url(output_filename)
            console_log += f"[OK] Generated download URL\n"
            console_log += f"  URL: {download_url[:80]}...\n"

            # Download result
            console_log += "\nDownloading output image...\n"

            # Download image
            async with aiohttp.ClientSession() as session:
                async with session.get(download_url) as resp:
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
            console_log += f"  {download_url}\n"
            console_log += f"{'='*55}\n"

            return (img_tensor, download_url, console_log)

        except Exception as e:
            console_log += f"\n{'='*55}\n"
            console_log += f"ERROR: {str(e)}\n"
            console_log += f"{'='*55}\n"
            try:
                print(console_log)  # Print to console even on error
            except OSError:
                pass  # Ignore Windows stdout flush errors

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


# Node registration
NODE_CLASS_MAPPINGS = {
    "PhotoshopActionJsonNode": PhotoshopActionJsonNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoshopActionJsonNode": "Photoshop ActionJSON",
}
