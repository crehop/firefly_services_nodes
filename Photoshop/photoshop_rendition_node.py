"""
Adobe Photoshop Rendition Create Node

Create flat image renditions from PSD files using Adobe Photoshop API.
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
from urllib.parse import urlparse
from PIL import Image

from .photoshop_api import (
    PhotoshopJobStatusEnum,
    PhotoshopActionsInput,
    LayerReference,
    RenditionOutput,
    RenditionCreateRequest,
    RenditionCreateResponse,
    RenditionCreateJobStatus,
)
from .photoshop_storage import upload_file_to_s3, generate_output_presigned_url, generate_download_url
from ..Firefly.firefly_storage import create_adobe_client
from ..client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)


class PhotoshopRenditionCreateNode:
    """
    Create flat image renditions from PSD files using Adobe Photoshop API.

    Features:
    - Export full document or specific layers as images
    - Support for PNG, JPEG, TIFF, and PSD output formats
    - Control width, quality, and compression
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
            "required": {
                "psd_input": ("STRING", {
                    "default": "",
                    "tooltip": "PSD input: local file path or URL",
                }),
            },
            "optional": {
                "output_type": (
                    ["image/png", "image/jpeg", "image/tiff", "image/vnd.adobe.photoshop"],
                    {
                        "default": "image/png",
                        "tooltip": "Output file format",
                    }
                ),
                "width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 32000,
                    "tooltip": "Output width in pixels (0 for full size)",
                }),
                "quality": ("INT", {
                    "default": 7,
                    "min": 1,
                    "max": 7,
                    "tooltip": "JPEG quality level (1-7, only for JPEG output)",
                }),
                "compression": (["small", "medium", "large"], {
                    "default": "small",
                    "tooltip": "PNG compression level",
                }),
                "trim_to_canvas": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Trim output to canvas bounds",
                }),
                "layers_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional: JSON array of layer references [{\"id\": 1} or {\"name\": \"Layer\"}]. Leave empty for full document.",
                }),
            },
        }

    def _build_debug_log(
        self,
        psd_input: str,
        is_local_file: bool,
        output_type: str,
        width: int,
        quality: int,
        compression: str,
        trim_to_canvas: bool,
        layers_json: str,
    ) -> str:
        """Build formatted debug log showing request details."""
        log = "=" * 55 + "\n"
        log += "POST /pie/psdService/renditionCreate\n"
        log += "-" * 55 + "\n"
        log += "Headers:\n"
        log += "  Authorization: Bearer [TOKEN]\n"
        log += "  x-api-key: [API_KEY]\n"
        log += "  Content-Type: application/json\n"
        log += "\nRequest Summary:\n"

        # Input
        log += "  inputs:\n"
        if is_local_file:
            log += f"    - href: [S3_PRESIGNED_URL] (from local file: {psd_input})\n"
        else:
            log += f"    - href: {psd_input[:80]}{'...' if len(psd_input) > 80 else ''}\n"
        log += "      storage: external\n"

        # Output settings
        log += f"  output_type: {output_type}\n"
        log += f"  width: {width} (0 = full size)\n"
        if output_type == "image/jpeg":
            log += f"  quality: {quality}\n"
        if output_type == "image/png":
            log += f"  compression: {compression}\n"
        log += f"  trim_to_canvas: {trim_to_canvas}\n"

        # Layers
        if layers_json:
            log += "\nLayers JSON:\n"
            log += f"{layers_json}\n"
        else:
            log += "\nLayers: Full document (no specific layers)\n"

        return log

    async def api_call(
        self,
        psd_input: str = "",
        output_type: str = "image/png",
        width: int = 0,
        quality: int = 7,
        compression: str = "small",
        trim_to_canvas: bool = False,
        layers_json: str = "",
    ):
        """Create rendition from PSD file using Photoshop API."""

        # Validate inputs
        if not psd_input:
            raise ValueError("Must provide 'psd_input' (local file path or URL)")

        # Parse layers JSON if provided
        layers_list = None
        if layers_json and layers_json.strip():
            try:
                layers_list = json.loads(layers_json)
                if not isinstance(layers_list, list):
                    raise ValueError("layers_json must be a JSON array")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in layers_json: {e}")

        # Detect if input is URL or local file path
        is_url = psd_input.startswith(("http://", "https://"))
        is_local_file = not is_url

        # Validate file exists if local path
        if is_local_file and not os.path.exists(psd_input):
            raise ValueError(f"PSD file not found at path: {psd_input}")

        # Build initial debug log
        console_log = self._build_debug_log(
            psd_input=psd_input,
            is_local_file=is_local_file,
            output_type=output_type,
            width=width,
            quality=quality,
            compression=compression,
            trim_to_canvas=trim_to_canvas,
            layers_json=layers_json,
        )

        print(f"\n[DEBUG] ===== RENDITION CREATE NODE START =====")
        print(f"[DEBUG] psd_input: {psd_input}")
        print(f"[DEBUG] output_type: {output_type}")
        print(f"[DEBUG] width: {width}")

        client = await create_adobe_client()

        try:
            # Get input URL
            if is_local_file:
                console_log += f"\n{'='*55}\n"
                console_log += "Uploading PSD file to S3...\n"

                # Get file size
                file_size = os.path.getsize(psd_input)
                console_log += f"  File path: {psd_input}\n"
                console_log += f"  File size: {file_size / (1024*1024):.2f} MB\n"

                print(f"[DEBUG] Uploading local file: {psd_input} ({file_size} bytes)")

                # Upload and measure time
                upload_start = time.time()
                input_url = await upload_file_to_s3(psd_input, content_type="image/vnd.adobe.photoshop")
                upload_duration = time.time() - upload_start

                console_log += f"[OK] Upload complete ({upload_duration:.2f}s)\n"
                console_log += f"  Pre-signed URL generated (valid 24h)\n"

                print(f"[DEBUG] Upload complete in {upload_duration:.2f}s")
                print(f"[DEBUG] Input URL: {input_url[:100]}...")
            else:
                input_url = psd_input
                console_log += f"\n{'='*55}\n"
                console_log += "Using provided PSD URL\n"
                console_log += f"  URL: {input_url[:80]}{'...' if len(input_url) > 80 else ''}\n"

                print(f"[DEBUG] Using provided URL: {input_url[:100]}...")

            # Generate output pre-signed URL
            console_log += f"\n{'='*55}\n"
            console_log += "Generating output pre-signed PUT URL...\n"

            # Determine file extension from output_type
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
            console_log += f"  Output filename: {output_filename}\n"

            print(f"[DEBUG] Output URL generated: {output_url_presigned[:100]}...")
            print(f"[DEBUG] Output filename: {output_filename}")

            # Build layer references if provided
            layer_refs = None
            if layers_list:
                layer_refs = []
                for layer in layers_list:
                    layer_refs.append(LayerReference(
                        id=layer.get("id"),
                        name=layer.get("name"),
                    ))

            # Build output specification
            output_spec = RenditionOutput(
                href=output_url_presigned,
                storage="external",
                type=output_type,
                overwrite=True,
                width=width if width > 0 else None,
                trimToCanvas=trim_to_canvas if trim_to_canvas else None,
                layers=layer_refs,
            )

            # Add format-specific options
            if output_type == "image/jpeg":
                output_spec.quality = quality
            elif output_type == "image/png":
                output_spec.compression = compression

            # Build request
            request = RenditionCreateRequest(
                inputs=[
                    PhotoshopActionsInput(
                        href=input_url,
                        storage="external"
                    )
                ],
                outputs=[output_spec]
            )

            # Log the actual request JSON being sent
            request_json_str = json.dumps(request.model_dump(exclude_none=True), indent=2)
            console_log += f"\n{'='*55}\n"
            console_log += "Request JSON being sent to API:\n"
            console_log += f"{request_json_str}\n"

            print(f"[DEBUG] Request JSON:\n{request_json_str}")

            # Submit job
            print(f"[DEBUG] Submitting job to /pie/psdService/renditionCreate...")

            submit_endpoint = ApiEndpoint(
                path="/pie/psdService/renditionCreate",
                method=HttpMethod.POST,
                request_model=RenditionCreateRequest,
                response_model=RenditionCreateResponse,
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

            print(f"[DEBUG] Submit response received")
            print(f"[DEBUG]   jobId: {job_id}")
            print(f"[DEBUG]   statusUrl: {status_url}")

            # Log raw response for debugging
            submit_response_dict = submit_response.model_dump()
            console_log += f"\nRaw Submit Response:\n"
            console_log += f"{json.dumps(submit_response_dict, indent=2)}\n"

            print(f"[DEBUG] Raw submit response: {json.dumps(submit_response_dict, indent=2)}")

            # Parse statusUrl to get correct polling endpoint
            parsed_status_url = urlparse(status_url)
            status_base_url = f"{parsed_status_url.scheme}://{parsed_status_url.netloc}"
            status_path = parsed_status_url.path

            print(f"[DEBUG] Parsed status URL:")
            print(f"[DEBUG]   base: {status_base_url}")
            print(f"[DEBUG]   path: {status_path}")

            # Log polling start
            console_log += f"\n{'='*55}\n"
            console_log += f"GET {status_path}\n"
            console_log += f"{'-'*55}\n"
            console_log += "Polling for job completion...\n"
            console_log += f"  Status URL: {status_url}\n"
            console_log += f"  Interval: {2.0}s\n"
            console_log += f"  Max attempts: {300} (timeout: {300 * 2.0}s = 10 min)\n"

            # Poll for completion
            poll_endpoint = ApiEndpoint(
                path=status_path,
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=RenditionCreateJobStatus,
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
            )

            # Track polling time
            poll_start_time = time.time()
            console_log += f"\nStarting polling at {time.strftime('%H:%M:%S', time.localtime(poll_start_time))}...\n"

            print(f"[DEBUG] Starting polling at {time.strftime('%H:%M:%S', time.localtime(poll_start_time))}")

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
            console_log += f"  jobId: {result.jobId}\n"

            print(f"[DEBUG] Polling completed in {poll_duration:.1f}s")
            print(f"[DEBUG]   status: {result.status}")
            print(f"[DEBUG]   jobId: {result.jobId}")

            # Log raw polling response for debugging
            console_log += f"\nRaw Poll Response:\n"
            try:
                result_dict = result.model_dump()
                console_log += f"{json.dumps(result_dict, indent=2)}\n"
                print(f"[DEBUG] Raw poll response: {json.dumps(result_dict, indent=2)}")
            except Exception as e:
                console_log += f"ERROR dumping result: {e}\n"
                console_log += f"result type: {type(result)}\n"
                console_log += f"result: {result}\n"
                print(f"[DEBUG] ERROR dumping result: {e}")

            # Check for errors in outputs
            if result.outputs and len(result.outputs) > 0:
                first_output = result.outputs[0]
                if first_output.errors:
                    console_log += f"\n{'='*55}\n"
                    console_log += "API ERRORS:\n"
                    console_log += f"{json.dumps(first_output.errors, indent=2) if isinstance(first_output.errors, dict) else json.dumps(first_output.errors, indent=2)}\n"
                    console_log += f"{'='*55}\n"

                    print(f"[DEBUG] API returned errors: {first_output.errors}")

                    error_msg = first_output.errors.get('title', 'Unknown error') if isinstance(first_output.errors, dict) else str(first_output.errors)
                    raise Exception(f"Rendition create failed: {error_msg}")

            # Validate outputs
            if not result.result or not result.result.outputs:
                console_log += f"\n{'='*55}\n"
                console_log += "ERROR: No outputs in response\n"
                console_log += f"  status: {result.status}\n"
                console_log += f"  jobId: {result.jobId}\n"

                print(f"[DEBUG] No outputs in response")
                print(console_log)

                raise Exception("No outputs returned from Rendition Create API.")

            # Generate GET URL for downloading the result
            console_log += f"\n{'='*55}\n"
            console_log += "Generating download URL...\n"

            print(f"[DEBUG] Generating download URL for: {output_filename}")

            download_url = await generate_download_url(output_filename)
            console_log += f"[OK] Generated download URL\n"
            console_log += f"  URL: {download_url[:80]}...\n"

            print(f"[DEBUG] Download URL: {download_url[:100]}...")

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

            # Convert to RGB if needed
            if img_pil.mode not in ['RGB', 'RGBA']:
                img_pil = img_pil.convert('RGB')

            img_array = np.array(img_pil).astype(np.float32) / 255.0

            # Ensure 3 channels (RGB)
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)

            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            console_log += f"\nOutput URL (valid for 24 hours):\n"
            console_log += f"  {download_url}\n"
            console_log += f"{'='*55}\n"

            print(f"[DEBUG] ===== RENDITION CREATE NODE COMPLETE =====\n")

            return (img_tensor, download_url, console_log)

        except Exception as e:
            console_log += f"\n{'='*55}\n"
            console_log += f"ERROR: {str(e)}\n"
            console_log += f"{'='*55}\n"

            print(f"[DEBUG] ===== ERROR =====")
            print(f"[DEBUG] {str(e)}")
            print(console_log)

            raise
        finally:
            try:
                await client.close()
            except:
                pass


# Node registration
NODE_CLASS_MAPPINGS = {
    "PhotoshopRenditionCreateNode": PhotoshopRenditionCreateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoshopRenditionCreateNode": "Photoshop Rendition Create",
}
