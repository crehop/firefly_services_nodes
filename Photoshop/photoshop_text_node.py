"""
Adobe Photoshop Text Edit Node

Edit text layers in PSD files using Adobe Photoshop API.
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
    PhotoshopActionsOutput,
    PhotoshopAsset,
    TextOptions,
    TextOptionsLayer,
    TextOptionsLayerText,
    TextEditRequest,
    TextEditResponse,
    TextEditJobStatus,
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


class PhotoshopTextEditNode:
    """
    Edit text layers in PSD files using Adobe Photoshop API.

    Features:
    - Edit text content in existing text layers
    - Change text styling (font, size, color)
    - Support for character and paragraph styles
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
                "layers_json": ("STRING", {
                    "default": """[
  {
    "name": "Text Layer",
    "text": {
      "content": "New Text Content"
    }
  }
]""",
                    "multiline": True,
                    "tooltip": "JSON array of text layer edits. Each object needs 'id' or 'name' and 'text' with content/styles.",
                }),
            },
            "optional": {
                "output_type": (
                    ["image/vnd.adobe.photoshop", "image/png", "image/jpeg", "image/tiff"],
                    {
                        "default": "image/vnd.adobe.photoshop",
                        "tooltip": "Output file format",
                    }
                ),
                "manage_missing_fonts": (["useDefault", "fail"], {
                    "default": "useDefault",
                    "tooltip": "Action for missing fonts",
                }),
                "global_font": ("STRING", {
                    "default": "",
                    "tooltip": "PostScript name of global default font",
                }),
            },
        }

    def _build_debug_log(
        self,
        psd_input: str,
        is_local_file: bool,
        layers_json: str,
        output_type: str,
        manage_missing_fonts: str,
        global_font: str,
    ) -> str:
        """Build formatted debug log showing request details."""
        log = "=" * 55 + "\n"
        log += "POST /pie/psdService/text\n"
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

        # Output
        log += f"  output_type: {output_type}\n"
        log += f"  manage_missing_fonts: {manage_missing_fonts}\n"
        if global_font:
            log += f"  global_font: {global_font}\n"

        # Layers preview
        log += "\nLayers JSON:\n"
        log += f"{layers_json}\n"

        return log

    async def api_call(
        self,
        psd_input: str = "",
        layers_json: str = "",
        output_type: str = "image/vnd.adobe.photoshop",
        manage_missing_fonts: str = "useDefault",
        global_font: str = "",
    ):
        """Edit text layers in PSD file using Photoshop API."""

        # Validate inputs
        if not psd_input:
            raise ValueError("Must provide 'psd_input' (local file path or URL)")
        if not layers_json:
            raise ValueError("Must provide 'layers_json' with text layer edits")

        # Parse layers JSON
        try:
            layers_list = json.loads(layers_json)
            if not isinstance(layers_list, list):
                raise ValueError("layers_json must be a JSON array")
            if len(layers_list) == 0:
                raise ValueError("layers_json must contain at least one layer")
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
            layers_json=layers_json,
            output_type=output_type,
            manage_missing_fonts=manage_missing_fonts,
            global_font=global_font,
        )

        print(f"\n[DEBUG] ===== TEXT EDIT NODE START =====")
        print(f"[DEBUG] psd_input: {psd_input}")
        print(f"[DEBUG] layers_json: {layers_json}")
        print(f"[DEBUG] output_type: {output_type}")

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
            file_extension = extension_map.get(output_type, "psd")

            url_gen_start = time.time()
            output_url_presigned, output_filename = await generate_output_presigned_url(file_extension=file_extension)
            url_gen_duration = time.time() - url_gen_start

            console_log += f"[OK] Generated pre-signed PUT URL ({url_gen_duration:.2f}s)\n"
            console_log += f"  Output filename: {output_filename}\n"

            print(f"[DEBUG] Output URL generated: {output_url_presigned[:100]}...")
            print(f"[DEBUG] Output filename: {output_filename}")

            # Build text options from layers_json
            text_layers = []
            for layer in layers_list:
                text_layer = TextOptionsLayer(
                    id=layer.get("id"),
                    name=layer.get("name"),
                    locked=layer.get("locked"),
                    visible=layer.get("visible"),
                )
                if "text" in layer:
                    text_layer.text = TextOptionsLayerText(
                        content=layer["text"].get("content"),
                        orientation=layer["text"].get("orientation"),
                        antiAlias=layer["text"].get("antiAlias"),
                        characterStyles=layer["text"].get("characterStyles"),
                        paragraphStyles=layer["text"].get("paragraphStyles"),
                    )
                text_layers.append(text_layer)

            # Build request
            request = TextEditRequest(
                inputs=[
                    PhotoshopActionsInput(
                        href=input_url,
                        storage="external"
                    )
                ],
                outputs=[
                    PhotoshopActionsOutput(
                        href=output_url_presigned,
                        storage="external",
                        type=output_type,
                        overwrite=True
                    )
                ],
                options=TextOptions(
                    layers=text_layers,
                    manageMissingFonts=manage_missing_fonts if manage_missing_fonts else None,
                    globalFont=global_font if global_font else None,
                )
            )

            # Log the actual request JSON being sent
            request_json_str = json.dumps(request.model_dump(exclude_none=True), indent=2)
            console_log += f"\n{'='*55}\n"
            console_log += "Request JSON being sent to API:\n"
            console_log += f"{request_json_str}\n"

            print(f"[DEBUG] Request JSON:\n{request_json_str}")

            # Submit job
            print(f"[DEBUG] Submitting job to /pie/psdService/text...")

            submit_endpoint = ApiEndpoint(
                path="/pie/psdService/text",
                method=HttpMethod.POST,
                request_model=TextEditRequest,
                response_model=TextEditResponse,
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
                response_model=TextEditJobStatus,
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

                    # Return the error info instead of raising
                    error_msg = first_output.errors.get('title', 'Unknown error') if isinstance(first_output.errors, dict) else str(first_output.errors)
                    raise Exception(f"Text edit failed: {error_msg}")

            # Validate outputs
            if not result.result or not result.result.outputs:
                console_log += f"\n{'='*55}\n"
                console_log += "ERROR: No outputs in response\n"
                console_log += f"  status: {result.status}\n"
                console_log += f"  jobId: {result.jobId}\n"

                print(f"[DEBUG] No outputs in response")
                print(console_log)

                raise Exception("No outputs returned from Text Edit API.")

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

            print(f"[DEBUG] ===== TEXT EDIT NODE COMPLETE =====\n")

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
    "PhotoshopTextEditNode": PhotoshopTextEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoshopTextEditNode": "Photoshop Text Edit",
}
