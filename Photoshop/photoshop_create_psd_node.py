"""
Adobe Photoshop Create PSD Node (Full Schema)

Create PSD files using full Adobe Photoshop API schema.
"""

from __future__ import annotations
from typing import Optional
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
    DocumentCreateRequest,
    DocumentCreateResponse,
    DocumentCreateJobStatus,
)
from .photoshop_storage import generate_output_presigned_url, generate_download_url
from ..Firefly.firefly_storage import create_adobe_client
from ..client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)


class PhotoshopCreatePSDNode:
    """
    Create PSD files using full Adobe Photoshop API schema.

    Features:
    - Accept complete JSON schema for maximum flexibility
    - All configuration in JSON (document settings, layers, etc.)
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
                # Full schema JSON input
                "schema_json": ("STRING", {
                    "default": """{
  "options": {
    "document": {
      "width": 1000,
      "height": 1000,
      "resolution": 72,
      "mode": "rgb",
      "fill": "transparent",
      "depth": 8
    },
    "layers": [
      {
        "name": "Layer 1",
        "type": "layer",
        "input": {
          "href": "https://your-s3-url-here.com/image1.jpg",
          "storage": "external"
        }
      }
    ]
  },
  "outputs": [
    {
      "href": "[OUTPUT_URL_WILL_BE_GENERATED]",
      "storage": "external",
      "type": "image/vnd.adobe.photoshop"
    }
  ]
}""",
                    "multiline": True,
                    "tooltip": "Complete JSON schema for Document Create API. Output href will be auto-generated.",
                }),
            },
            # "hidden": {
            #     "unique_id": "UNIQUE_ID",  # Disabled - no task tracking
            # },
        }

    def _build_debug_log(self, schema_json_str: str) -> str:
        """Build formatted debug log showing request details."""
        log = "=" * 55 + "\n"
        log += "POST /pie/psdService/documentCreate\n"
        log += "-" * 55 + "\n"
        log += "Headers:\n"
        log += "  Authorization: Bearer [TOKEN]\n"
        log += "  x-api-key: [API_KEY]\n"
        log += "  Content-Type: application/json\n"
        log += "\nRequest Body (from schema_json):\n"
        log += f"{schema_json_str}\n"

        return log

    async def api_call(
        self,
        schema_json: str = "",
        # unique_id: Optional[str] = None,  # Disabled - no task tracking
    ):
        """Create PSD file using full schema JSON and Photoshop API."""

        # Validate JSON input
        if not schema_json or schema_json.strip() == "":
            raise ValueError("Must provide schema_json input")

        # Parse JSON
        try:
            schema_dict = json.loads(schema_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema_json: {e}")

        # Validate required fields
        if "options" not in schema_dict:
            raise ValueError("schema_json must include 'options' field")
        if "outputs" not in schema_dict:
            raise ValueError("schema_json must include 'outputs' field")
        if not isinstance(schema_dict["outputs"], list) or len(schema_dict["outputs"]) == 0:
            raise ValueError("schema_json 'outputs' must be a non-empty array")

        # Extract output type to determine if we need psd-tools
        output_type = schema_dict["outputs"][0].get("type", "image/vnd.adobe.photoshop")

        # Check if psd-tools is available when PSD output is requested
        if output_type == "image/vnd.adobe.photoshop" and not PSD_TOOLS_AVAILABLE:
            raise ImportError(
                "psd-tools library is required for PSD output but is not installed.\n"
                "Please install it with: pip install psd-tools"
            )

        # Build initial debug log
        console_log = self._build_debug_log(schema_json_str=schema_json)

        client = await create_adobe_client()

        try:
            # Generate output URL if not provided or is placeholder
            output_href = schema_dict["outputs"][0].get("href", "")
            if not output_href or "[OUTPUT_URL" in output_href or "your-s3-url" in output_href:
                console_log += f"\n{'='*55}\n"
                console_log += "Generating output pre-signed PUT URL for S3...\n"

                # Determine file extension from output_type
                extension_map = {
                    "image/vnd.adobe.photoshop": "psd",
                    "image/jpeg": "jpg",
                    "image/png": "png",
                    "image/tiff": "tiff",
                }
                file_extension = extension_map.get(output_type, "psd")

                # Generate pre-signed PUT URL for Adobe to upload the result
                url_gen_start = time.time()
                output_url_presigned, output_filename = await generate_output_presigned_url(file_extension=file_extension)
                url_gen_duration = time.time() - url_gen_start

                console_log += f"[OK] Generated pre-signed PUT URL ({url_gen_duration:.2f}s)\n"
                console_log += f"  URL: {output_url_presigned[:80]}...\n"
                console_log += f"  Adobe will upload result to this URL\n"
                console_log += f"  Output filename: {output_filename}\n"

                # Update schema with generated URL
                schema_dict["outputs"][0]["href"] = output_url_presigned
            else:
                # Use provided URL
                console_log += f"\n{'='*55}\n"
                console_log += "Using provided output URL from schema\n"
                # Extract filename from URL for later download
                output_filename = output_href.split("?")[0].split("/")[-1]
                console_log += f"  Output filename: {output_filename}\n"

            # Log the actual request JSON being sent
            console_log += f"\nFinal Request JSON being sent to API:\n"
            console_log += f"{json.dumps(schema_dict, indent=2)}\n"

            # Validate and create request using Pydantic
            try:
                request = DocumentCreateRequest(**schema_dict)
            except Exception as e:
                raise ValueError(f"Invalid schema: {e}")

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
                    pil_image = psd.composite()

                    console_log += "[OK] Rendered PSD composite\n"
                    console_log += f"  PSD size: {psd.width}x{psd.height}\n"
                    console_log += f"  Layer count: {len(list(psd))}\n"

                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_psd_path)
                    except:
                        pass

                # Convert PIL Image to tensor, handling alpha channel properly
                if pil_image.mode == 'RGBA':
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
                # Download JPEG/PNG/TIFF directly
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
    "PhotoshopCreatePSDNode": PhotoshopCreatePSDNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoshopCreatePSDNode": "Create PSD",
}
