"""
Adobe Photoshop PSD Manifest Node

Extract document manifest (metadata, layer info) from PSD files using Adobe Photoshop API.
"""

from __future__ import annotations
from typing import Optional
import time
import json
from urllib.parse import urlparse
import os

from .photoshop_api import (
    DocumentManifestInput,
    DocumentManifestOptions,
    DocumentManifestRequest,
    DocumentManifestResponse,
    DocumentManifestJobStatus,
    ThumbnailOptions,
)
from .photoshop_storage import upload_file_to_s3
from ..Firefly.firefly_storage import create_adobe_client
from ..client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)


class PhotoshopPsdManifestNode:
    """
    Extract document manifest from PSD files using Adobe Photoshop API.

    Features:
    - Extract layer tree, document info, and metadata from PSD files
    - Support for PSD file input (local path or URL)
    - Optional thumbnail generation for layers
    - Returns JSON manifest with full layer structure
    - Comprehensive debug logging
    - Async polling for results
    """

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("manifest_json", "debug_log")
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
                "include_thumbnails": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include thumbnail URLs for renderable layers",
                }),
                "thumbnail_type": (["image/png", "image/jpeg"], {
                    "default": "image/png",
                    "tooltip": "Thumbnail format (only used if include_thumbnails is True)",
                }),
            },
        }

    def _build_debug_log(
        self,
        psd_input: str,
        is_local_file: bool,
        include_thumbnails: bool,
        thumbnail_type: str,
    ) -> str:
        """Build formatted debug log showing request details."""
        log = "=" * 55 + "\n"
        log += "POST /pie/psdService/documentManifest\n"
        log += "-" * 55 + "\n"
        log += "Headers:\n"
        log += "  Authorization: Bearer [TOKEN]\n"
        log += "  x-api-key: [API_KEY]\n"
        log += "  Content-Type: application/json\n"
        log += "\nRequest Body:\n"

        # Input
        log += "  inputs:\n"
        if is_local_file:
            log += f"    - href: [S3_PRESIGNED_URL] (from local file: {psd_input})\n"
        else:
            log += f"    - href: {psd_input}\n"
        log += "      storage: external\n"

        # Options
        if include_thumbnails:
            log += "  options:\n"
            log += "    thumbnails:\n"
            log += f"      type: {thumbnail_type}\n"

        return log

    async def api_call(
        self,
        psd_input: str = "",
        include_thumbnails: bool = False,
        thumbnail_type: str = "image/png",
    ):
        """Extract document manifest from PSD file using Photoshop API."""

        # Validate inputs
        if not psd_input:
            raise ValueError("Must provide 'psd_input' (local file path or URL)")

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
            include_thumbnails=include_thumbnails,
            thumbnail_type=thumbnail_type,
        )

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

                # Upload and measure time
                upload_start = time.time()
                input_url = await upload_file_to_s3(psd_input, content_type="image/vnd.adobe.photoshop")
                upload_duration = time.time() - upload_start

                console_log += f"[OK] Upload complete ({upload_duration:.2f}s)\n"
                console_log += f"  Pre-signed URL generated (valid 24h)\n"
                console_log += f"  URL: {input_url[:100]}...\n"
            else:
                input_url = psd_input
                console_log += f"\n{'='*55}\n"
                console_log += "Using provided PSD URL\n"
                console_log += f"  URL: {input_url[:80]}{'...' if len(input_url) > 80 else ''}\n"

            # Build options
            options = None
            if include_thumbnails:
                options = DocumentManifestOptions(
                    thumbnails=ThumbnailOptions(type=thumbnail_type)
                )

            # Build request
            request = DocumentManifestRequest(
                inputs=[
                    DocumentManifestInput(
                        href=input_url,
                        storage="external"
                    )
                ],
                options=options,
            )

            # Log the actual request JSON being sent
            console_log += f"\nRequest JSON being sent to API:\n"
            console_log += f"{json.dumps(request.model_dump(exclude_none=True), indent=2)}\n"

            # Submit job
            submit_endpoint = ApiEndpoint(
                path="/pie/psdService/documentManifest",
                method=HttpMethod.POST,
                request_model=DocumentManifestRequest,
                response_model=DocumentManifestResponse,
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
            console_log += f"  Max attempts: {150} (timeout: {150 * 2.0}s = 5 min)\n"

            # Poll for completion
            poll_endpoint = ApiEndpoint(
                path=status_path,
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=DocumentManifestJobStatus,
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
                estimated_duration=15.0,  # Estimated 15s for manifest extraction
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

            # Log raw polling response for debugging
            console_log += f"\nRaw Poll Response:\n"
            try:
                result_dict = result.model_dump()
                console_log += f"{json.dumps(result_dict, indent=2)}\n"
            except Exception as e:
                console_log += f"ERROR dumping result: {e}\n"
                console_log += f"result type: {type(result)}\n"
                console_log += f"result: {result}\n"

            # Validate outputs
            if not result.outputs or len(result.outputs) == 0:
                console_log += f"\n{'='*55}\n"
                console_log += "ERROR: No outputs in response\n"
                console_log += f"  status: {result.status}\n"
                print(console_log)
                raise Exception("No outputs returned from Photoshop Document Manifest API")

            # Check for errors in output
            output = result.outputs[0]
            if output.errors:
                console_log += f"\n{'='*55}\n"
                console_log += "ERROR: Manifest extraction failed\n"
                for error in output.errors:
                    console_log += f"  {error}\n"
                print(console_log)
                raise Exception(f"Manifest extraction failed: {output.errors}")

            # Extract manifest data
            manifest_data = {
                "document": output.document.model_dump() if output.document else None,
                "layers": [layer.model_dump() for layer in output.layers] if output.layers else [],
            }

            manifest_json = json.dumps(manifest_data, indent=2)

            console_log += f"\n{'='*55}\n"
            console_log += "Manifest extracted successfully\n"
            if output.document:
                console_log += f"  Document: {output.document.name}\n"
                console_log += f"  Size: {output.document.width}x{output.document.height}\n"
            if output.layers:
                console_log += f"  Layers: {len(output.layers)}\n"
            console_log += f"{'='*55}\n"

            return (manifest_json, console_log)

        except Exception as e:
            console_log += f"\n{'='*55}\n"
            console_log += f"ERROR: {str(e)}\n"
            console_log += f"{'='*55}\n"
            print(console_log)

            # Close client
            try:
                await client.close()
            except:
                pass

            # Re-raise the exception
            raise
        finally:
            # Ensure client is closed
            try:
                await client.close()
            except:
                pass


# Node registration
NODE_CLASS_MAPPINGS = {
    "PhotoshopPsdManifestNode": PhotoshopPsdManifestNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoshopPsdManifestNode": "Photoshop PSD Manifest",
}
