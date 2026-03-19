"""
Adobe Photoshop Document Operations Node

Apply edits to existing PSD files using Adobe Photoshop API.
Supports layer modifications, adjustments, text edits, and more.
"""

from __future__ import annotations
from typing import Optional
import time
import json
import os
from urllib.parse import urlparse

from .photoshop_api import (
    PhotoshopJobStatusEnum,
    DocumentOperationsInput,
    DocumentOperationsOutput,
    DocumentOperationsRequest,
    DocumentOperationsResponse,
    DocumentOperationsJobStatus,
    DocumentManifestInput,
    DocumentManifestRequest,
    DocumentManifestResponse,
    DocumentManifestJobStatus,
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


class PhotoshopDocumentOperationsNode:
    """
    Apply edits to existing PSD files using Adobe Photoshop API.

    Features:
    - Modify existing layers (visibility, opacity, blend mode, etc.)
    - Add new layers (pixel, text, adjustment, fill, smart object)
    - Delete layers
    - Move/reorder layers
    - Edit text layer content
    - Apply adjustments
    - Returns the manifest of the resulting document
    """

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_url", "manifest_json", "debug_log")
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
                "operations_json": ("STRING", {
                    "default": """{
  "layers": [
    {
      "id": 1,
      "visible": true,
      "blendOptions": {
        "opacity": 100
      }
    }
  ]
}""",
                    "multiline": True,
                    "tooltip": "JSON defining layer operations. See Adobe Photoshop API docs for full schema.",
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
            },
        }

    def _build_debug_log(
        self,
        psd_input: str,
        is_local_file: bool,
        operations_json: str,
        output_type: str,
    ) -> str:
        """Build formatted debug log showing request details."""
        log = "=" * 55 + "\n"
        log += "POST /pie/psdService/documentOperations\n"
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

        # Operations preview
        log += "\nOperations JSON:\n"
        log += f"{operations_json}\n"

        return log

    async def api_call(
        self,
        psd_input: str = "",
        operations_json: str = "",
        output_type: str = "image/vnd.adobe.photoshop",
    ):
        """Apply document operations to PSD file using Photoshop API."""

        # Validate inputs
        if not psd_input:
            raise ValueError("Must provide 'psd_input' (local file path or URL)")
        if not operations_json:
            raise ValueError("Must provide 'operations_json' with layer operations")

        # Parse operations JSON
        try:
            operations_dict = json.loads(operations_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in operations_json: {e}")

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
            operations_json=operations_json,
            output_type=output_type,
        )

        print(f"\n[DEBUG] ===== DOCUMENT OPERATIONS NODE START =====")
        print(f"[DEBUG] psd_input: {psd_input}")
        print(f"[DEBUG] operations_json: {operations_json}")
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

            # Build request
            request_dict = {
                "inputs": [
                    {
                        "href": input_url,
                        "storage": "external"
                    }
                ],
                "outputs": [
                    {
                        "href": output_url_presigned,
                        "storage": "external",
                        "type": output_type,
                        "overwrite": True
                    }
                ],
                "options": operations_dict
            }

            # Log the actual request JSON being sent
            request_json_str = json.dumps(request_dict, indent=2)
            console_log += f"\n{'='*55}\n"
            console_log += "Request JSON being sent to API:\n"
            console_log += f"{request_json_str}\n"

            print(f"[DEBUG] Request JSON:\n{request_json_str}")

            # Submit job
            print(f"[DEBUG] Submitting job to /pie/psdService/documentOperations...")

            submit_endpoint = ApiEndpoint(
                path="/pie/psdService/documentOperations",
                method=HttpMethod.POST,
                request_model=dict,
                response_model=DocumentOperationsResponse,
            )

            submit_op = SynchronousOperation(
                endpoint=submit_endpoint,
                request=request_dict,
                api_base="https://image.adobe.io",
            )

            # Update client's base_url
            client.base_url = "https://image.adobe.io"
            submit_response = await submit_op.execute(client=client)

            # Log submit response
            console_log += f"\nResponse: 202 Accepted\n"
            console_log += f"  jobId: {submit_response.jobId}\n"
            console_log += f"  statusUrl: {submit_response.statusUrl}\n"

            print(f"[DEBUG] Submit response received")
            print(f"[DEBUG]   jobId: {submit_response.jobId}")
            print(f"[DEBUG]   statusUrl: {submit_response.statusUrl}")

            # Log raw response for debugging
            submit_response_dict = submit_response.model_dump()
            console_log += f"\nRaw Submit Response:\n"
            console_log += f"{json.dumps(submit_response_dict, indent=2)}\n"

            print(f"[DEBUG] Raw submit response: {json.dumps(submit_response_dict, indent=2)}")

            # Parse statusUrl to get correct polling endpoint
            parsed_status_url = urlparse(submit_response.statusUrl)
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
            console_log += f"  Status URL: {submit_response.statusUrl}\n"
            console_log += f"  Interval: {2.0}s\n"
            console_log += f"  Max attempts: {300} (timeout: {300 * 2.0}s = 10 min)\n"

            # Poll for completion
            poll_endpoint = ApiEndpoint(
                path=status_path,
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=DocumentOperationsJobStatus,
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
                    return ("", json.dumps({"error": first_output.errors}, indent=2), console_log)

            # Check if job failed
            if result.status == PhotoshopJobStatusEnum.FAILED or str(result.status) == "failed":
                console_log += f"\n{'='*55}\n"
                console_log += "JOB FAILED\n"
                console_log += f"  status: {result.status}\n"
                console_log += f"{'='*55}\n"

                print(f"[DEBUG] Job failed with status: {result.status}")

                # Return error info
                return ("", json.dumps({"error": "Job failed", "status": str(result.status)}, indent=2), console_log)

            # Validate outputs
            if not result.result or not result.result.outputs:
                console_log += f"\n{'='*55}\n"
                console_log += "ERROR: No outputs in response\n"
                console_log += f"  status: {result.status}\n"
                console_log += f"  jobId: {result.jobId}\n"

                print(f"[DEBUG] No outputs in response")
                print(console_log)

                raise Exception(f"No outputs returned from Document Operations API.")

            # Generate GET URL for downloading the result
            console_log += f"\n{'='*55}\n"
            console_log += "Generating download URL...\n"

            print(f"[DEBUG] Generating download URL for: {output_filename}")

            download_url = await generate_download_url(output_filename)
            console_log += f"[OK] Generated download URL\n"
            console_log += f"  URL: {download_url[:80]}...\n"

            print(f"[DEBUG] Download URL: {download_url[:100]}...")

            # Now get the manifest of the resulting document
            console_log += f"\n{'='*55}\n"
            console_log += "Fetching manifest of resulting document...\n"

            print(f"[DEBUG] Fetching manifest of resulting document...")

            # Build manifest request
            manifest_request_dict = {
                "inputs": [
                    {
                        "href": download_url,
                        "storage": "external"
                    }
                ]
            }

            print(f"[DEBUG] Manifest request: {json.dumps(manifest_request_dict, indent=2)}")

            # Submit manifest job
            manifest_submit_endpoint = ApiEndpoint(
                path="/pie/psdService/documentManifest",
                method=HttpMethod.POST,
                request_model=dict,
                response_model=DocumentOperationsResponse,
            )

            manifest_submit_op = SynchronousOperation(
                endpoint=manifest_submit_endpoint,
                request=manifest_request_dict,
                api_base="https://image.adobe.io",
            )

            manifest_response = await manifest_submit_op.execute(client=client)
            console_log += f"  Manifest job submitted: {manifest_response.jobId}\n"
            console_log += f"  Status URL: {manifest_response.statusUrl}\n"

            print(f"[DEBUG] Manifest job submitted: {manifest_response.jobId}")
            print(f"[DEBUG] Manifest status URL: {manifest_response.statusUrl}")

            # Parse manifest statusUrl
            manifest_parsed_url = urlparse(manifest_response.statusUrl)
            manifest_base_url = f"{manifest_parsed_url.scheme}://{manifest_parsed_url.netloc}"
            manifest_status_path = manifest_parsed_url.path

            # Poll for manifest completion
            manifest_poll_endpoint = ApiEndpoint(
                path=manifest_status_path,
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=DocumentManifestJobStatus,
            )

            manifest_poll_op = PollingOperation(
                poll_endpoint=manifest_poll_endpoint,
                request=EmptyRequest(),
                completed_statuses=["succeeded"],
                failed_statuses=["failed"],
                status_extractor=lambda x: x.status,
                api_base=manifest_base_url,
                poll_interval=2.0,
                max_poll_attempts=60,
            )

            console_log += "  Polling for manifest...\n"
            print(f"[DEBUG] Polling for manifest...")

            manifest_result = await manifest_poll_op.execute(client=client)
            console_log += f"  Manifest retrieved successfully\n"

            print(f"[DEBUG] Manifest retrieved successfully")
            print(f"[DEBUG] Manifest result: {manifest_result.model_dump() if manifest_result else 'None'}")

            # Extract manifest JSON
            manifest_json = ""
            if manifest_result.outputs and len(manifest_result.outputs) > 0:
                manifest_output = manifest_result.outputs[0]

                # Check for errors in manifest response
                if manifest_output.errors:
                    console_log += f"  WARNING: Manifest has errors: {manifest_output.errors}\n"
                    print(f"[DEBUG] Manifest has errors: {manifest_output.errors}")

                manifest_data = {
                    "document": manifest_output.document.model_dump() if manifest_output.document else None,
                    "layers": [layer.model_dump() for layer in manifest_output.layers] if manifest_output.layers else []
                }
                manifest_json = json.dumps(manifest_data, indent=2)

                print(f"[DEBUG] Manifest JSON extracted, length: {len(manifest_json)}")
            else:
                manifest_json = "{}"
                console_log += "  WARNING: No manifest data in response\n"
                print(f"[DEBUG] WARNING: No manifest data in response")

            console_log += f"\n{'='*55}\n"
            console_log += "Document operations completed successfully!\n"
            console_log += f"{'='*55}\n"

            print(f"[DEBUG] ===== DOCUMENT OPERATIONS NODE COMPLETE =====\n")

            return (download_url, manifest_json, console_log)

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
