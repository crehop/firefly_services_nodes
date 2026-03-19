"""
Adobe Photoshop Smart Object Node

Replace smart object contents in PSD files using Adobe Photoshop API.
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
    SmartObjectLayer,
    SmartObjectLayerInput,
    SmartObjectOptions,
    SmartObjectOutput,
    SmartObjectRequest,
    SmartObjectResponse,
    SmartObjectJobStatus,
)
from .photoshop_storage import upload_file_to_s3, upload_image_to_s3, generate_output_presigned_url, generate_download_url
from ..Firefly.firefly_storage import create_adobe_client
from ..client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)


class PhotoshopSmartObjectNode:
    """
    Replace smart object contents in PSD files using Adobe Photoshop API.

    Features:
    - Replace embedded smart object with new image
    - Support for multiple smart object layers
    - Output as PSD, PNG, JPEG, or TIFF
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
                # Replacement image - tensor or URL
                "replacement_image": ("IMAGE", {
                    "tooltip": "Image tensor to replace smart object content",
                }),
                "alpha_channel_mask": ("MASK", {
                    "tooltip": "Connect LoadImage MASK output here to preserve transparency",
                }),
                "replacement_image_url": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Pre-signed URL to replacement image (alternative to tensor)",
                }),

                # Layer identification
                "layer_id": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Smart object layer ID (use either layer_id or layer_name)",
                }),
                "layer_name": ("STRING", {
                    "default": "",
                    "tooltip": "Smart object layer name (use either layer_id or layer_name)",
                }),

                # Advanced options via JSON
                "layers_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional: JSON array for multiple smart object replacements. Overrides layer_id/layer_name if provided.",
                }),

                # Output settings
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
        layer_id: int,
        layer_name: str,
        layers_json: str,
        output_type: str,
        has_replacement_image: bool,
        replacement_image_url: str,
        has_mask: bool = False,
    ) -> str:
        """Build formatted debug log showing request details."""
        log = "=" * 55 + "\n"
        log += "POST /pie/psdService/smartObject\n"
        log += "-" * 55 + "\n"
        log += "Headers:\n"
        log += "  Authorization: Bearer [TOKEN]\n"
        log += "  x-api-key: [API_KEY]\n"
        log += "  Content-Type: application/json\n"
        log += "\nRequest Summary:\n"

        # Input PSD
        log += "  inputs:\n"
        if is_local_file:
            log += f"    - href: [S3_PRESIGNED_URL] (from local file: {psd_input})\n"
        else:
            log += f"    - href: {psd_input[:80]}{'...' if len(psd_input) > 80 else ''}\n"
        log += "      storage: external\n"

        # Replacement image
        log += "\n  replacement_image:\n"
        if has_replacement_image:
            if has_mask:
                log += "    - source: image tensor + mask (RGBA, transparency preserved)\n"
            else:
                log += "    - source: image tensor (RGB, no transparency)\n"
        elif replacement_image_url:
            log += f"    - href: {replacement_image_url[:80]}{'...' if len(replacement_image_url) > 80 else ''}\n"
        else:
            log += "    - ERROR: No replacement image provided!\n"

        # Layer info
        log += "\n  target_layer:\n"
        if layers_json:
            log += "    - (multiple layers via JSON)\n"
        elif layer_name:
            log += f"    - name: {layer_name}\n"
        elif layer_id > 0:
            log += f"    - id: {layer_id}\n"
        else:
            log += "    - ERROR: No layer specified!\n"

        # Output
        log += f"\n  output_type: {output_type}\n"

        # Layers JSON if provided
        if layers_json:
            log += "\nLayers JSON:\n"
            log += f"{layers_json}\n"

        return log

    async def api_call(
        self,
        psd_input: str = "",
        replacement_image: Optional[torch.Tensor] = None,
        alpha_channel_mask: Optional[torch.Tensor] = None,
        replacement_image_url: str = "",
        layer_id: int = 0,
        layer_name: str = "",
        layers_json: str = "",
        output_type: str = "image/vnd.adobe.photoshop",
    ):
        """Replace smart object in PSD file using Photoshop API."""

        # Validate inputs
        if not psd_input:
            raise ValueError("Must provide 'psd_input' (local file path or URL)")

        # Validate replacement image
        if replacement_image is None and not replacement_image_url:
            raise ValueError("Must provide either 'replacement_image' or 'replacement_image_url'")
        if replacement_image is not None and replacement_image_url:
            raise ValueError("Cannot provide both 'replacement_image' and 'replacement_image_url' - choose only one")

        # Validate layer specification (unless using layers_json)
        if not layers_json:
            if layer_id == 0 and not layer_name:
                raise ValueError("Must provide 'layer_id', 'layer_name', or 'layers_json' to identify the smart object layer")

        # Parse layers JSON if provided
        layers_list = None
        if layers_json and layers_json.strip():
            try:
                layers_list = json.loads(layers_json)
                if not isinstance(layers_list, list):
                    raise ValueError("layers_json must be a JSON array")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in layers_json: {e}")

        # Detect if PSD input is URL or local file path
        is_url = psd_input.startswith(("http://", "https://"))
        is_local_file = not is_url

        # Validate file exists if local path
        if is_local_file and not os.path.exists(psd_input):
            raise ValueError(f"PSD file not found at path: {psd_input}")

        # Build initial debug log
        console_log = self._build_debug_log(
            psd_input=psd_input,
            is_local_file=is_local_file,
            layer_id=layer_id,
            layer_name=layer_name,
            layers_json=layers_json,
            output_type=output_type,
            has_replacement_image=replacement_image is not None,
            replacement_image_url=replacement_image_url,
            has_mask=alpha_channel_mask is not None,
        )

        print(f"\n[DEBUG] ===== SMART OBJECT NODE START =====")
        print(f"[DEBUG] psd_input: {psd_input}")
        print(f"[DEBUG] layer_id: {layer_id}, layer_name: {layer_name}")
        print(f"[DEBUG] output_type: {output_type}")

        client = await create_adobe_client()

        try:
            # Get PSD input URL
            if is_local_file:
                console_log += f"\n{'='*55}\n"
                console_log += "Uploading PSD file to S3...\n"

                # Get file size
                file_size = os.path.getsize(psd_input)
                console_log += f"  File path: {psd_input}\n"
                console_log += f"  File size: {file_size / (1024*1024):.2f} MB\n"

                print(f"[DEBUG] Uploading local PSD file: {psd_input} ({file_size} bytes)")

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

                print(f"[DEBUG] Using provided PSD URL: {input_url[:100]}...")

            # Get replacement image URL
            if replacement_image is not None:
                console_log += f"\n{'='*55}\n"
                console_log += "Uploading replacement image to S3...\n"

                # Get image info
                img_tensor = replacement_image[0]  # [H, W, C]
                h, w, c = img_tensor.shape

                # If mask provided, combine RGB + mask → RGBA to preserve transparency
                if alpha_channel_mask is not None:
                    mask_tensor = alpha_channel_mask[0] if len(alpha_channel_mask.shape) > 2 else alpha_channel_mask
                    # Invert: ComfyUI MASK is 1=transparent, but alpha needs 1=opaque
                    mask_tensor = 1.0 - mask_tensor
                    # Resize mask to match image if needed
                    if mask_tensor.shape[0] != h or mask_tensor.shape[1] != w:
                        mask_pil = Image.fromarray((mask_tensor.cpu().numpy() * 255).astype(np.uint8))
                        mask_pil = mask_pil.resize((w, h), Image.LANCZOS)
                        mask_tensor = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0)
                    # Combine: [H, W, 3] + [H, W, 1] → [H, W, 4]
                    if len(mask_tensor.shape) == 2:
                        mask_tensor = mask_tensor.unsqueeze(-1)
                    img_tensor = torch.cat([img_tensor, mask_tensor], dim=-1)
                    c = 4
                    console_log += f"  Combined IMAGE + MASK → RGBA ({w}x{h}, 4 channels)\n"
                    print(f"[DEBUG] Combined IMAGE + MASK → RGBA: {w}x{h}")
                else:
                    console_log += f"  Image size: {w}x{h} ({c} channels, no mask/alpha)\n"
                    print(f"[DEBUG] Uploading replacement image: {w}x{h} ({c}ch, no alpha)")

                # Upload and measure time
                upload_start = time.time()
                replacement_url = await upload_image_to_s3(img_tensor)
                upload_duration = time.time() - upload_start

                console_log += f"[OK] Upload complete ({upload_duration:.2f}s)\n"
                console_log += f"  Pre-signed URL generated (valid 24h)\n"

                print(f"[DEBUG] Replacement image upload complete in {upload_duration:.2f}s")
                print(f"[DEBUG] Replacement URL: {replacement_url[:100]}...")
            else:
                replacement_url = replacement_image_url
                console_log += f"\n{'='*55}\n"
                console_log += "Using provided replacement image URL\n"
                console_log += f"  URL: {replacement_url[:80]}{'...' if len(replacement_url) > 80 else ''}\n"

                print(f"[DEBUG] Using provided replacement URL: {replacement_url[:100]}...")

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

            # Build smart object layers
            smart_object_layers = []

            if layers_list:
                # Use layers_json for multiple smart objects
                for layer in layers_list:
                    # Each layer in JSON needs its own replacement image input
                    layer_input_url = layer.get("input", {}).get("href", replacement_url)
                    smart_object_layers.append(SmartObjectLayer(
                        id=layer.get("id"),
                        name=layer.get("name"),
                        input=SmartObjectLayerInput(
                            href=layer_input_url,
                            storage="external"
                        ),
                        locked=layer.get("locked", False),
                        visible=layer.get("visible", True),
                    ))
            else:
                # Use simple layer_id/layer_name
                smart_object_layers.append(SmartObjectLayer(
                    id=layer_id if layer_id > 0 else None,
                    name=layer_name if layer_name else None,
                    input=SmartObjectLayerInput(
                        href=replacement_url,
                        storage="external"
                    ),
                ))

            # Build request
            request = SmartObjectRequest(
                inputs=[
                    PhotoshopActionsInput(
                        href=input_url,
                        storage="external"
                    )
                ],
                outputs=[
                    SmartObjectOutput(
                        href=output_url_presigned,
                        storage="external",
                        type=output_type,
                        overwrite=True
                    )
                ],
                options=SmartObjectOptions(
                    layers=smart_object_layers
                )
            )

            # Log the actual request JSON being sent
            request_json_str = json.dumps(request.model_dump(exclude_none=True), indent=2)
            console_log += f"\n{'='*55}\n"
            console_log += "Request JSON being sent to API:\n"
            console_log += f"{request_json_str}\n"

            print(f"[DEBUG] Request JSON:\n{request_json_str}")

            # Submit job
            print(f"[DEBUG] Submitting job to /pie/psdService/smartObject...")

            submit_endpoint = ApiEndpoint(
                path="/pie/psdService/smartObject",
                method=HttpMethod.POST,
                request_model=SmartObjectRequest,
                response_model=SmartObjectResponse,
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
                response_model=SmartObjectJobStatus,
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
                    raise Exception(f"Smart object replacement failed: {error_msg}")

            # Validate outputs
            if not result.result or not result.result.outputs:
                console_log += f"\n{'='*55}\n"
                console_log += "ERROR: No outputs in response\n"
                console_log += f"  status: {result.status}\n"
                console_log += f"  jobId: {result.jobId}\n"

                print(f"[DEBUG] No outputs in response")
                print(console_log)

                raise Exception("No outputs returned from Smart Object API.")

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

            print(f"[DEBUG] ===== SMART OBJECT NODE COMPLETE =====\n")

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
    "PhotoshopSmartObjectNode": PhotoshopSmartObjectNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoshopSmartObjectNode": "Photoshop Smart Object",
}
