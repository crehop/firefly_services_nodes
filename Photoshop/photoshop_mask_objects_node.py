"""
Adobe Photoshop Mask Objects Node

Implements mask detection with label-based filtering.
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
    PhotoshopImageSource,
    PhotoshopImageInput,
    MaskObjectsRequest,
    MaskObjectsStatusResponse,
    RemoveBackgroundResponse,
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


class PhotoshopMaskObjectsNode:
    """
    Detect and generate object and background masks using Adobe Photoshop API.

    Features:
    - Dual input support (tensor or pre-signed URL)
    - Automatic detection of semantic (object) and background masks
    - Multiple output modes: by type (semantic/background) AND by label
    - Label filtering supports multiple masks per label (e.g., 2 "grass" masks → both output)
    - Parallel metadata outputs: each mask has corresponding JSON metadata (label, score, bbox, type, URL)
    - Comprehensive debug logging with timing info
    - Async polling for results

    Outputs (16 total):
    Image Lists (with parallel metadata):
    - semantic_masks + semantic_meta: ALL semantic/object masks with metadata
    - background_masks + background_meta: ALL background masks with metadata
    - list_1 + list_1_meta through list_5 + list_5_meta: Filtered by labels
      - Each image at index i has metadata at index i in corresponding _meta list
      - Metadata includes: label, score, boundingBox (normalized 0-1), type, URL
      - Empty lists get blank image with {"label": "none", "type": "blank"} metadata

    Global Metadata:
    - masks_json: Complete metadata for all detected masks (both semantic + background)
    - debug_log: Detailed execution log with timing and matching info

    Example: If list_1[0] is a grass mask, then list_1_meta[0] contains:
    {"label": "grass", "score": 0.99, "boundingBox": {...}, "type": "background", "url": "..."}
    """

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "STRING", "IMAGE", "STRING", "IMAGE", "STRING", "IMAGE", "STRING", "IMAGE", "STRING", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("semantic_masks", "semantic_meta", "background_masks", "background_meta", "list_1", "list_1_meta", "list_2", "list_2_meta", "list_3", "list_3_meta", "list_4", "list_4_meta", "list_5", "list_5_meta", "masks_json", "debug_log")
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False)
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/photoshop"

    @classmethod
    def INPUT_TYPES(cls):
        return {
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

                # Label filters for specific mask extraction
                "label_1": ("STRING", {
                    "default": "",
                    "tooltip": "Label to filter for list_1 (e.g., 'grass', 'tree', 'sky'). ALL masks with this label will be output to list_1.",
                }),
                "label_2": ("STRING", {
                    "default": "",
                    "tooltip": "Label to filter for list_2. ALL masks with this label will be output to list_2.",
                }),
                "label_3": ("STRING", {
                    "default": "",
                    "tooltip": "Label to filter for list_3. ALL masks with this label will be output to list_3.",
                }),
                "label_4": ("STRING", {
                    "default": "",
                    "tooltip": "Label to filter for list_4. ALL masks with this label will be output to list_4.",
                }),
                "label_5": ("STRING", {
                    "default": "",
                    "tooltip": "Label to filter for list_5. ALL masks with this label will be output to list_5.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def _build_debug_log(
        self,
        image: Optional[torch.Tensor],
        image_reference: str,
        label_1: str,
        label_2: str,
        label_3: str,
        label_4: str,
        label_5: str,
    ) -> str:
        """Build formatted debug log showing request details."""
        log = "=" * 55 + "\n"
        log += "POST /v1/mask-objects\n"
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

        # Label filters
        labels = [label_1, label_2, label_3, label_4, label_5]
        has_labels = any(labels)

        if has_labels:
            log += "\nLabel Filter Mode: ENABLED\n"
            for idx, label in enumerate(labels, 1):
                if label:
                    log += f"  output_{idx}: '{label}'\n"
        else:
            log += "\nLabel Filter Mode: DISABLED (will output first 5 masks)\n"

        return log

    async def api_call(
        self,
        image: Optional[torch.Tensor] = None,
        image_reference: str = "",
        label_1: str = "",
        label_2: str = "",
        label_3: str = "",
        label_4: str = "",
        label_5: str = "",
        unique_id: Optional[str] = None,
    ):
        """Detect objects and backgrounds using Photoshop API and filter by labels."""

        # Validate inputs
        if image is None and not image_reference:
            raise ValueError("Must provide either 'image' or 'image_reference'")
        if image is not None and image_reference:
            raise ValueError("Cannot provide both 'image' and 'image_reference' - choose only one")

        # Build initial debug log
        console_log = self._build_debug_log(
            image=image,
            image_reference=image_reference,
            label_1=label_1,
            label_2=label_2,
            label_3=label_3,
            label_4=label_4,
            label_5=label_5,
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

            # Build request (v1 API - no storage field)
            request = MaskObjectsRequest(
                image=PhotoshopImageInput(
                    source=PhotoshopImageSource(url=input_url)
                )
            )

            # Log the actual request JSON being sent
            console_log += f"\nRequest JSON being sent to API:\n"
            console_log += f"{json.dumps(request.model_dump(exclude_none=True), indent=2)}\n"

            # Submit job
            submit_endpoint = ApiEndpoint(
                path="/v1/mask-objects",
                method=HttpMethod.POST,
                request_model=MaskObjectsRequest,
                response_model=RemoveBackgroundResponse,  # Reuse same response model (jobId, statusUrl)
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
            poll_endpoint = ApiEndpoint(
                path=status_path,
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=MaskObjectsStatusResponse,
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
                node_id=unique_id,
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

            # Check for semantic and background masks
            semantic_count = len(result.semanticMasks) if result.semanticMasks else 0
            background_count = len(result.backgroundMasks) if result.backgroundMasks else 0
            console_log += f"  semanticMasks: {semantic_count} detected\n"
            console_log += f"  backgroundMasks: {background_count} detected\n"

            # Log complete polling response as JSON for debugging
            console_log += f"\n{'='*55}\n"
            console_log += "FULL POLLING RESPONSE (JSON):\n"
            console_log += f"{'-'*55}\n"
            console_log += f"URL: {submit_response.statusUrl}\n"
            console_log += f"{'-'*55}\n"
            try:
                response_dict = result.model_dump(mode='json', exclude_none=False)
                console_log += f"{json.dumps(response_dict, indent=2)}\n"
            except Exception as e:
                console_log += f"ERROR serializing response to JSON: {e}\n"
            console_log += f"{'='*55}\n"

            # Combine all masks into a single list for processing
            all_masks = []

            # Add semantic masks
            if result.semanticMasks:
                for mask_item in result.semanticMasks:
                    all_masks.append({
                        'type': 'semantic',
                        'item': mask_item
                    })

            # Add background masks
            if result.backgroundMasks:
                for mask_item in result.backgroundMasks:
                    all_masks.append({
                        'type': 'background',
                        'item': mask_item
                    })

            console_log += f"\n{'='*55}\n"
            console_log += f"Total masks available: {len(all_masks)}\n"
            console_log += "Available labels:\n"
            for mask_data in all_masks:
                mask_item = mask_data['item']
                console_log += f"  - '{mask_item.label}' ({mask_data['type']}, score: {mask_item.score:.3f})\n"

            # Build masks JSON for output
            masks_data = {
                "semanticMasks": [],
                "backgroundMasks": []
            }

            if result.semanticMasks:
                for mask_item in result.semanticMasks:
                    masks_data["semanticMasks"].append({
                        "label": mask_item.label,
                        "score": mask_item.score,
                        "boundingBox": {
                            "x": mask_item.boundingBox.x,
                            "y": mask_item.boundingBox.y,
                            "width": mask_item.boundingBox.width,
                            "height": mask_item.boundingBox.height
                        },
                        "url": mask_item.destination.url
                    })

            if result.backgroundMasks:
                for mask_item in result.backgroundMasks:
                    masks_data["backgroundMasks"].append({
                        "label": mask_item.label,
                        "score": mask_item.score,
                        "boundingBox": {
                            "x": mask_item.boundingBox.x,
                            "y": mask_item.boundingBox.y,
                            "width": mask_item.boundingBox.width,
                            "height": mask_item.boundingBox.height
                        },
                        "url": mask_item.destination.url
                    })

            # Download all masks ONCE and cache them
            console_log += f"\n{'='*55}\n"
            console_log += "DOWNLOADING ALL MASKS\n"
            console_log += f"{'-'*55}\n"

            # Dictionary to cache downloaded masks: mask_item.label -> tensor
            mask_cache = {}

            async with aiohttp.ClientSession() as session:
                for mask_data in all_masks:
                    mask_item = mask_data['item']
                    mask_type = mask_data['type']

                    # Create unique key for this mask
                    cache_key = f"{mask_type}_{mask_item.label}_{mask_item.score}"

                    console_log += f"[{mask_type}] '{mask_item.label}' (score: {mask_item.score:.3f})..."

                    try:
                        async with session.get(mask_item.destination.url) as resp:
                            if resp.status != 200:
                                console_log += f" ERROR HTTP {resp.status}\n"
                                continue
                            image_bytes = await resp.read()

                        # Convert to tensor
                        img_pil = Image.open(io.BytesIO(image_bytes))

                        # Always convert to RGB (handles RGBA, L, P, etc.)
                        if img_pil.mode != 'RGB':
                            img_pil = img_pil.convert('RGB')

                        img_array = np.array(img_pil).astype(np.float32) / 255.0

                        # Ensure we have [H, W, 3] shape
                        if len(img_array.shape) == 2:  # Grayscale
                            img_array = np.stack([img_array] * 3, axis=-1)
                        elif len(img_array.shape) == 3 and img_array.shape[2] != 3:
                            # Wrong number of channels, force RGB conversion
                            img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                            img_array = np.array(img_pil).astype(np.float32) / 255.0

                        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

                        # Cache for reuse
                        mask_cache[cache_key] = {
                            'tensor': img_tensor,
                            'data': mask_data
                        }

                        console_log += " OK\n"

                    except Exception as e:
                        console_log += f" ERROR: {str(e)}\n"

            # Build semantic and background mask lists (ALL masks by type)
            semantic_masks_list = []
            semantic_meta_list = []
            background_masks_list = []
            background_meta_list = []

            for cache_key, cached in mask_cache.items():
                mask_data = cached['data']
                mask_item = mask_data['item']
                tensor = cached['tensor']  # Shape: [1, H, W, 3]

                # Build metadata JSON for this mask
                meta_json = json.dumps({
                    "label": mask_item.label,
                    "score": mask_item.score,
                    "boundingBox": {
                        "x": mask_item.boundingBox.x,
                        "y": mask_item.boundingBox.y,
                        "width": mask_item.boundingBox.width,
                        "height": mask_item.boundingBox.height
                    },
                    "type": mask_data['type'],
                    "url": mask_item.destination.url
                }, indent=2)

                if mask_data['type'] == 'semantic':
                    semantic_masks_list.append(tensor)
                    semantic_meta_list.append(meta_json)
                elif mask_data['type'] == 'background':
                    background_masks_list.append(tensor)
                    background_meta_list.append(meta_json)

            # Add blank images and metadata if empty
            blank_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            blank_meta = json.dumps({"label": "none", "score": 0, "type": "blank"}, indent=2)

            if len(semantic_masks_list) == 0:
                semantic_masks_list.append(blank_image)
                semantic_meta_list.append(blank_meta)
                console_log += "\n[INFO] No semantic masks detected - added blank image to semantic_masks list\n"
            if len(background_masks_list) == 0:
                background_masks_list.append(blank_image)
                background_meta_list.append(blank_meta)
                console_log += "[INFO] No background masks detected - added blank image to background_masks list\n"

            # Initialize 5 label-filtered output lists with metadata
            output_lists = [[], [], [], [], []]
            output_meta_lists = [[], [], [], [], []]
            label_filters = [label_1, label_2, label_3, label_4, label_5]

            console_log += f"\n{'='*55}\n"
            console_log += "BUILDING LABEL-FILTERED LISTS\n"
            console_log += f"{'-'*55}\n"

            # Build lists by matching labels
            for idx, label_filter in enumerate(label_filters):
                if not label_filter:
                    console_log += f"List {idx+1}: [SKIP] No label specified\n"
                    continue

                console_log += f"\nList {idx+1}: Searching for '{label_filter}'\n"
                matches_found = 0

                # Find ALL masks matching this label
                for cache_key, cached in mask_cache.items():
                    mask_data = cached['data']
                    mask_item = mask_data['item']
                    tensor = cached['tensor']  # Shape: [1, H, W, 3]

                    if mask_item.label.lower() == label_filter.lower():
                        output_lists[idx].append(tensor)

                        # Build metadata for this mask
                        meta_json = json.dumps({
                            "label": mask_item.label,
                            "score": mask_item.score,
                            "boundingBox": {
                                "x": mask_item.boundingBox.x,
                                "y": mask_item.boundingBox.y,
                                "width": mask_item.boundingBox.width,
                                "height": mask_item.boundingBox.height
                            },
                            "type": mask_data['type'],
                            "url": mask_item.destination.url
                        }, indent=2)
                        output_meta_lists[idx].append(meta_json)

                        matches_found += 1
                        mask_type = mask_data['type']
                        console_log += f"  [{matches_found}] '{mask_item.label}' ({mask_type}, score: {mask_item.score:.3f})\n"

                if matches_found == 0:
                    console_log += f"  [NOT FOUND] No masks with label '{label_filter}'\n"
                else:
                    console_log += f"  Total matches: {matches_found}\n"

            # Add blank images and metadata to empty lists to prevent ComfyUI crash
            blank_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            blank_meta = json.dumps({"label": "none", "score": 0, "type": "blank"}, indent=2)

            for idx, output_list in enumerate(output_lists):
                if len(output_list) == 0:
                    output_lists[idx].append(blank_image)
                    output_meta_lists[idx].append(blank_meta)
                    if label_filters[idx]:
                        console_log += f"\nList {idx+1}: Added blank image (no matches for '{label_filters[idx]}')\n"
                    else:
                        console_log += f"\nList {idx+1}: Added blank image (no label specified)\n"

            # Summary
            console_log += "SUMMARY\n"
            console_log += f"{'-'*55}\n"
            console_log += f"Total masks detected: {len(all_masks)}\n"
            console_log += f"  Semantic: {semantic_count}\n"
            console_log += f"  Background: {background_count}\n"

            console_log += "\nType-based outputs:\n"
            console_log += f"  semantic_masks: {len(semantic_masks_list)} mask(s)\n"
            console_log += f"  background_masks: {len(background_masks_list)} mask(s)\n"

            console_log += "\nLabel-filtered outputs:\n"
            for idx, (label, output_list) in enumerate(zip(label_filters, output_lists)):
                # Count real masks (exclude blank images)
                real_masks = len(output_list) if label and len(output_list) > 0 and output_list[0].shape != torch.Size([1, 64, 64, 3]) else 0
                if not label:
                    console_log += f"  list_{idx+1}: [empty] (no label)\n"
                elif real_masks > 0:
                    console_log += f"  list_{idx+1}: {real_masks} mask(s) for '{label}'\n"
                else:
                    console_log += f"  list_{idx+1}: [blank] (no matches for '{label}')\n"
            console_log += f"{'='*55}\n"

            # Convert JSON to string
            masks_json = json.dumps(masks_data, indent=2)

            return (
                semantic_masks_list,
                semantic_meta_list,
                background_masks_list,
                background_meta_list,
                output_lists[0],
                output_meta_lists[0],
                output_lists[1],
                output_meta_lists[1],
                output_lists[2],
                output_meta_lists[2],
                output_lists[3],
                output_meta_lists[3],
                output_lists[4],
                output_meta_lists[4],
                masks_json,
                console_log
            )

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
