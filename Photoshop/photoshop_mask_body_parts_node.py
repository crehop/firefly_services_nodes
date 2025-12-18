"""
Adobe Photoshop Mask Body Parts Node

Implements body part detection with label-based filtering.
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
    MaskBodyPartsRequest,
    MaskBodyPartsStatusResponse,
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


class PhotoshopMaskBodyPartsNode:
    """
    Detect and generate body part masks using Adobe Photoshop API.

    Features:
    - Dual input support for image and mask (tensor or pre-signed URL)
    - Automatic detection of body part masks (Hair, Face, Eyes, Hands, Coat, etc.)
    - Smart filtering with category support and contains matching (case-insensitive)
    - Parallel metadata outputs: each mask has corresponding JSON metadata
    - Comprehensive debug logging with timing info
    - Async polling for results

    Body Parts Detected:
    - Background, Hair, Eyebrows, Eyes, Nose, Mouth, Ears, Face, Neck
    - Hands, Coat, Upper Clothes, Lower Clothes, Shoes, Accessories, Pupil
    - May include variations like "Left Hand", "Right Eye", etc.

    Filtering:
    - Category filters: "head", "body", "clothing", "accessories"
    - Contains search: "left" → matches "Left Eye", "Left Hand", etc.
    - All matching is case-insensitive

    Examples:
    - filter_1 = "head" → outputs Hair, Eyes, Eyebrows, Nose, Mouth, Ears, Face, Neck, Pupil
    - filter_2 = "left" → outputs all masks containing "left" (Left Eye, Left Hand, etc.)
    - filter_3 = "clothing" → outputs Coat, Upper Clothes, Lower Clothes, Shoes
    - filter_4 = "eye" → outputs Eyes, Eyebrows
    - filter_5 = "Face" → outputs Face mask

    Outputs (12 total):
    - all_masks + all_masks_meta: ALL detected masks
    - filter_1 + filter_1_meta through filter_5 + filter_5_meta: Filtered outputs
    - masks_json: Complete metadata for all masks
    - debug_log: Detailed execution log
    """

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "STRING", "IMAGE", "STRING", "IMAGE", "STRING", "IMAGE", "STRING", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("all_masks", "all_masks_meta", "filter_1", "filter_1_meta", "filter_2", "filter_2_meta", "filter_3", "filter_3_meta", "filter_4", "filter_4_meta", "filter_5", "filter_5_meta", "masks_json", "debug_log")
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True, True, True, True, True, True, False, False)
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/photoshop"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                # Dual input for image: tensor OR reference URL (mutually exclusive)
                "image": ("IMAGE", {
                    "tooltip": "Input image tensor to process",
                }),
                "image_reference": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Pre-signed URL to input image (alternative to tensor)",
                }),

                # Dual input for mask: tensor OR reference URL (mutually exclusive)
                "mask": ("MASK", {
                    "tooltip": "Mask tensor to process",
                }),
                "mask_image": ("IMAGE", {
                    "tooltip": "Mask as IMAGE tensor to process (alternative to MASK)",
                }),
                "mask_reference": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Pre-signed URL to mask image (alternative to tensor)",
                }),

                # Filter inputs - first 2 are dropdown selectors, rest are string inputs
                "filter_1": (["none", "head", "body", "clothing", "accessories"], {
                    "default": "none",
                    "tooltip": "Select category (head/body/clothing/accessories) to filter masks. Case-insensitive contains matching.",
                }),
                "filter_2": (["none", "head", "body", "clothing", "accessories"], {
                    "default": "none",
                    "tooltip": "Select category to filter masks. Case-insensitive contains matching.",
                }),
                "filter_3": ("STRING", {
                    "default": "",
                    "tooltip": "Label to filter for filter_3 (e.g., 'face', 'hand', 'eye'). ALL masks with this label will be output.",
                }),
                "filter_4": ("STRING", {
                    "default": "",
                    "tooltip": "Label to filter for filter_4. ALL masks with this label will be output.",
                }),
                "filter_5": ("STRING", {
                    "default": "",
                    "tooltip": "Label to filter for filter_5. ALL masks with this label will be output.",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @staticmethod
    def _get_category_keywords(category: str) -> list:
        """
        Get list of keywords for predefined categories.
        Returns empty list if not a recognized category.
        All matching is case-insensitive.
        """
        category_lower = category.lower()

        CATEGORIES = {
            "head": ["hair", "eye", "pupil", "nose", "mouth", "ear", "face", "neck"],
            "body": ["hand", "arm", "leg", "neck", "torso"],
            "clothing": ["coat", "clothes", "shirt", "pant", "shoe", "sock", "dress", "skirt"],
            "accessories": ["accessor", "hat", "glass", "watch", "jewelry"],
        }

        return CATEGORIES.get(category_lower, [])

    @staticmethod
    def _filter_masks_by_keyword(all_masks: list, filter_keyword: str) -> list:
        """
        Filter masks based on a keyword using case-insensitive contains matching.

        Logic:
        1. If keyword is "none" → return empty list
        2. If keyword matches a category → apply category contains logic
        3. Otherwise → do contains search on all mask labels
        """
        if not filter_keyword or filter_keyword.lower() == "none":
            return []

        # Check if this is a category
        category_keywords = PhotoshopMaskBodyPartsNode._get_category_keywords(filter_keyword)

        if category_keywords:
            # Category matching: check if any keyword is contained in the label
            matched_masks = []
            for mask_data in all_masks:
                label_lower = mask_data['item'].label.lower()
                for keyword in category_keywords:
                    if keyword.lower() in label_lower:
                        matched_masks.append(mask_data)
                        break  # Don't add same mask multiple times
            return matched_masks
        else:
            # Direct contains matching: check if filter keyword is in the label
            keyword_lower = filter_keyword.lower()
            return [
                mask_data for mask_data in all_masks
                if keyword_lower in mask_data['item'].label.lower()
            ]

    def _build_debug_log(
        self,
        image: Optional[torch.Tensor],
        image_reference: str,
        mask: Optional[torch.Tensor],
        mask_image: Optional[torch.Tensor],
        mask_reference: str,
        filter_1: str,
        filter_2: str,
        filter_3: str,
        filter_4: str,
        filter_5: str,
    ) -> str:
        """Build formatted debug log showing request details."""
        log = "=" * 55 + "\n"
        log += "POST /v1/mask-body-parts\n"
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

        # Mask source
        if mask is not None:
            log += "  mask:\n"
            log += "    source:\n"
            log += "      url: [S3_PRESIGNED_URL] (MASK tensor)\n"
        elif mask_image is not None:
            log += "  mask:\n"
            log += "    source:\n"
            log += "      url: [S3_PRESIGNED_URL] (IMAGE tensor)\n"
        else:
            log += "  mask:\n"
            log += "    source:\n"
            log += f"      url: {mask_reference}\n"

        # Filter configuration
        filters = [filter_1, filter_2, filter_3, filter_4, filter_5]
        active_filters = [f for f in filters if f and f.lower() != "none"]

        if active_filters:
            log += "\nFilter Mode: ENABLED\n"
            for idx, filter_val in enumerate(filters, 1):
                if filter_val and filter_val.lower() != "none":
                    log += f"  filter_{idx}: '{filter_val}'\n"
        else:
            log += "\nFilter Mode: DISABLED (all masks output to all_masks)\n"

        return log

    async def api_call(
        self,
        image: Optional[torch.Tensor] = None,
        image_reference: str = "",
        mask: Optional[torch.Tensor] = None,
        mask_image: Optional[torch.Tensor] = None,
        mask_reference: str = "",
        filter_1: str = "none",
        filter_2: str = "none",
        filter_3: str = "none",
        filter_4: str = "none",
        filter_5: str = "none",
        unique_id: Optional[str] = None,
    ):
        """Detect body parts using Photoshop API and filter by keywords/categories."""

        # Validate image inputs
        image_inputs_provided = sum([image is not None, bool(image_reference)])
        if image_inputs_provided == 0:
            raise ValueError("Must provide one of: 'image' or 'image_reference'")
        if image_inputs_provided > 1:
            raise ValueError("Cannot provide both 'image' and 'image_reference' - choose only one")

        # Validate mask inputs
        mask_inputs_provided = sum([mask is not None, mask_image is not None, bool(mask_reference)])
        if mask_inputs_provided == 0:
            raise ValueError("Must provide one of: 'mask', 'mask_image', or 'mask_reference'")
        if mask_inputs_provided > 1:
            raise ValueError("Cannot provide multiple mask inputs - choose only one: 'mask', 'mask_image', or 'mask_reference'")

        # Build initial debug log
        console_log = self._build_debug_log(
            image=image,
            image_reference=image_reference,
            mask=mask,
            mask_image=mask_image,
            mask_reference=mask_reference,
            filter_1=filter_1,
            filter_2=filter_2,
            filter_3=filter_3,
            filter_4=filter_4,
            filter_5=filter_5,
        )

        client = await create_adobe_client()

        try:
            # Convert mask to RGB format if provided as MASK tensor
            if mask is not None:
                if len(mask.shape) == 3:  # [B, H, W]
                    mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3)  # Convert to [B, H, W, 3]
                elif len(mask.shape) == 2:  # [H, W]
                    mask = mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)  # Convert to [1, H, W, 3]
                console_log += f"\n{'='*55}\n"
                console_log += "Converting mask to RGB format...\n"
                console_log += f"  Mask shape: {mask.shape}\n"

            # Get image URL
            if image is not None:
                console_log += f"\n{'='*55}\n"
                console_log += "Uploading image to S3...\n"

                # Get image info
                img_tensor = image[0]
                h, w, c = img_tensor.shape
                console_log += f"  Image size: {w}x{h} ({c} channels)\n"

                # Upload and measure time
                upload_start = time.time()
                image_url = await upload_image_to_s3(img_tensor)
                upload_duration = time.time() - upload_start

                console_log += f"[OK] Upload complete ({upload_duration:.2f}s)\n"
                console_log += f"  Pre-signed URL generated (valid 24h)\n"
                console_log += f"  URL: {image_url[:100]}...\n"
            else:
                image_url = image_reference
                console_log += f"\n{'='*55}\n"
                console_log += "Using provided image reference URL\n"
                console_log += f"  URL: {image_url[:80]}{'...' if len(image_url) > 80 else ''}\n"

            # Get mask URL
            if mask is not None:
                console_log += f"\n{'='*55}\n"
                console_log += "Uploading mask to S3...\n"

                # Get mask info
                mask_tensor = mask[0]
                h, w, c = mask_tensor.shape
                console_log += f"  Mask size: {w}x{h} ({c} channels)\n"

                # Upload and measure time
                upload_start = time.time()
                mask_url = await upload_image_to_s3(mask_tensor)
                upload_duration = time.time() - upload_start

                console_log += f"[OK] Upload complete ({upload_duration:.2f}s)\n"
                console_log += f"  Pre-signed URL generated (valid 24h)\n"
                console_log += f"  URL: {mask_url[:100]}...\n"
            elif mask_image is not None:
                console_log += f"\n{'='*55}\n"
                console_log += "Uploading mask image to S3...\n"

                # Get mask image info
                mask_tensor = mask_image[0]
                h, w, c = mask_tensor.shape
                console_log += f"  Mask image size: {w}x{h} ({c} channels)\n"

                # Upload and measure time
                upload_start = time.time()
                mask_url = await upload_image_to_s3(mask_tensor)
                upload_duration = time.time() - upload_start

                console_log += f"[OK] Upload complete ({upload_duration:.2f}s)\n"
                console_log += f"  Pre-signed URL generated (valid 24h)\n"
                console_log += f"  URL: {mask_url[:100]}...\n"
            else:
                mask_url = mask_reference
                console_log += f"\n{'='*55}\n"
                console_log += "Using provided mask reference URL\n"
                console_log += f"  URL: {mask_url[:80]}{'...' if len(mask_url) > 80 else ''}\n"

            # Log the URLs being used
            console_log += f"\nURLs for Photoshop API:\n"
            console_log += f"  Image: {image_url}\n"
            console_log += f"  Mask: {mask_url}\n"

            # Build request (v1 API - no storage field)
            request = MaskBodyPartsRequest(
                image=PhotoshopImageInput(
                    source=PhotoshopImageSource(url=image_url)
                ),
                mask=PhotoshopImageInput(
                    source=PhotoshopImageSource(url=mask_url)
                )
            )

            # Log the actual request JSON being sent
            console_log += f"\nRequest JSON being sent to API:\n"
            console_log += f"{json.dumps(request.model_dump(exclude_none=True), indent=2)}\n"

            # Submit job
            submit_endpoint = ApiEndpoint(
                path="/v1/mask-body-parts",
                method=HttpMethod.POST,
                request_model=MaskBodyPartsRequest,
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
                response_model=MaskBodyPartsStatusResponse,
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

            # Check for masks
            total_masks = len(result.masks) if result.masks else 0
            console_log += f"  masks: {total_masks} detected\n"

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

            # Categorize masks into semantic (body parts) and background
            all_masks = []
            semantic_count = 0
            background_count = 0

            if result.masks:
                for mask_item in result.masks:
                    # Categorize based on label
                    mask_type = 'background' if mask_item.label.lower() == 'background' else 'semantic'
                    all_masks.append({
                        'type': mask_type,
                        'item': mask_item
                    })

                    if mask_type == 'semantic':
                        semantic_count += 1
                    else:
                        background_count += 1

            console_log += f"\n{'='*55}\n"
            console_log += f"Total masks available: {len(all_masks)}\n"
            console_log += f"  Semantic (body parts): {semantic_count}\n"
            console_log += f"  Background: {background_count}\n"
            console_log += "\nAvailable labels:\n"
            for mask_data in all_masks:
                mask_item = mask_data['item']
                console_log += f"  - '{mask_item.label}' ({mask_data['type']}, score: {mask_item.score:.3f})\n"

            # Build masks JSON for output
            masks_data = {
                "semanticMasks": [],
                "backgroundMasks": []
            }

            # Categorize masks into semantic and background based on label
            if result.masks:
                for mask_item in result.masks:
                    mask_dict = {
                        "label": mask_item.label,
                        "score": mask_item.score,
                        "boundingBox": {
                            "x": mask_item.boundingBox.x,
                            "y": mask_item.boundingBox.y,
                            "width": mask_item.boundingBox.width,
                            "height": mask_item.boundingBox.height
                        },
                        "url": mask_item.destination.url
                    }

                    # Categorize based on label
                    if mask_item.label.lower() == 'background':
                        masks_data["backgroundMasks"].append(mask_dict)
                    else:
                        masks_data["semanticMasks"].append(mask_dict)

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

            # Build all_masks list (ALL masks together)
            all_masks_list = []
            all_masks_meta_list = []

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

                all_masks_list.append(tensor)
                all_masks_meta_list.append(meta_json)

            # Add blank image and metadata if no masks detected
            blank_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            blank_meta = json.dumps({"label": "none", "score": 0, "type": "blank"}, indent=2)

            if len(all_masks_list) == 0:
                all_masks_list.append(blank_image)
                all_masks_meta_list.append(blank_meta)
                console_log += "\n[INFO] No masks detected - added blank image to all_masks list\n"

            # Initialize 5 filter output lists with metadata
            filter_lists = [[], [], [], [], []]
            filter_meta_lists = [[], [], [], [], []]
            filters = [filter_1, filter_2, filter_3, filter_4, filter_5]

            console_log += f"\n{'='*55}\n"
            console_log += "BUILDING FILTERED LISTS\n"
            console_log += f"{'-'*55}\n"

            # Build lists by matching filters using helper method
            for idx, filter_keyword in enumerate(filters):
                if not filter_keyword or filter_keyword.lower() == "none":
                    console_log += f"Filter {idx+1}: [SKIP] No filter specified\n"
                    continue

                console_log += f"\nFilter {idx+1}: Searching for '{filter_keyword}'\n"

                # Use helper method to filter masks
                # Need to convert mask_cache to the format expected by helper
                masks_for_filtering = [cached['data'] for cached in mask_cache.values()]
                matched_masks = self._filter_masks_by_keyword(masks_for_filtering, filter_keyword)

                # Convert matched masks to tensors and metadata
                for mask_data in matched_masks:
                    mask_item = mask_data['item']

                    # Find the tensor in cache
                    cache_key = f"{mask_data['type']}_{mask_item.label}_{mask_item.score}"
                    if cache_key in mask_cache:
                        tensor = mask_cache[cache_key]['tensor']
                        filter_lists[idx].append(tensor)

                        # Build metadata
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
                        filter_meta_lists[idx].append(meta_json)

                        console_log += f"  [{len(filter_lists[idx])}] '{mask_item.label}' ({mask_data['type']}, score: {mask_item.score:.3f})\n"

                if len(matched_masks) == 0:
                    console_log += f"  [NOT FOUND] No masks matching '{filter_keyword}'\n"
                else:
                    console_log += f"  Total matches: {len(matched_masks)}\n"

            # Add blank images and metadata to empty lists to prevent ComfyUI crash
            blank_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            blank_meta = json.dumps({"label": "none", "score": 0, "type": "blank"}, indent=2)

            for idx, filter_list in enumerate(filter_lists):
                if len(filter_list) == 0:
                    filter_lists[idx].append(blank_image)
                    filter_meta_lists[idx].append(blank_meta)
                    if filters[idx] and filters[idx].lower() != "none":
                        console_log += f"\nFilter {idx+1}: Added blank image (no matches for '{filters[idx]}')\n"
                    else:
                        console_log += f"\nFilter {idx+1}: Added blank image (no filter specified)\n"

            # Summary
            console_log += "SUMMARY\n"
            console_log += f"{'-'*55}\n"
            console_log += f"Total masks detected: {len(all_masks)}\n"
            console_log += f"  Body parts (semantic): {semantic_count}\n"
            console_log += f"  Background: {background_count}\n"

            console_log += "\nAll masks output:\n"
            console_log += f"  all_masks: {len(all_masks_list)} mask(s)\n"

            console_log += "\nFiltered outputs:\n"
            for idx, (filter_val, filter_list) in enumerate(zip(filters, filter_lists)):
                # Count real masks (exclude blank images)
                real_masks = len(filter_list) if filter_val and filter_val.lower() != "none" and len(filter_list) > 0 and filter_list[0].shape != torch.Size([1, 64, 64, 3]) else 0
                if not filter_val or filter_val.lower() == "none":
                    console_log += f"  filter_{idx+1}: [empty] (no filter)\n"
                elif real_masks > 0:
                    console_log += f"  filter_{idx+1}: {real_masks} mask(s) for '{filter_val}'\n"
                else:
                    console_log += f"  filter_{idx+1}: [blank] (no matches for '{filter_val}')\n"
            console_log += f"{'='*55}\n"

            # Convert JSON to string
            masks_json = json.dumps(masks_data, indent=2)

            return (
                all_masks_list,
                all_masks_meta_list,
                filter_lists[0],
                filter_meta_lists[0],
                filter_lists[1],
                filter_meta_lists[1],
                filter_lists[2],
                filter_meta_lists[2],
                filter_lists[3],
                filter_meta_lists[3],
                filter_lists[4],
                filter_meta_lists[4],
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
