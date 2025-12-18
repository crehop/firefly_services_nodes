"""
Filter Masks Utility Node

Provides additional filtering for masks from Mask Objects or Mask Body Parts nodes.
"""

from __future__ import annotations
from typing import List
import torch
import json


class FilterMasksNode:
    """
    Filter masks from Mask Objects or Mask Body Parts nodes.

    Takes the all_masks and all_masks_meta outputs and applies additional filtering.
    Supports both category-based filtering (for body parts) and label-based filtering.

    Features:
    - 2 dropdown selectors for categories (head, body, clothing, accessories)
    - 3 string inputs for direct label matching
    - 5 filtered outputs with parallel metadata
    - Case-insensitive contains matching
    - Blank images for empty results

    Use Case:
    Connect the all_masks + all_masks_meta outputs from Mask Objects or Mask Body Parts
    to this node for additional filtering beyond the original 5 filters.
    """

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "STRING", "IMAGE", "STRING", "IMAGE", "STRING", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("filter_1", "filter_1_meta", "filter_2", "filter_2_meta", "filter_3", "filter_3_meta", "filter_4", "filter_4_meta", "filter_5", "filter_5_meta", "filtered_json", "debug_log")
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True, True, True, True, False, False)
    FUNCTION = "filter_masks"
    CATEGORY = "api node/Firefly Utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "all_masks": ("IMAGE", {
                    "tooltip": "All masks from Mask Objects or Mask Body Parts node",
                }),
                "all_masks_meta": ("STRING", {
                    "tooltip": "All masks metadata from Mask Objects or Mask Body Parts node",
                }),
            },
            "optional": {
                # First 2 are dropdown selectors for body parts categories
                "filter_1": (["none", "head", "body", "clothing", "accessories"], {
                    "default": "none",
                    "tooltip": "Select category or use as direct label search. Case-insensitive contains matching.",
                }),
                "filter_2": (["none", "head", "body", "clothing", "accessories"], {
                    "default": "none",
                    "tooltip": "Select category or use as direct label search. Case-insensitive contains matching.",
                }),
                # Last 3 are string inputs for direct label matching
                "filter_3": ("STRING", {
                    "default": "",
                    "tooltip": "Label to filter for filter_3 (e.g., 'grass', 'tree', 'face'). ALL masks matching will be output.",
                }),
                "filter_4": ("STRING", {
                    "default": "",
                    "tooltip": "Label to filter for filter_4. ALL masks matching will be output.",
                }),
                "filter_5": ("STRING", {
                    "default": "",
                    "tooltip": "Label to filter for filter_5. ALL masks matching will be output.",
                }),
            },
        }

    @staticmethod
    def _get_category_keywords(category: str) -> list:
        """
        Get list of keywords for predefined categories (for body parts).
        Returns empty list if not a recognized category.
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
    def _parse_metadata(meta_str: str) -> dict:
        """Parse metadata JSON string."""
        try:
            return json.loads(meta_str)
        except:
            return {"label": "unknown", "score": 0, "type": "unknown"}

    @staticmethod
    def _filter_by_keyword(masks: List[torch.Tensor], metadata: List[str], filter_keyword: str) -> tuple:
        """
        Filter masks based on keyword using case-insensitive contains matching.

        Returns: (filtered_masks, filtered_metadata)
        """
        if not filter_keyword or filter_keyword.lower() == "none":
            return [], []

        # Check if this is a category
        category_keywords = FilterMasksNode._get_category_keywords(filter_keyword)

        filtered_masks = []
        filtered_meta = []

        for i, (mask, meta_str) in enumerate(zip(masks, metadata)):
            meta = FilterMasksNode._parse_metadata(meta_str)
            label = meta.get("label", "").lower()

            matched = False

            if category_keywords:
                # Category matching: check if any keyword is contained in the label
                for keyword in category_keywords:
                    if keyword.lower() in label:
                        matched = True
                        break
            else:
                # Direct contains matching
                if filter_keyword.lower() in label:
                    matched = True

            if matched:
                filtered_masks.append(mask)
                filtered_meta.append(meta_str)

        return filtered_masks, filtered_meta

    def filter_masks(
        self,
        all_masks: List[torch.Tensor],
        all_masks_meta: List[str],
        filter_1: str = "none",
        filter_2: str = "none",
        filter_3: str = "",
        filter_4: str = "",
        filter_5: str = "",
    ):
        """Filter masks based on provided filters."""

        print(f"[FILTER MASKS] Input: {len(all_masks)} masks")

        # Build debug log
        debug_log = "=" * 55 + "\n"
        debug_log += "FILTER MASKS UTILITY\n"
        debug_log += "-" * 55 + "\n"
        debug_log += f"Input masks: {len(all_masks)}\n"
        debug_log += f"Input metadata: {len(all_masks_meta)}\n"

        # List available labels
        debug_log += "\nAvailable labels:\n"
        for meta_str in all_masks_meta:
            meta = self._parse_metadata(meta_str)
            label = meta.get("label", "unknown")
            score = meta.get("score", 0)
            mask_type = meta.get("type", "unknown")
            if label != "none" and mask_type != "blank":
                debug_log += f"  - '{label}' ({mask_type}, score: {score:.3f})\n"

        # Initialize filter outputs
        filter_lists = [[], [], [], [], []]
        filter_meta_lists = [[], [], [], [], []]
        filters = [filter_1, filter_2, filter_3, filter_4, filter_5]

        debug_log += f"\n{'='*55}\n"
        debug_log += "APPLYING FILTERS\n"
        debug_log += f"{'-'*55}\n"

        # Apply each filter
        for idx, filter_keyword in enumerate(filters):
            if not filter_keyword or filter_keyword.lower() == "none":
                debug_log += f"Filter {idx+1}: [SKIP] No filter specified\n"
                continue

            debug_log += f"\nFilter {idx+1}: Searching for '{filter_keyword}'\n"

            # Filter masks
            filtered_masks, filtered_meta = self._filter_by_keyword(
                all_masks, all_masks_meta, filter_keyword
            )

            filter_lists[idx] = filtered_masks
            filter_meta_lists[idx] = filtered_meta

            # Log results
            if len(filtered_masks) > 0:
                debug_log += f"  Found {len(filtered_masks)} match(es):\n"
                for meta_str in filtered_meta:
                    meta = self._parse_metadata(meta_str)
                    label = meta.get("label", "unknown")
                    score = meta.get("score", 0)
                    mask_type = meta.get("type", "unknown")
                    debug_log += f"    - '{label}' ({mask_type}, score: {score:.3f})\n"
            else:
                debug_log += f"  [NOT FOUND] No masks matching '{filter_keyword}'\n"

        # Add blank images to empty lists
        blank_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        blank_meta = json.dumps({"label": "none", "score": 0, "type": "blank"}, indent=2)

        for idx, filter_list in enumerate(filter_lists):
            if len(filter_list) == 0:
                filter_lists[idx].append(blank_image)
                filter_meta_lists[idx].append(blank_meta)
                if filters[idx] and filters[idx].lower() != "none":
                    debug_log += f"\nFilter {idx+1}: Added blank image (no matches for '{filters[idx]}')\n"

        # Build filtered JSON summary
        filtered_data = {
            "filters": [],
            "total_input_masks": len(all_masks),
        }

        for idx, (filter_keyword, filter_list) in enumerate(zip(filters, filter_lists)):
            # Count real masks (exclude blank)
            real_count = 0
            if len(filter_list) > 0 and filter_list[0].shape != torch.Size([1, 64, 64, 3]):
                real_count = len(filter_list)

            filtered_data["filters"].append({
                "filter_number": idx + 1,
                "keyword": filter_keyword if filter_keyword and filter_keyword.lower() != "none" else "none",
                "matches": real_count,
            })

        filtered_json = json.dumps(filtered_data, indent=2)

        # Summary
        debug_log += f"\n{'='*55}\n"
        debug_log += "SUMMARY\n"
        debug_log += f"{'-'*55}\n"
        debug_log += f"Input masks: {len(all_masks)}\n"
        debug_log += "\nFiltered outputs:\n"
        for idx, (filter_keyword, filter_list) in enumerate(zip(filters, filter_lists)):
            # Count real masks
            real_count = 0
            if len(filter_list) > 0 and filter_list[0].shape != torch.Size([1, 64, 64, 3]):
                real_count = len(filter_list)

            if not filter_keyword or filter_keyword.lower() == "none":
                debug_log += f"  filter_{idx+1}: [empty] (no filter)\n"
            elif real_count > 0:
                debug_log += f"  filter_{idx+1}: {real_count} mask(s) for '{filter_keyword}'\n"
            else:
                debug_log += f"  filter_{idx+1}: [blank] (no matches for '{filter_keyword}')\n"
        debug_log += f"{'='*55}\n"

        print(debug_log)

        return (
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
            filtered_json,
            debug_log,
        )


# Node registration
NODE_CLASS_MAPPINGS = {
    "FilterMasksNode": FilterMasksNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FilterMasksNode": "Filter Masks",
}
