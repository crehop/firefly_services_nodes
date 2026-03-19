"""
InDesign Data Merge Node

Sends an InDesign template + CSV data source to the Adobe InDesign
Data Merge API. Image assets referenced in the CSV (@image column)
are uploaded alongside the template. Returns result images.
"""

import logging
import json
import aiohttp
import io
import os
import numpy as np
import torch
from PIL import Image
from typing import Optional

import folder_paths
from .indesign_api import (
    DataMergeRequest,
    DataMergeParams,
    Asset,
    AssetSource,
    OutputAsset,
    OutputDestination,
)
from .indesign_client import submit_and_poll_indesign
from ..Photoshop.photoshop_storage import upload_file_to_s3, generate_output_presigned_url

logger = logging.getLogger(__name__)


def _parse_csv_image_refs(csv_path):
    """Parse CSV/TSV to find @image column values and return unique filenames."""
    import csv
    filenames = set()
    try:
        # Try UTF-16 first (the datasource.csv is UTF-16 LE with BOM)
        for encoding in ('utf-16', 'utf-8', 'utf-8-sig', 'latin-1'):
            try:
                with open(csv_path, 'r', encoding=encoding) as f:
                    # Detect delimiter
                    sample = f.read(2048)
                    f.seek(0)
                    dialect = csv.Sniffer().sniff(sample, delimiters='\t,')
                    reader = csv.DictReader(f, dialect=dialect)
                    for row in reader:
                        for key, val in row.items():
                            if key and key.strip().startswith('@') and val:
                                # Strip leading ./ from paths like ./can_01.png
                                fname = val.strip().lstrip('.').lstrip('/')
                                if fname:
                                    filenames.add(fname)
                    break
            except (UnicodeDecodeError, UnicodeError):
                continue
    except Exception as e:
        logger.warning(f"[InDesign] Could not parse CSV for image refs: {e}")
    return sorted(filenames)


class InDesignDataMergeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "template_url": ("STRING", {"forceInput": True}),
                "data_source_url": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "output_format": (["image/png", "image/jpeg", "application/pdf"], {"default": "image/png"}),
                "output_base_name": ("STRING", {"default": "image_"}),
                "template_filename": ("STRING", {"default": "web_template.indd"}),
                "data_source_filename": ("STRING", {"default": "datasource.csv"}),
                "num_outputs": ("INT", {"default": 16, "min": 1, "max": 50,
                                        "tooltip": "Number of output slots to pre-generate (should match expected output pages)"}),
                "auto_upload_csv_images": ("BOOLEAN", {"default": True,
                                                       "tooltip": "Auto-detect and upload images referenced in the CSV @image column"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "debug_log")
    FUNCTION = "merge_data"
    CATEGORY = "api node/InDesign"
    API_NODE = True
    OUTPUT_IS_LIST = (True, False)

    async def merge_data(
        self,
        template_url,
        data_source_url,
        output_format="image/png",
        output_base_name="image_",
        template_filename="web_template.indd",
        data_source_filename="datasource.csv",
        num_outputs=16,
        auto_upload_csv_images=True,
        unique_id=None,
    ):
        debug_lines = []

        # Build assets array
        assets = []

        # Template
        assets.append(Asset(
            destination=template_filename,
            source=AssetSource(url=template_url),
        ))
        debug_lines.append(f"Template: {template_filename}")

        # Data source
        assets.append(Asset(
            destination=data_source_filename,
            source=AssetSource(url=data_source_url),
        ))
        debug_lines.append(f"Data source: {data_source_filename}")

        # Auto-detect and upload images from CSV
        if auto_upload_csv_images:
            # Find the CSV file in ComfyUI's input directory to parse it
            input_dir = folder_paths.get_input_directory()
            # Try to find a matching CSV file
            csv_local = None
            for candidate in (data_source_filename, data_source_filename.replace('.csv', '.tsv')):
                path = os.path.join(input_dir, candidate)
                if os.path.isfile(path):
                    csv_local = path
                    break

            if csv_local:
                image_refs = _parse_csv_image_refs(csv_local)
                debug_lines.append(f"CSV image references found: {image_refs}")

                for img_filename in image_refs:
                    img_path = os.path.join(input_dir, img_filename)
                    if os.path.isfile(img_path):
                        import mimetypes
                        ct, _ = mimetypes.guess_type(img_path)
                        if ct is None:
                            ct = "application/octet-stream"
                        debug_lines.append(f"Uploading CSV-referenced image: {img_filename}")
                        try:
                            img_url = await upload_file_to_s3(img_path, content_type=ct)
                            assets.append(Asset(
                                destination=img_filename,
                                source=AssetSource(url=img_url),
                            ))
                        except Exception as e:
                            logger.warning(f"[InDesign] Failed to upload CSV image '{img_filename}': {e}")
                            debug_lines.append(f"Warning: Failed to upload '{img_filename}': {e}, skipping")
                    else:
                        debug_lines.append(f"Warning: CSV references '{img_filename}' but not found in input/")
            else:
                debug_lines.append(f"Warning: Could not find {data_source_filename} locally to parse image refs")

        # Determine file extension
        ext_map = {"image/png": "png", "image/jpeg": "jpg", "application/pdf": "pdf"}
        ext = ext_map.get(output_format, "png")

        # Generate pre-signed PUT URLs for outputs
        output_specs = []
        output_keys = []
        for i in range(1, num_outputs + 1):
            output_source = f"output/{output_base_name}{i}.{ext}"
            put_url, s3_key = await generate_output_presigned_url(ext)
            output_specs.append(OutputAsset(
                destination=OutputDestination(url=put_url, storageType="AWS"),
                source=output_source,
            ))
            output_keys.append(s3_key)

        debug_lines.append(f"Output slots: {num_outputs} ({ext})")

        # Build request
        params = DataMergeParams(
            dataSource=data_source_filename,
            outputFolderPath="output",
            outputFileBaseString=output_base_name,
            outputMediaType=output_format,
            targetDocument=template_filename,
        )

        request = DataMergeRequest(
            assets=assets,
            params=params,
            outputs=output_specs,
        )

        debug_lines.append(f"\nRequest JSON:\n{request.model_dump_json(indent=2)}")
        debug_lines.append("\nSubmitting Data Merge job...")

        response = await submit_and_poll_indesign(request, node_id=unique_id)
        debug_lines.append(f"Job completed! Status: {response.status}")

        # Collect download URLs
        download_urls = []
        if response.outputs:
            for out in response.outputs:
                url = None
                if out.destination and out.destination.url:
                    url = out.destination.url
                if url:
                    download_urls.append(url)

        # Fallback: generate GET URLs from our PUT keys
        if not download_urls:
            debug_lines.append("No output URLs in response, generating GET URLs from S3 keys...")
            from ..Photoshop.photoshop_storage import generate_download_url
            for s3_key in output_keys:
                try:
                    get_url = await generate_download_url(s3_key)
                    download_urls.append(get_url)
                except Exception as e:
                    debug_lines.append(f"  Failed for {s3_key}: {e}")

        debug_lines.append(f"Download URLs: {len(download_urls)}")

        # Download images
        images = []
        if download_urls and output_format.startswith("image/"):
            async with aiohttp.ClientSession() as session:
                for i, url in enumerate(download_urls):
                    try:
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                data = await resp.read()
                                if len(data) < 100:
                                    debug_lines.append(f"  [{i+1}] Skipped (empty/tiny response)")
                                    continue
                                pil_img = Image.open(io.BytesIO(data)).convert("RGB")
                                np_img = np.array(pil_img).astype(np.float32) / 255.0
                                tensor = torch.from_numpy(np_img).unsqueeze(0)
                                images.append(tensor)
                                debug_lines.append(f"  [{i+1}] Downloaded: {pil_img.size}")
                            elif resp.status == 404:
                                debug_lines.append(f"  [{i+1}] Not found (fewer outputs than expected)")
                            else:
                                debug_lines.append(f"  [{i+1}] HTTP {resp.status}")
                    except Exception as e:
                        debug_lines.append(f"  [{i+1}] Error: {e}")

        if not images:
            images = [torch.zeros(1, 64, 64, 3)]
            debug_lines.append("Warning: No images downloaded, returning placeholder.")

        debug_log = "\n".join(debug_lines)
        logger.info(f"[InDesign] Data Merge complete: {len(images)} images")

        return (images, debug_log)
