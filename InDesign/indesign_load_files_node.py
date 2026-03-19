"""
InDesign Load Files Node

File pickers for InDesign template and CSV data source.
Uploads selected files to S3 and outputs presigned URLs.
Image assets referenced in the CSV are auto-uploaded by the Data Merge node.
"""

import os
import hashlib
import mimetypes
import logging

import folder_paths
from ..Photoshop.photoshop_storage import upload_file_to_s3

logger = logging.getLogger(__name__)

INDESIGN_EXTENSIONS = {".indd", ".idml"}
CSV_EXTENSIONS = {".csv", ".tsv", ".txt"}


def _get_input_files(extensions):
    input_dir = folder_paths.get_input_directory()
    return sorted(
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and os.path.splitext(f)[1].lower() in extensions
    )


class InDesignLoadFilesNode:
    @classmethod
    def INPUT_TYPES(s):
        template_files = _get_input_files(INDESIGN_EXTENSIONS)
        csv_files = _get_input_files(CSV_EXTENSIONS)

        return {
            "required": {
                "template_file": (template_files, {"file_upload": True,
                                                   "tooltip": "InDesign template file (.indd or .idml) to use for data merge"}),
                "data_source_file": (csv_files, {"file_upload": True,
                                                  "tooltip": "CSV/TSV data source file containing merge field values"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("template_url", "data_source_url")
    FUNCTION = "upload_files"
    CATEGORY = "api node/InDesign"
    API_NODE = True

    async def upload_files(self, template_file, data_source_file):
        # Upload template
        template_path = folder_paths.get_annotated_filepath(template_file)
        ct, _ = mimetypes.guess_type(template_path)
        if ct is None:
            ct = "application/octet-stream"
        logger.info(f"[InDesign Load] Uploading template: {template_file}")
        template_url = await upload_file_to_s3(template_path, content_type=ct)

        # Upload data source
        csv_path = folder_paths.get_annotated_filepath(data_source_file)
        ct, _ = mimetypes.guess_type(csv_path)
        if ct is None:
            ct = "text/csv"
        logger.info(f"[InDesign Load] Uploading data source: {data_source_file}")
        data_source_url = await upload_file_to_s3(csv_path, content_type=ct)

        return (template_url, data_source_url)

    @classmethod
    def IS_CHANGED(s, template_file, data_source_file):
        hasher = hashlib.sha256()
        for filename in [template_file, data_source_file]:
            file_path = folder_paths.get_annotated_filepath(filename)
            if os.path.isfile(file_path):
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(s, template_file, data_source_file):
        if not folder_paths.exists_annotated_filepath(template_file):
            return f"Template not found: {template_file}"
        if not folder_paths.exists_annotated_filepath(data_source_file):
            return f"Data source not found: {data_source_file}"
        return True
