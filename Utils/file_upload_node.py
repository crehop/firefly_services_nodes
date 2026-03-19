"""
File Upload to S3 Node

Provides a file picker that accepts any file type, uploads the selected file
to AWS S3, and outputs a pre-signed GET URL for use with Adobe APIs.
"""

import os
import hashlib
import mimetypes
import logging
import folder_paths

from ..Photoshop.photoshop_storage import upload_file_to_s3


class FileUploadToS3Node:
    """Upload any file from the input directory to S3 and return a presigned URL."""

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = sorted(
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        )
        return {
            "required": {
                "file": (files, {"file_upload": True, "tooltip": "Select any file to upload to S3 and get a presigned URL"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("presigned_url",)
    FUNCTION = "upload_file"
    CATEGORY = "api node/Firefly Utils"
    API_NODE = True
    OUTPUT_NODE = False

    async def upload_file(self, file):
        # Resolve full path
        file_path = folder_paths.get_annotated_filepath(file)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect MIME type
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            content_type = "application/octet-stream"

        file_size = os.path.getsize(file_path)
        logging.info(
            f"[FileUploadToS3] Uploading '{file}' ({file_size:,} bytes, {content_type})"
        )

        # Upload to S3 and get presigned GET URL
        presigned_url = await upload_file_to_s3(file_path, content_type=content_type)

        logging.info(f"[FileUploadToS3] Upload complete, presigned URL generated")

        return (presigned_url,)

    @classmethod
    def IS_CHANGED(s, file):
        file_path = folder_paths.get_annotated_filepath(file)
        m = hashlib.sha256()
        with open(file_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, file):
        if not folder_paths.exists_annotated_filepath(file):
            return "Invalid file: {}".format(file)
        return True
