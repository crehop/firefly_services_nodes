"""
S3D Load Files Node

Provides individual file pickers for each Substance 3D file type.
Each picker has its own upload button. Uploads selected files to S3
and outputs presigned URLs for use with S3D API nodes.
"""

import os
import hashlib
import mimetypes
import logging
from pathlib import Path

import folder_paths
from ..Photoshop.photoshop_storage import upload_file_to_s3

logger = logging.getLogger(__name__)


def _normalize_path(path):
    return path.replace("\\", "/")


def _get_files_from_subdir(subdir, extensions):
    """Get files from input/<subdir> directory, filtered by extensions."""
    input_dir = os.path.join(folder_paths.get_input_directory(), subdir)
    os.makedirs(input_dir, exist_ok=True)
    input_path = Path(input_dir)
    base_path = Path(folder_paths.get_input_directory())
    files = [
        _normalize_path(str(fp.relative_to(base_path)))
        for fp in input_path.rglob("*")
        if fp.suffix.lower() in extensions
    ]
    return sorted(files)


def _get_input_files(extensions):
    """Get files from input directory root, filtered by extensions."""
    input_dir = folder_paths.get_input_directory()
    files = sorted(
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and os.path.splitext(f)[1].lower() in extensions
    )
    return files


def _get_image_files():
    """Get image files from input directory."""
    input_dir = folder_paths.get_input_directory()
    all_files = sorted(
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    )
    return folder_paths.filter_files_content_types(all_files, ["image"])


# File type groups
MODEL_EXTENSIONS = {".gltf", ".glb", ".obj", ".fbx", ".stl", ".usdz", ".usda", ".usdc"}
MATERIAL_EXTENSIONS = {".sbsar"}
ENVIRONMENT_EXTENSIONS = {".hdr", ".exr"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class Substance3DLoadFilesNode:
    @classmethod
    def INPUT_TYPES(s):
        # 3D models from input/3d/ (same as Load 3D node)
        model_files = _get_files_from_subdir("3d", MODEL_EXTENSIONS)

        # Materials from input root
        material_files = ["none"] + _get_input_files(MATERIAL_EXTENSIONS)

        # Environments from input root
        environment_files = ["none"] + _get_input_files(ENVIRONMENT_EXTENSIONS)

        # Images from input root
        image_files = ["none"] + _get_image_files()

        return {
            "required": {
                "model_file": (model_files, {"file_upload": True, "tooltip": "3D model file to upload (glTF, FBX, OBJ, USD, etc.)"}),
            },
            "optional": {
                "material_file": (material_files, {"default": "none", "file_upload": True, "tooltip": "SBSAR material file to upload and apply to the model"}),
                "environment_file": (environment_files, {"default": "none", "file_upload": True, "tooltip": "HDR or EXR environment map for scene lighting"}),
                "texture_file": (image_files, {"default": "none", "file_upload": True, "tooltip": "Texture image to upload for use with the model"}),
                "background_image": (image_files, {"default": "none", "file_upload": True, "tooltip": "Background image for compositing behind the render"}),
                "style_image": (image_files, {"default": "none", "file_upload": True, "tooltip": "Style reference image for AI background generation"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_url", "material_url", "environment_url", "texture_url", "background_url", "style_url")
    FUNCTION = "upload_files"
    CATEGORY = "api node/Substance 3D"
    API_NODE = True

    async def upload_files(
        self,
        model_file,
        material_file="none",
        environment_file="none",
        texture_file="none",
        background_image="none",
        style_image="none",
    ):
        results = {}

        # Upload 3D model (always required)
        model_path = folder_paths.get_annotated_filepath(model_file)
        ct, _ = mimetypes.guess_type(model_path)
        if ct is None:
            ct = "application/octet-stream"
        logger.info(f"[S3D Load] Uploading model: {model_file} ({ct})")
        results["model_url"] = await upload_file_to_s3(model_path, content_type=ct)

        # Upload optional files
        optional_files = {
            "material_url": material_file,
            "environment_url": environment_file,
            "texture_url": texture_file,
            "background_url": background_image,
            "style_url": style_image,
        }

        for key, filename in optional_files.items():
            if not filename or filename == "none":
                results[key] = ""
                continue

            file_path = folder_paths.get_annotated_filepath(filename)
            ct, _ = mimetypes.guess_type(file_path)
            if ct is None:
                ct = "application/octet-stream"
            logger.info(f"[S3D Load] Uploading {key}: {filename} ({ct})")
            results[key] = await upload_file_to_s3(file_path, content_type=ct)

        return (
            results["model_url"],
            results["material_url"],
            results["environment_url"],
            results["texture_url"],
            results["background_url"],
            results["style_url"],
        )

    @classmethod
    def IS_CHANGED(s, model_file, **kwargs):
        hasher = hashlib.sha256()
        model_path = folder_paths.get_annotated_filepath(model_file)
        if os.path.isfile(model_path):
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        for key in ("material_file", "environment_file", "texture_file", "background_image", "style_image"):
            filename = kwargs.get(key, "none")
            if not filename or filename == "none":
                continue
            file_path = folder_paths.get_annotated_filepath(filename)
            if os.path.isfile(file_path):
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(s, model_file, **kwargs):
        if not folder_paths.exists_annotated_filepath(model_file):
            return f"Model file not found: {model_file}"
        for key in ("material_file", "environment_file", "texture_file", "background_image", "style_image"):
            filename = kwargs.get(key, "none")
            if not filename or filename == "none":
                continue
            if not folder_paths.exists_annotated_filepath(filename):
                return f"File not found: {filename}"
        return True
