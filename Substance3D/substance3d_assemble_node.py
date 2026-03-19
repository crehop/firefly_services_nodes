"""
ComfyUI node for Adobe Substance 3D Assemble endpoint.
POST /v1/scenes/assemble
"""

from __future__ import annotations
import json
import time
import logging

from .substance3d_api import (
    AssembleRequest,
    SceneDescription,
    SceneModels,
    SceneModelImport,
    SceneEnvironment,
    MaterialAssign,
    MountedSource,
    MountedSourceURL,
)
from .substance3d_client import submit_and_poll_s3d
from server import PromptServer

logger = logging.getLogger(__name__)


class Substance3DAssembleNode:
    """Assemble multiple 3D models into a single scene using Adobe Substance 3D API."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_url": ("STRING", {"forceInput": True, "tooltip": "URL of the primary 3D model to assemble"}),
                "output_format": (["glb", "gltf", "fbx", "usdz", "usda", "usdc", "obj"], {"tooltip": "Target 3D file format for the assembled output"}),
                "file_base_name": ("STRING", {"default": "assembled_scene", "tooltip": "Base filename for the assembled output file"}),
            },
            "optional": {
                "model_url_2": ("STRING", {"forceInput": True, "default": "", "tooltip": "URL of a second 3D model to include in the scene"}),
                "model_url_3": ("STRING", {"forceInput": True, "default": "", "tooltip": "URL of a third 3D model to include in the scene"}),
                "model_url_4": ("STRING", {"forceInput": True, "default": "", "tooltip": "URL of a fourth 3D model to include in the scene"}),
                "texture_url": ("STRING", {"forceInput": True, "default": "", "tooltip": "URL of a texture image to apply to the model"}),
                "environment_url": ("STRING", {"forceInput": True, "default": "", "tooltip": "URL of an HDR environment map for the scene"}),
                "material_sbsar_url": ("STRING", {"forceInput": True, "default": "", "tooltip": "URL of an SBSAR material to apply to the model"}),
                "scene_json": ("STRING", {"default": "", "multiline": True, "tooltip": "Raw JSON override for the full scene description"}),
                "meters_per_unit": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 100.0, "step": 0.001, "tooltip": "Scale factor converting model units to meters"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("scene_url", "output_url", "debug_log")
    FUNCTION = "assemble"
    CATEGORY = "api node/Substance 3D"
    API_NODE = True

    async def assemble(
        self,
        model_url,
        output_format,
        file_base_name="assembled_scene",
        model_url_2="",
        model_url_3="",
        model_url_4="",
        texture_url="",
        environment_url="",
        material_sbsar_url="",
        scene_json="",
        meters_per_unit=0.01,
        unique_id=None,
    ):
        start_time = time.time()
        debug_lines = []
        debug_lines.append("=== Substance 3D Assemble ===")

        # ── Collect active URL inputs ────────────────────────────────────
        model_urls_raw = [
            model_url,
            model_url_2 if model_url_2 and model_url_2.strip() else None,
            model_url_3 if model_url_3 and model_url_3.strip() else None,
            model_url_4 if model_url_4 and model_url_4.strip() else None,
        ]
        active_model_urls = [u for u in model_urls_raw if u]

        has_texture = bool(texture_url and texture_url.strip())
        has_env = bool(environment_url and environment_url.strip())
        has_sbsar = bool(material_sbsar_url and material_sbsar_url.strip())

        debug_lines.append(f"Active models: {len(active_model_urls)}")
        debug_lines.append(f"Output format: {output_format}")
        debug_lines.append(f"File base name: {file_base_name}")

        # ── Parse or build scene description ─────────────────────────────
        # We parse scene_json FIRST so we can extract the filenames it
        # references, then build sources with matching filenames.

        if scene_json and scene_json.strip():
            debug_lines.append("Using provided scene JSON override")
            try:
                scene_dict = json.loads(scene_json)
                scene_desc = SceneDescription(**scene_dict)
            except (json.JSONDecodeError, Exception) as e:
                debug_lines.append(f"ERROR parsing scene_json: {e}")
                raise ValueError(f"Invalid scene_json: {e}")

            # Extract filenames referenced in the scene_json so sources
            # use the EXACT same names (the S3D API matches by filename).
            scene_model_files = []
            scene_texture_files = []
            if scene_desc.models and scene_desc.models.imports:
                for imp in scene_desc.models.imports:
                    if imp.file:
                        scene_model_files.append(imp.file)
                    if imp.materialOverrides:
                        for mo in imp.materialOverrides:
                            for key in ("baseColorTexture", "normalTexture",
                                        "metallicRoughnessTexture", "emissiveTexture"):
                                if key in mo and mo[key]:
                                    scene_texture_files.append(mo[key])

            scene_env_file = None
            if scene_desc.environment and scene_desc.environment.file:
                scene_env_file = scene_desc.environment.file

            debug_lines.append(f"Scene references models: {scene_model_files}")
            debug_lines.append(f"Scene references textures: {scene_texture_files}")

            # Map model URLs to scene-referenced filenames (by position)
            model_filename_map = []
            for i, url in enumerate(active_model_urls):
                if i < len(scene_model_files):
                    model_filename_map.append((url, scene_model_files[i]))
                else:
                    model_filename_map.append((url, f"model_{i+1}.{output_format}"))

            # Map texture URL to scene-referenced texture filename
            texture_filename = None
            if has_texture:
                if scene_texture_files:
                    texture_filename = scene_texture_files[0]
                else:
                    texture_filename = "texture.png"

            env_filename = scene_env_file if scene_env_file else "environment.hdr"
        else:
            debug_lines.append("Building scene description from inputs")

            # Use deterministic filenames and build scene to match
            model_filename_map = []
            for i, url in enumerate(active_model_urls):
                model_filename_map.append((url, f"model_{i+1}.{output_format}"))

            texture_filename = "texture.png" if has_texture else None
            env_filename = "environment.hdr"
            sbsar_filename = "material.sbsar"

            # Build scene description with matching filenames
            model_imports = []
            for _url, filename in model_filename_map:
                model_imports.append(SceneModelImport(file=filename))

            models = SceneModels(imports=model_imports) if model_imports else None

            scene_environment = None
            if has_env:
                scene_environment = SceneEnvironment(file=env_filename)

            materials = None
            if has_sbsar:
                materials = [
                    MaterialAssign(
                        materialName="default",
                        material={"sbsar": sbsar_filename},
                        assignByDefault=True,
                    )
                ]

            scene_desc = SceneDescription(
                models=models,
                environment=scene_environment,
                materials=materials,
                metersPerUnit=meters_per_unit,
            )

        # ── Build sources with explicit filenames ────────────────────────
        # The filename field in MountedSourceURL is what the S3D API uses
        # to identify files in its virtual filesystem. It MUST match the
        # filenames referenced in the scene description.
        sources = []

        for url, filename in model_filename_map:
            sources.append(
                MountedSource(
                    url=MountedSourceURL(url=url, filename=filename),
                    mountPoint="/",
                )
            )
            debug_lines.append(f"Model source: {filename} -> {url[:100]}...")

        if has_texture and texture_filename:
            sources.append(
                MountedSource(
                    url=MountedSourceURL(url=texture_url, filename=texture_filename),
                    mountPoint="/",
                )
            )
            debug_lines.append(f"Texture source: {texture_filename} -> {texture_url[:100]}...")

        if has_env:
            sources.append(
                MountedSource(
                    url=MountedSourceURL(url=environment_url, filename=env_filename),
                    mountPoint="/",
                )
            )
            debug_lines.append(f"Environment source: {env_filename}")

        if has_sbsar:
            sbsar_fn = "material.sbsar"
            sources.append(
                MountedSource(
                    url=MountedSourceURL(url=material_sbsar_url, filename=sbsar_fn),
                    mountPoint="/",
                )
            )
            debug_lines.append(f"SBSAR source: {sbsar_fn}")

        debug_lines.append(f"Total sources: {len(sources)}")

        debug_lines.append(f"Meters per unit: {meters_per_unit}")

        # ── Build the request ─────────────────────────────────────────────
        request = AssembleRequest(
            scene=scene_desc,
            sources=sources,
            encoding=output_format,
            fileBaseName=file_base_name,
        )

        # ── Submit and poll ───────────────────────────────────────────────
        debug_lines.append("Submitting assemble job...")
        submit_time = time.time()

        try:
            job_response = await submit_and_poll_s3d(
                endpoint_path="/v1/scenes/assemble",
                request_model=AssembleRequest,
                request_data=request,
                node_id=unique_id,
                estimated_duration=120.0,
            )
        except Exception as e:
            debug_lines.append(f"ERROR: {str(e)}")
            raise

        poll_duration = time.time() - submit_time
        debug_lines.append(f"Job ID: {job_response.id}")
        debug_lines.append(f"Job status: {job_response.status}")
        debug_lines.append(f"Job duration: {poll_duration:.1f}s")

        # ── Extract results ───────────────────────────────────────────────
        scene_url = ""
        output_url = ""

        if job_response.result:
            scene_url = job_response.result.get("sceneUrl", "") or ""

            output_space = job_response.result.get("outputSpace")
            if output_space and output_space.get("files"):
                files = output_space["files"]
                if len(files) > 0 and files[0].get("url"):
                    output_url = files[0]["url"]

            if job_response.result.get("warnings"):
                for w in job_response.result["warnings"]:
                    debug_lines.append(f"Warning: {w}")

        if not scene_url and not output_url:
            debug_lines.append(
                f"WARNING: No sceneUrl or output file in result. "
                f"Full result: {json.dumps(job_response.result, indent=2)}"
            )

        if scene_url:
            debug_lines.append(f"Scene URL: {scene_url[:120]}...")
        if output_url:
            debug_lines.append(f"Output URL: {output_url[:120]}...")

        total_duration = time.time() - start_time
        debug_lines.append(f"Total time: {total_duration:.1f}s")
        debug_lines.append("=== Done ===")

        # Display success notification on the node
        if unique_id:
            PromptServer.instance.send_progress_text(
                f"Assemble complete ({total_duration:.1f}s)", unique_id
            )

        debug_log = "\n".join(debug_lines)
        logger.info(debug_log)

        return (scene_url, output_url, debug_log)
