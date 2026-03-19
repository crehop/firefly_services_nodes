"""
ComfyUI node for Adobe Substance 3D Render Basic endpoint.
POST /v1/scenes/render-basic
"""

from __future__ import annotations
import json
import time
import logging

from .substance3d_api import (
    RenderBasicRequest,
    SimpleSceneDescription,
    SizeOptions,
    AutoFramingOptions,
    BackgroundOptions,
    GroundPlaneOptions,
    RenderExtraOutputs,
    SceneCamera,
    SceneEnvironment,
    MountedSource,
    MountedSourceURL,
)
from .substance3d_client import submit_and_poll_s3d, make_source
from ..apinode_utils import download_url_to_image_tensor
from server import PromptServer

logger = logging.getLogger(__name__)


class Substance3DRenderBasicNode:
    """Render a 3D model using Adobe Substance 3D render-basic API."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_url": ("STRING", {"forceInput": True, "tooltip": "URL of the 3D model file to render"}),
            },
            "optional": {
                "environment_url": ("STRING", {"forceInput": True, "default": "", "tooltip": "URL of an HDR environment map for lighting"}),
                "material_sbsar_url": ("STRING", {"forceInput": True, "default": "", "tooltip": "URL of an SBSAR material to apply to the model"}),
                "width": ("INT", {"default": 1920, "min": 16, "max": 3840, "step": 1, "tooltip": "Output image width in pixels"}),
                "height": ("INT", {"default": 1080, "min": 16, "max": 2304, "step": 1, "tooltip": "Output image height in pixels"}),
                "focal_length": ("FLOAT", {"default": 50.0, "min": 10.0, "max": 1000.0, "step": 0.1, "tooltip": "Camera focal length in millimeters"}),
                "sensor_width": ("FLOAT", {"default": 36.0, "min": 1.0, "max": 100.0, "step": 0.1, "tooltip": "Camera sensor width in millimeters"}),
                "environment_exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1, "tooltip": "Exposure adjustment for the environment lighting"}),
                "environment_rotation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0, "tooltip": "Rotation of the environment map in degrees"}),
                "auto_framing": (["auto", "bounding_cylinder", "frustum_fit"], {"default": "auto", "tooltip": "Algorithm used to automatically frame the model in the camera view"}),
                "auto_framing_zoom": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Zoom multiplier for auto-framing (higher = closer)"}),
                "ground_plane": ("BOOLEAN", {"default": True, "tooltip": "Enable a ground plane beneath the model"}),
                "shadows": ("BOOLEAN", {"default": True, "tooltip": "Enable shadow casting on the ground plane"}),
                "shadow_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Opacity of ground plane shadows"}),
                "reflections": ("BOOLEAN", {"default": False, "tooltip": "Enable reflections on the ground plane"}),
                "reflection_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Opacity of ground plane reflections"}),
                "reflection_roughness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Roughness of ground plane reflections"}),
                "background_color_r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Background color red channel (0-1)"}),
                "background_color_g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Background color green channel (0-1)"}),
                "background_color_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Background color blue channel (0-1)"}),
                "export_material_ids": ("BOOLEAN", {"default": False, "tooltip": "Export a material ID pass as extra output"}),
                "export_object_ids": ("BOOLEAN", {"default": False, "tooltip": "Export an object ID pass as extra output"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "debug_log")
    FUNCTION = "render"
    CATEGORY = "api node/Substance 3D"
    API_NODE = True

    async def render(
        self,
        model_url,
        environment_url="",
        material_sbsar_url="",
        width=1920,
        height=1080,
        focal_length=50.0,
        sensor_width=36.0,
        environment_exposure=0.0,
        environment_rotation=0.0,
        auto_framing="auto",
        auto_framing_zoom=1.0,
        ground_plane=True,
        shadows=True,
        shadow_opacity=1.0,
        reflections=False,
        reflection_opacity=1.0,
        reflection_roughness=0.0,
        background_color_r=0.0,
        background_color_g=0.0,
        background_color_b=0.0,
        export_material_ids=False,
        export_object_ids=False,
        unique_id=None,
    ):
        start_time = time.time()
        debug_lines = []
        debug_lines.append("=== Substance 3D Render Basic ===")

        # ── Build sources list ────────────────────────────────────────────
        sources = [
            MountedSource(
                url=MountedSourceURL(url=model_url, filename="model.glb"),
                mountPoint="/",
            )
        ]
        debug_lines.append(f"Model URL: {model_url[:120]}...")

        # Environment file reference (for scene description)
        env_file = None
        if environment_url and environment_url.strip():
            sources.append(
                MountedSource(
                    url=MountedSourceURL(url=environment_url, filename="environment.hdr"),
                    mountPoint="/",
                )
            )
            env_file = "environment.hdr"
            debug_lines.append(f"Environment URL: {environment_url[:120]}...")

        # Material SBSAR file reference
        sbsar_file = None
        if material_sbsar_url and material_sbsar_url.strip():
            sources.append(
                MountedSource(
                    url=MountedSourceURL(url=material_sbsar_url, filename="material.sbsar"),
                    mountPoint="/",
                )
            )
            sbsar_file = "material.sbsar"
            debug_lines.append(f"Material SBSAR URL: {material_sbsar_url[:120]}...")

        debug_lines.append(f"Sources count: {len(sources)}")

        # ── Scene environment ─────────────────────────────────────────────
        scene_environment = None
        if env_file or environment_exposure != 0.0 or environment_rotation != 0.0:
            scene_environment = SceneEnvironment(
                file=env_file,
                exposure=environment_exposure if environment_exposure != 0.0 else None,
                rotation=environment_rotation if environment_rotation != 0.0 else None,
            )

        # ── Scene camera ──────────────────────────────────────────────────
        scene_camera = SceneCamera(
            focal=focal_length,
            sensorWidth=sensor_width,
        )

        # ── Material overrides ────────────────────────────────────────────
        material_overrides = None
        if sbsar_file:
            material_overrides = [
                {
                    "assignByDefault": True,
                    "material": {
                        "sbsar": sbsar_file,
                    },
                }
            ]

        # ── Simple scene description ──────────────────────────────────────
        scene = SimpleSceneDescription(
            modelFile="model.glb",
            camera=scene_camera,
            environment=scene_environment,
            materialOverrides=material_overrides,
        )

        # ── Size options ──────────────────────────────────────────────────
        size = SizeOptions(width=width, height=height)

        # ── Auto framing ──────────────────────────────────────────────────
        auto_framing_opts = AutoFramingOptions(
            algorithm=auto_framing,
            zoom=auto_framing_zoom if auto_framing_zoom != 1.0 else None,
        )

        # ── Background ────────────────────────────────────────────────────
        background = BackgroundOptions(
            color=[background_color_r, background_color_g, background_color_b],
        )

        # ── Ground plane ──────────────────────────────────────────────────
        ground_plane_opts = GroundPlaneOptions(
            enable=ground_plane,
            shadows=shadows,
            shadowsOpacity=shadow_opacity,
            reflections=reflections,
            reflectionsOpacity=reflection_opacity,
            reflectionsRoughness=reflection_roughness,
        )

        # ── Extra outputs ─────────────────────────────────────────────────
        extra_outputs = None
        if export_material_ids or export_object_ids:
            extra_outputs = RenderExtraOutputs(
                exportMaterialIds=export_material_ids if export_material_ids else None,
                exportObjectIds=export_object_ids if export_object_ids else None,
            )

        # ── Build the request ─────────────────────────────────────────────
        request = RenderBasicRequest(
            scene=scene,
            sources=sources,
            size=size,
            autoFraming=auto_framing_opts,
            background=background,
            groundPlane=ground_plane_opts,
            extraOutputs=extra_outputs,
        )

        debug_lines.append(f"Size: {width}x{height}")
        debug_lines.append(f"Focal length: {focal_length}mm, Sensor width: {sensor_width}mm")
        debug_lines.append(f"Auto framing: {auto_framing} (zoom={auto_framing_zoom})")
        debug_lines.append(f"Ground plane: {ground_plane}, Shadows: {shadows} (opacity={shadow_opacity})")
        debug_lines.append(f"Reflections: {reflections} (opacity={reflection_opacity}, roughness={reflection_roughness})")
        debug_lines.append(f"Background color: ({background_color_r}, {background_color_g}, {background_color_b})")
        if environment_exposure != 0.0 or environment_rotation != 0.0:
            debug_lines.append(f"Environment: exposure={environment_exposure}, rotation={environment_rotation}")

        # ── Submit and poll ────────────────────────────────────────────────
        debug_lines.append("Submitting render job...")
        submit_time = time.time()

        try:
            job_response = await submit_and_poll_s3d(
                endpoint_path="/v1/scenes/render-basic",
                request_model=RenderBasicRequest,
                request_data=request,
                node_id=unique_id,
                estimated_duration=60.0,
            )
        except Exception as e:
            debug_lines.append(f"ERROR: {str(e)}")
            raise

        poll_duration = time.time() - submit_time
        debug_lines.append(f"Job ID: {job_response.id}")
        debug_lines.append(f"Job status: {job_response.status}")
        debug_lines.append(f"Job duration: {poll_duration:.1f}s")

        # ── Extract render URL from result ─────────────────────────────────
        render_url = None
        if job_response.result:
            render_url = job_response.result.get("renderUrl")
            if job_response.result.get("warnings"):
                for w in job_response.result["warnings"]:
                    debug_lines.append(f"Warning: {w}")

        if not render_url:
            error_msg = f"No renderUrl in job result. Full result: {json.dumps(job_response.result, indent=2)}"
            debug_lines.append(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)

        debug_lines.append(f"Render URL: {render_url[:120]}...")

        # ── Download rendered image ────────────────────────────────────────
        debug_lines.append("Downloading rendered image...")
        try:
            image_tensor = await download_url_to_image_tensor(render_url, timeout=120)
        except Exception as e:
            debug_lines.append(f"ERROR downloading image: {str(e)}")
            logger.error(f"[S3D Render Basic] Failed to download rendered image: {e}")
            raise

        total_duration = time.time() - start_time
        debug_lines.append(f"Image shape: {list(image_tensor.shape)}")
        debug_lines.append(f"Total time: {total_duration:.1f}s")
        debug_lines.append("=== Done ===")

        # Display success notification on the node
        if unique_id:
            PromptServer.instance.send_progress_text(
                f"Render complete ({total_duration:.1f}s)", unique_id
            )

        debug_log = "\n".join(debug_lines)
        logger.info(debug_log)

        return (image_tensor, debug_log)
