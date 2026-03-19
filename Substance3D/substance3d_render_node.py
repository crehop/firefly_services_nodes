"""
ComfyUI node for Substance 3D full scene render (POST /v1/scenes/render).

Builds a SceneDescription from inputs (or accepts raw JSON override),
submits the render job, polls for completion, and returns the rendered image.
"""

import json
import logging
from .substance3d_api import (
    RenderSceneRequest, SceneDescription, SceneCamera, SceneEnvironment,
    SceneModels, SceneModelImport, MaterialAssign, SizeOptions,
    AutoFramingOptions, BackgroundOptions, GroundPlaneOptions,
    RenderExtraOutputs, MountedSource, MountedSourceURL,
)
from .substance3d_client import submit_and_poll_s3d
from ..apinode_utils import download_url_to_image_tensor
from server import PromptServer

logger = logging.getLogger(__name__)


class Substance3DRenderNode:
    """Full Substance 3D scene render node (POST /v1/scenes/render)."""

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
                "meters_per_unit": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 100.0, "step": 0.001, "tooltip": "Scale factor converting model units to meters"}),
                "auto_framing": (["auto", "bounding_cylinder", "frustum_fit"], {"tooltip": "Algorithm used to automatically frame the model in the camera view"}),
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
                "camera_name": ("STRING", {"default": "", "tooltip": "Name of a specific camera in the scene to render from"}),
                "scene_json": ("STRING", {"default": "", "multiline": True, "tooltip": "Raw JSON override for the full scene description"}),
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
        meters_per_unit=0.01,
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
        camera_name="",
        scene_json="",
        unique_id=None,
    ):
        log_lines = []

        def log(msg):
            logger.info(f"[S3D Render] {msg}")
            log_lines.append(msg)

        log(f"Starting full scene render ({width}x{height})")

        # ── Build sources list ────────────────────────────────────────────
        sources = [
            MountedSource(url=MountedSourceURL(url=model_url), mountPoint="/"),
        ]
        log(f"Source 0 (model): {model_url[:120]}...")

        has_environment = environment_url and environment_url.strip()
        if has_environment:
            sources.append(
                MountedSource(url=MountedSourceURL(url=environment_url), mountPoint="/"),
            )
            log(f"Source {len(sources)-1} (environment): {environment_url[:120]}...")

        has_material = material_sbsar_url and material_sbsar_url.strip()
        if has_material:
            sources.append(
                MountedSource(url=MountedSourceURL(url=material_sbsar_url), mountPoint="/"),
            )
            log(f"Source {len(sources)-1} (material): {material_sbsar_url[:120]}...")

        # ── Build or parse SceneDescription ───────────────────────────────
        if scene_json and scene_json.strip():
            log("Using raw scene_json override")
            try:
                scene_dict = json.loads(scene_json)
                scene = SceneDescription(**scene_dict)
            except Exception as e:
                raise ValueError(f"Failed to parse scene_json: {e}")
        else:
            # Camera
            camera = SceneCamera(
                focal=focal_length,
                sensorWidth=sensor_width,
            )

            # Environment
            environment = None
            if has_environment:
                env_kwargs = {}
                if environment_exposure != 0.0:
                    env_kwargs["exposure"] = environment_exposure
                if environment_rotation != 0.0:
                    env_kwargs["rotation"] = environment_rotation
                environment = SceneEnvironment(**env_kwargs)

            # Models — import the model file (API resolves from sources)
            models = SceneModels(
                imports=[SceneModelImport(file="model.glb")],
            )

            # Materials
            materials = None
            if has_material:
                materials = [
                    MaterialAssign(
                        materialName="*",
                        material={"sbs": {"sbsar": "material.sbsar"}},
                        assignByDefault=True,
                    ),
                ]

            scene = SceneDescription(
                camera=camera,
                environment=environment,
                models=models,
                materials=materials,
                metersPerUnit=meters_per_unit,
            )

        log(f"Scene description built: camera focal={focal_length}, sensor={sensor_width}")
        log(f"metersPerUnit={meters_per_unit}, environment={'yes' if has_environment else 'no'}, material={'yes' if has_material else 'no'}")

        # ── Render options ────────────────────────────────────────────────
        size = SizeOptions(width=width, height=height)

        auto_framing_opts = AutoFramingOptions(
            algorithm=auto_framing,
            zoom=auto_framing_zoom if auto_framing_zoom != 1.0 else None,
        )

        background = BackgroundOptions(
            color=[background_color_r, background_color_g, background_color_b],
        )

        ground_plane_opts = GroundPlaneOptions(
            enable=ground_plane,
            shadows=shadows,
            shadowsOpacity=shadow_opacity,
            reflections=reflections,
            reflectionsOpacity=reflection_opacity,
            reflectionsRoughness=reflection_roughness,
        )

        extra_outputs = None
        if export_material_ids or export_object_ids:
            extra_outputs = RenderExtraOutputs(
                exportMaterialIds=export_material_ids if export_material_ids else None,
                exportObjectIds=export_object_ids if export_object_ids else None,
            )

        # ── Assemble request ──────────────────────────────────────────────
        request = RenderSceneRequest(
            scene=scene,
            sources=sources,
            size=size,
            autoFraming=auto_framing_opts,
            background=background,
            groundPlane=ground_plane_opts,
            extraOutputs=extra_outputs,
            cameraName=camera_name if camera_name and camera_name.strip() else None,
        )

        # Log the submitted JSON
        request_json = request.model_dump(exclude_none=True)
        log(f"Request JSON:\n{json.dumps(request_json, indent=2)}")

        # ── Submit and poll ───────────────────────────────────────────────
        log("Submitting render job to /v1/scenes/render ...")
        result = await submit_and_poll_s3d(
            endpoint_path="/v1/scenes/render",
            request_model=RenderSceneRequest,
            request_data=request,
            node_id=unique_id,
            estimated_duration=120.0,
        )

        log(f"Job ID: {result.id}")
        log(f"Job status: {result.status}")

        # ── Extract render URL ────────────────────────────────────────────
        render_url = None
        if result.result:
            render_url = result.result.get("renderUrl")
            log(f"Result payload: {json.dumps(result.result, indent=2)}")

        if not render_url:
            error_msg = f"No renderUrl in job result. Status: {result.status}, Error: {result.error}"
            log(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)

        log(f"Render URL: {render_url}")

        # ── Download rendered image ───────────────────────────────────────
        log("Downloading rendered image ...")
        image_tensor = await download_url_to_image_tensor(render_url)
        log(f"Image tensor shape: {image_tensor.shape}")

        # Display success notification on the node
        if unique_id:
            PromptServer.instance.send_progress_text(
                f"Render complete - {width}x{height}", unique_id
            )

        debug_log = "\n".join(log_lines)
        return (image_tensor, debug_log)
