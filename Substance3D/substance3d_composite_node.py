"""
ComfyUI node for Adobe Substance 3D Composite endpoint.

Renders a 3D object and generates a Firefly AI background via
POST /v1/composites/compose on the Substance 3D API.
"""

import torch
import logging
from .substance3d_api import (
    CompositeRequest, OutputSize, ComposeEnvironment, ComposeSceneDetails,
    SceneCamera, MountedSource, MountedSourceURL,
)
from .substance3d_client import submit_and_poll_s3d
from ..apinode_utils import download_url_to_image_tensor
from server import PromptServer

logger = logging.getLogger(__name__)


def _blank_image() -> torch.Tensor:
    """Return a 1x1 black image tensor [1, 1, 1, 3]."""
    return torch.zeros(1, 1, 1, 3, dtype=torch.float32)


class Substance3DCompositeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_url": ("STRING", {"forceInput": True, "tooltip": "URL of the 3D model to composite into a scene"}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "Text prompt describing the desired background scene"}),
                "hero_asset": ("STRING", {"default": "", "tooltip": "Name or description of the hero product asset"}),
            },
            "optional": {
                "environment_url": ("STRING", {"forceInput": True, "default": "", "tooltip": "URL of an HDR environment map for lighting"}),
                "style_image_url": ("STRING", {"forceInput": True, "default": "", "tooltip": "URL of a style reference image for the AI background"}),
                "content_class": (["photo", "art"], {"default": "photo", "tooltip": "Style class for the generated background"}),
                "model_version": (["image4_ultra", "image4_standard", "image3_fast"], {"default": "image4_ultra", "tooltip": "Firefly model version for background generation"}),
                "num_variations": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1, "tooltip": "Number of output variations to generate"}),
                "width": ("INT", {"default": 2688, "min": 1, "max": 2688, "step": 1, "tooltip": "Output image width in pixels"}),
                "height": ("INT", {"default": 1536, "min": 1, "max": 2688, "step": 1, "tooltip": "Output image height in pixels"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "Random seed for reproducible results (0 = random)"}),
                "camera_name": ("STRING", {"default": "", "tooltip": "Name of a specific camera in the scene to render from"}),
                "enable_ground_plane": ("BOOLEAN", {"default": False, "tooltip": "Enable a ground plane beneath the model"}),
                "environment_exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1, "tooltip": "Exposure adjustment for the environment lighting"}),
                "focal_length": ("FLOAT", {"default": 50.0, "min": 10.0, "max": 1000.0, "step": 0.1, "tooltip": "Camera focal length in millimeters"}),
                "lighting_seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "Seed for lighting variation (0 = random)"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("composite_image", "background_image", "mask_image", "debug_log")
    FUNCTION = "compose"
    CATEGORY = "api node/Substance 3D"
    API_NODE = True

    async def compose(
        self,
        model_url,
        prompt,
        hero_asset="",
        environment_url="",
        style_image_url="",
        content_class="photo",
        model_version="image4_ultra",
        num_variations=1,
        width=2688,
        height=1536,
        seed=0,
        camera_name="",
        enable_ground_plane=False,
        environment_exposure=0.0,
        focal_length=50.0,
        lighting_seed=0,
        unique_id=None,
    ):
        debug_lines = []
        debug_lines.append(f"[S3D Composite] Starting compose job")

        try:
            # Coerce numeric types in case of widget misalignment
            width = int(width) if width else 2688
            height = int(height) if height else 1536
            seed = int(seed) if seed else 0
            num_variations = int(num_variations) if num_variations else 1
            lighting_seed = int(lighting_seed) if lighting_seed else 0
            environment_exposure = float(environment_exposure) if environment_exposure else 0.0
            focal_length = float(focal_length) if focal_length else 50.0

            debug_lines.append(f"  Prompt: {prompt}")
            debug_lines.append(f"  Hero Asset: {hero_asset}")
            debug_lines.append(f"  Model Version: {model_version}")
            debug_lines.append(f"  Content Class: {content_class}")
            debug_lines.append(f"  Size: {width}x{height}")
            debug_lines.append(f"  Seed: {seed}")
            debug_lines.append(f"  Num Variations: {num_variations}")
            debug_lines.append(f"  Camera Name: {camera_name or '(auto)'}")
            debug_lines.append(f"  Ground Plane: {enable_ground_plane}")
            debug_lines.append(f"  Environment Exposure: {environment_exposure}")
            debug_lines.append(f"  Focal Length: {focal_length}")
            debug_lines.append(f"  Lighting Seed: {lighting_seed}")
            debug_lines.append(f"  Model URL: {model_url[:100]}...")

            # ── Build sources list ────────────────────────────────────────
            sources = [
                MountedSource(
                    url=MountedSourceURL(url=model_url),
                    mountPoint="/",
                )
            ]

            has_environment = environment_url and environment_url.strip()
            has_style_image = style_image_url and style_image_url.strip()

            if has_environment:
                sources.append(MountedSource(
                    url=MountedSourceURL(url=environment_url),
                    mountPoint="/",
                ))
                debug_lines.append(f"  Environment URL: provided")

            if has_style_image:
                sources.append(MountedSource(
                    url=MountedSourceURL(url=style_image_url),
                    mountPoint="/",
                ))
                debug_lines.append(f"  Style Image URL: provided")

            # ── Build request ─────────────────────────────────────────────
            request_kwargs = dict(
                sources=sources,
                prompt=prompt,
                heroAsset=hero_asset,
                contentClass=content_class,
                modelVersion=model_version,
                numVariations=num_variations,
                size=OutputSize(width=width, height=height),
                enableGroundPlane=enable_ground_plane,
            )

            # Camera name (from scene describe)
            if camera_name and camera_name.strip():
                request_kwargs["cameraName"] = camera_name.strip()

            # Seeds (only if seed > 0)
            if seed > 0:
                request_kwargs["seeds"] = [seed]

            # Lighting seeds (only if lighting_seed > 0)
            if lighting_seed > 0:
                request_kwargs["lightingSeeds"] = [lighting_seed]

            # Environment exposure
            if environment_exposure != 0.0:
                request_kwargs["environmentExposure"] = environment_exposure

            # Environment file reference
            if has_environment:
                request_kwargs["environment"] = ComposeEnvironment(file="environment.hdr")

            # Style image file reference
            if has_style_image:
                request_kwargs["styleImage"] = "style_image.jpg"

            # Scene camera focal length (only if non-default)
            if focal_length != 50.0:
                request_kwargs["scene"] = ComposeSceneDetails(
                    camera=SceneCamera(focal=focal_length)
                )

            request_data = CompositeRequest(**request_kwargs)

            # ── Submit and poll ───────────────────────────────────────────
            job_response = await submit_and_poll_s3d(
                endpoint_path="/v1/composites/compose",
                request_model=CompositeRequest,
                request_data=request_data,
                node_id=unique_id,
                estimated_duration=120.0,
            )

            debug_lines.append(f"  Job ID: {job_response.id}")
            debug_lines.append(f"  Job Status: {job_response.status}")

            # ── Parse results ─────────────────────────────────────────────
            result_dict = job_response.result or {}
            outputs_list = result_dict.get("outputs", [])

            if not outputs_list:
                debug_lines.append("  WARNING: No outputs returned from job")
                return (_blank_image(), _blank_image(), _blank_image(), "\n".join(debug_lines))

            output = outputs_list[0]
            composite_url = None
            background_url = None
            mask_url = None

            # Extract image URLs from the output dict
            image_data = output.get("image")
            if image_data and isinstance(image_data, dict):
                composite_url = image_data.get("url")

            bg_data = output.get("backgroundImage")
            if bg_data and isinstance(bg_data, dict):
                background_url = bg_data.get("url")

            mask_data = output.get("maskImage")
            if mask_data and isinstance(mask_data, dict):
                mask_url = mask_data.get("url")

            debug_lines.append(f"  Composite URL: {composite_url or 'N/A'}")
            debug_lines.append(f"  Background URL: {background_url or 'N/A'}")
            debug_lines.append(f"  Mask URL: {mask_url or 'N/A'}")

            # ── Download images ───────────────────────────────────────────
            if composite_url:
                try:
                    composite_tensor = await download_url_to_image_tensor(composite_url)
                except Exception as e:
                    logger.error(f"Failed to download composite image: {e}")
                    debug_lines.append(f"  ERROR downloading composite: {e}")
                    composite_tensor = _blank_image()
            else:
                composite_tensor = _blank_image()

            if background_url:
                try:
                    background_tensor = await download_url_to_image_tensor(background_url)
                except Exception as e:
                    logger.error(f"Failed to download background image: {e}")
                    debug_lines.append(f"  ERROR downloading background: {e}")
                    background_tensor = _blank_image()
            else:
                background_tensor = _blank_image()

            if mask_url:
                try:
                    mask_tensor = await download_url_to_image_tensor(mask_url)
                except Exception as e:
                    logger.error(f"Failed to download mask image: {e}")
                    debug_lines.append(f"  ERROR downloading mask: {e}")
                    mask_tensor = _blank_image()
            else:
                mask_tensor = _blank_image()

            debug_lines.append(f"  Composite shape: {list(composite_tensor.shape)}")
            debug_lines.append(f"  Background shape: {list(background_tensor.shape)}")
            debug_lines.append(f"  Mask shape: {list(mask_tensor.shape)}")
            debug_lines.append("[S3D Composite] Done")

            debug_log = "\n".join(debug_lines)
            logger.info(debug_log)

            # Display success notification on the node
            if unique_id:
                PromptServer.instance.send_progress_text(
                    f"Composite complete - {width}x{height}", unique_id
                )

            return (composite_tensor, background_tensor, mask_tensor, debug_log)

        except Exception as e:
            error_msg = f"S3D Composite error: {e}"
            logger.error(error_msg, exc_info=True)
            debug_lines.append(f"  ERROR: {error_msg}")
            debug_log = "\n".join(debug_lines)
            return (_blank_image(), _blank_image(), _blank_image(), debug_log)
