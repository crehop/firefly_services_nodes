"""
Video Reframe Node for ComfyUI.

Applies overlays to videos using the Adobe Audio-Video Reframe API.
Accepts video and overlay as tensors, VIDEO objects, or presigned URLs.
"""

import io
import json
import logging
import torch
import tempfile
import os
from typing import Optional

from .videoreframe_api import (
    ReframeRequest,
    VideoSource,
    SourceURL,
    Composition,
    Overlay,
    OverlayScale,
    OverlayPosition,
    OutputSettings,
    OutputFormat,
    Rendition,
    RenditionResolution,
)
from .videoreframe_client import submit_reframe_job
from ..Photoshop.photoshop_storage import upload_image_to_s3, upload_file_to_s3
from ..apinode_utils import tensor_to_bytesio, download_url_to_video_output

logger = logging.getLogger(__name__)


async def _upload_video_to_s3(video) -> str:
    """Upload a VIDEO object (VideoFromFile) to S3 and return a presigned URL."""
    # VideoFromFile stores either a file path (str) or BytesIO via get_stream_source()
    source = video.get_stream_source() if hasattr(video, 'get_stream_source') else video

    if isinstance(source, str):
        # It's a file path on disk — upload directly
        return await upload_file_to_s3(source, content_type="video/mp4")

    # It's a BytesIO — write to temp file then upload
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)

        if isinstance(source, io.BytesIO):
            source.seek(0)
            data = source.read()
        elif isinstance(source, (bytes, bytearray)):
            data = source
        else:
            raise ValueError(f"Unsupported video type: {type(video)}")

        with open(tmp_path, 'wb') as f:
            f.write(data)

        return await upload_file_to_s3(tmp_path, content_type="video/mp4")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


class VideoReframeNode:
    """
    Reframe/overlay a video using Adobe Audio-Video API.

    Accepts video and overlay as direct inputs (IMAGE/VIDEO tensors)
    or as presigned URL strings. Outputs both the processed VIDEO
    and the output URL.
    """

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "output_url", "debug_log")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                # Dual input: tensor OR URL for video
                "video_input": ("VIDEO", {
                    "tooltip": "Input video (from another node)",
                }),
                "video_url": ("STRING", {
                    "forceInput": True,
                    "default": "",
                    "tooltip": "Presigned URL of the input video (alternative to video_input)",
                }),

                # Dual input: tensor+mask OR URL for overlay
                "overlay_image": ("IMAGE", {
                    "tooltip": "Overlay image tensor (from another node)",
                }),
                "overlay_mask": ("MASK", {
                    "tooltip": "Overlay alpha/transparency mask (connect MASK from LoadImage)",
                }),
                "overlay_url": ("STRING", {
                    "forceInput": True,
                    "default": "",
                    "tooltip": "Presigned URL of the overlay image (alternative to overlay_image)",
                }),

                # Timing
                "start_time": ("STRING", {
                    "default": "00:00:00:000",
                    "tooltip": "Overlay start time (HH:MM:SS:mmm).",
                }),
                "duration": ("STRING", {
                    "default": "00:00:05:000",
                    "tooltip": "Overlay duration (HH:MM:SS:mmm).",
                }),

                # Output resolution
                "output_width": ("INT", {
                    "default": 1080, "min": 100, "max": 3840,
                    "tooltip": "Output video width.",
                }),
                "output_height": ("INT", {
                    "default": 1920, "min": 100, "max": 3840,
                    "tooltip": "Output video height.",
                }),

                # Overlay scale
                "overlay_width": ("INT", {
                    "default": 1080, "min": 100, "max": 3840,
                    "tooltip": "Overlay scale width.",
                }),
                "overlay_height": ("INT", {
                    "default": 1920, "min": 100, "max": 3840,
                    "tooltip": "Overlay scale height.",
                }),

                # Position
                "anchor_point": (
                    ["center", "topLeft", "topRight", "bottomLeft", "bottomRight"],
                    {"default": "center", "tooltip": "Overlay anchor point."},
                ),
                "offset_x": ("INT", {
                    "default": 0, "min": -3840, "max": 3840,
                    "tooltip": "Horizontal offset from anchor point.",
                }),
                "offset_y": ("INT", {
                    "default": 0, "min": -3840, "max": 3840,
                    "tooltip": "Vertical offset from anchor point.",
                }),

                "repeat": (
                    ["loop", "once"],
                    {"default": "loop", "tooltip": "Overlay repeat mode."},
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        video_input=None,
        video_url: str = "",
        overlay_image: Optional[torch.Tensor] = None,
        overlay_mask: Optional[torch.Tensor] = None,
        overlay_url: str = "",
        start_time: str = "00:00:00:000",
        duration: str = "00:00:05:000",
        output_width: int = 1080,
        output_height: int = 1920,
        overlay_width: int = 1080,
        overlay_height: int = 1920,
        anchor_point: str = "center",
        offset_x: int = 0,
        offset_y: int = 0,
        repeat: str = "loop",
        unique_id: Optional[str] = None,
    ):
        """Submit a video reframe job with overlay."""
        console_log = "=" * 55 + "\n"
        console_log += "Video Reframe\n"
        console_log += "-" * 55 + "\n"

        # Resolve video source
        if video_input is not None:
            console_log += "Uploading video tensor to S3...\n"
            try:
                resolved_video_url = await _upload_video_to_s3(video_input)
            except Exception as e:
                console_log += f"[ERROR] Video upload failed: {e}\n"
                logger.error("Video upload to S3 failed: %s", e)
                raise
            console_log += f"Video uploaded: {resolved_video_url[:80]}...\n"
        elif video_url and video_url.strip():
            resolved_video_url = video_url
            console_log += f"Video URL: {resolved_video_url[:80]}...\n"
        else:
            raise ValueError("Must provide either 'video_input' or 'video_url'")

        # Resolve overlay source — must preserve alpha/transparency
        if overlay_image is not None:
            console_log += "Uploading overlay image to S3...\n"
            img_tensor = overlay_image[0] if len(overlay_image.shape) == 4 else overlay_image

            # Combine IMAGE (RGB) + MASK (alpha) into RGBA PNG
            from PIL import Image as PILImage
            import numpy as np

            rgb_np = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            if overlay_mask is not None:
                mask_tensor = overlay_mask[0] if len(overlay_mask.shape) == 3 else overlay_mask
                # ComfyUI mask: 1=transparent, 0=opaque — invert for alpha
                alpha_np = ((1.0 - mask_tensor.cpu().numpy()) * 255).clip(0, 255).astype(np.uint8)
                rgba_np = np.concatenate([rgb_np, alpha_np[:, :, None]], axis=2)
                pil_img = PILImage.fromarray(rgba_np, mode="RGBA")
            else:
                pil_img = PILImage.fromarray(rgb_np, mode="RGB")

            # Save to temp file and upload
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(tmp_fd)
            try:
                pil_img.save(tmp_path, format="PNG")
                try:
                    resolved_overlay_url = await upload_file_to_s3(tmp_path, content_type="image/png")
                except Exception as e:
                    console_log += f"[ERROR] Overlay upload failed: {e}\n"
                    logger.error("Overlay upload to S3 failed: %s", e)
                    raise
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            console_log += f"Overlay uploaded (RGBA): {resolved_overlay_url[:80]}...\n"
        elif overlay_url and overlay_url.strip():
            resolved_overlay_url = overlay_url
            console_log += f"Overlay URL: {resolved_overlay_url[:80]}...\n"
        else:
            raise ValueError("Must provide either 'overlay_image' or 'overlay_url'")

        console_log += f"Start: {start_time}, Duration: {duration}\n"
        console_log += f"Output: {output_width}x{output_height}\n"
        console_log += f"Overlay Scale: {overlay_width}x{overlay_height}\n"
        console_log += f"Position: {anchor_point} ({offset_x}, {offset_y})\n"
        console_log += f"Repeat: {repeat}\n"
        console_log += "-" * 55 + "\n"

        request = ReframeRequest(
            video=VideoSource(source=SourceURL(url=resolved_video_url)),
            composition=Composition(
                overlays=[
                    Overlay(
                        source=SourceURL(url=resolved_overlay_url),
                        startTime=start_time,
                        duration=duration,
                        scale=OverlayScale(width=overlay_width, height=overlay_height),
                        position=OverlayPosition(
                            anchorPoint=anchor_point,
                            offsetX=offset_x,
                            offsetY=offset_y,
                        ),
                        repeat=repeat,
                    )
                ]
            ),
            output=OutputSettings(
                format=OutputFormat(media="mp4"),
                renditions=[
                    Rendition(
                        resolution=RenditionResolution(
                            width=output_width,
                            height=output_height,
                        )
                    )
                ]
            ),
        )

        console_log += "Submitting reframe job...\n"

        try:
            result = await submit_reframe_job(request)
        except Exception as e:
            console_log += f"[ERROR] Reframe job submission failed: {e}\n"
            logger.error("Reframe job submission failed: %s", e)
            raise

        console_log += f"Job status: {result.status}\n"

        # Log raw response for debugging
        raw = result.model_dump()
        console_log += f"Raw result: {json.dumps(raw, indent=2)[:500]}\n"

        output_url = ""
        if result.outputs and len(result.outputs) > 0:
            first_output = result.outputs[0]
            # Try mediaDestination.url first (actual API response field)
            if first_output.mediaDestination and first_output.mediaDestination.url:
                output_url = first_output.mediaDestination.url
            # Fallback to destination.url
            elif first_output.destination and first_output.destination.url:
                output_url = first_output.destination.url

        if output_url:
            console_log += f"Output URL: {output_url[:120]}...\n"
        else:
            console_log += f"Warning: No output URL found in response\n"
            console_log += "=" * 55 + "\n"
            raise RuntimeError(f"Reframe succeeded but no output URL found. Raw: {json.dumps(raw, indent=2)[:500]}")

        # Download result video
        console_log += "Downloading output video...\n"
        try:
            video_output = await download_url_to_video_output(output_url)
        except Exception as e:
            console_log += f"[ERROR] Video download failed: {e}\n"
            logger.error("Video download failed: %s", e)
            raise
        console_log += "[OK] Video downloaded\n"
        console_log += "=" * 55 + "\n"

        return (video_output, output_url, console_log)
