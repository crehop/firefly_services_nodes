"""
Video Reframe nodes for ComfyUI.

Provides Adobe Audio-Video Reframe API node for video overlay compositing.
"""

from .videoreframe_node import VideoReframeNode

NODE_CLASS_MAPPINGS = {
    "VideoReframeNode": VideoReframeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoReframeNode": "Video Reframe (Overlay)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
