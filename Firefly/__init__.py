"""
Firefly nodes for ComfyUI.

Provides 8 Adobe Firefly API nodes for image and video generation.
"""

from .firefly_nodes import (
    FireflyUploadImageNode,
    FireflyTextToImageNode,
    FireflyTextToVideoNode,
    FireflyListCustomModelsNode,
    FireflyGenerativeFillNode,
    FireflyGenerativeExpandNode,
    FireflyGenerateSimilarNode,
    FireflyGenerateObjectCompositeNode,
)

NODE_CLASS_MAPPINGS = {
    "FireflyUploadImageNode": FireflyUploadImageNode,
    "FireflyTextToImageNode": FireflyTextToImageNode,
    "FireflyTextToVideoNode": FireflyTextToVideoNode,
    "FireflyListCustomModelsNode": FireflyListCustomModelsNode,
    "FireflyGenerativeFillNode": FireflyGenerativeFillNode,
    "FireflyGenerativeExpandNode": FireflyGenerativeExpandNode,
    "FireflyGenerateSimilarNode": FireflyGenerateSimilarNode,
    "FireflyGenerateObjectCompositeNode": FireflyGenerateObjectCompositeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FireflyUploadImageNode": "Upload Image",
    "FireflyTextToImageNode": "Text to Image",
    "FireflyTextToVideoNode": "Text to Video",
    "FireflyListCustomModelsNode": "List Custom Models",
    "FireflyGenerativeFillNode": "Generative Fill",
    "FireflyGenerativeExpandNode": "Generative Expand",
    "FireflyGenerateSimilarNode": "Generate Similar",
    "FireflyGenerateObjectCompositeNode": "Object Composite",
}

# V5 node loaded separately — requires separate config with staging credentials
try:
    from .firefly_nodes import FireflyImage5Node
    NODE_CLASS_MAPPINGS["FireflyImage5Node"] = FireflyImage5Node
    NODE_DISPLAY_NAME_MAPPINGS["FireflyImage5Node"] = "Firefly Image 5"
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
