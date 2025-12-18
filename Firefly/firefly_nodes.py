"""
Adobe Firefly API Nodes - Refactored with ComfyUI-EasyNodes

This is a cleaner, more maintainable version of the Firefly API nodes
using the EasyNodes framework for simplified node creation.
"""

from __future__ import annotations
from typing import Optional, Any
import torch
import logging

from .firefly_api import (
    FireflyContentClass,
    FireflyTaskStatus,
    FireflyPromptBiasingLocale,
    FireflyImageFormat,
    FireflySize,
    FireflyPublicBinaryInput,
    FireflyInputImage,
    FireflyInputMask,
    FireflyStyles,
    FireflyStructure,
    FireflyStyleImageReferenceV3,
    FireflyStructureImageReferenceV3,
    FireflyUpsamplerType,
    GenerateImagesRequest,
    FillImageRequest,
    ExpandImageRequest,
    GenerateSimilarImagesRequest,
    GenerateObjectCompositeRequest,
    UploadImageRequest,
    AsyncAcceptResponse,
    AsyncTaskResponse,
    AsyncVideoTaskResponse,
    UploadImageResponse,
    FireflyVideoFormat,
    FireflyVideoSize,
    FireflyVideoSettings,
    FireflyVideoImagePlacement,
    FireflyVideoImageCondition,
    GenerateVideoRequest,
    FireflyOutputVideo,
    CustomModelsResponse,
)
from ..client import (
    ApiClient,
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)
from ..adobe_auth import get_adobe_auth_manager
from ..apinode_utils import (
    bytesio_to_image_tensor,
    download_url_to_bytesio,
    download_url_to_video_output,
    tensor_to_bytesio,
    resize_mask_to_image,
    validate_string,
)
from comfy.utils import ProgressBar
from io import BytesIO


# ============================================================================
# Helper Functions (Shared with original nodes)
# ============================================================================

async def create_adobe_client(model_version: str = "image3") -> ApiClient:
    """
    Create an ApiClient configured for Adobe Firefly API with OAuth authentication.

    Args:
        model_version: Firefly model version (e.g., "image3", "image4_standard")

    Returns:
        ApiClient instance with Adobe authentication
    """
    auth_manager = get_adobe_auth_manager()
    auth_headers = await auth_manager.get_auth_headers()

    # Add model version header
    auth_headers["x-model-version"] = model_version

    # Create client with dummy auth to bypass ComfyUI login check
    client = ApiClient(
        base_url="https://firefly-api.adobe.io",
        verify_ssl=True,
        comfy_api_key="adobe_oauth",
    )

    # Store auth headers for use in requests
    client._adobe_headers = auth_headers

    # Override get_headers to include Adobe auth
    original_get_headers = client.get_headers

    def get_headers_with_adobe():
        headers = original_get_headers()
        headers.update(client._adobe_headers)
        headers.pop("X-API-KEY", None)
        return headers

    client.get_headers = get_headers_with_adobe

    return client


async def upload_image_to_firefly(
    image: torch.Tensor,
    total_pixels: int = 4096 * 4096,
) -> str:
    """
    Upload an image to Firefly storage and return the upload ID.

    Args:
        image: Image tensor to upload
        total_pixels: Maximum total pixels for the image

    Returns:
        Upload ID from Firefly storage
    """
    client = await create_adobe_client()

    try:
        # Convert tensor to bytes
        image_bytes = tensor_to_bytesio(image, total_pixels=total_pixels)
        image_bytes.seek(0)  # Reset buffer position
        data = image_bytes.read()

        # Build headers
        headers = client.get_headers()
        headers["Content-Type"] = FireflyImageFormat.IMAGE_PNG.value

        # Make direct HTTP request with raw binary data
        url = client.base_url.rstrip("/") + "/v2/storage/image"
        session = await client._get_session()

        async with session.post(url, data=data, headers=headers, ssl=client.verify_ssl) as resp:
            resp.raise_for_status()
            response_json = await resp.json()

        # Extract upload ID from response {"images": [{"id": "..."}]}
        if "images" in response_json and len(response_json["images"]) > 0:
            upload_id = response_json["images"][0]["id"]
            return upload_id
        else:
            raise Exception(f"Unexpected response format: {response_json}")

    except Exception as e:
        raise Exception(f"Failed to upload image to Firefly: {str(e)}")
    finally:
        await client.close()


async def download_firefly_outputs(
    outputs: list,
    timeout: int = 1024,
    unique_id: Optional[str] = None,
) -> tuple[list[BytesIO], list[str]]:
    """
    Download output images from Firefly API response and capture presigned URLs.

    Args:
        outputs: List of output objects with image references
        timeout: Download timeout in seconds
        unique_id: Node unique ID for progress updates

    Returns:
        Tuple of (bytesio_list, url_list) containing downloaded images and their presigned URLs
    """
    all_bytesio = []
    all_urls = []

    for output in outputs:
        if hasattr(output, 'image') and output.image and output.image.url:
            url = output.image.url
            all_urls.append(url)
            all_bytesio.append(await download_url_to_bytesio(url, timeout=timeout))

    return all_bytesio, all_urls


async def download_firefly_video_outputs(
    outputs: list,
    timeout: int = 1024,
    unique_id: Optional[str] = None,
) -> tuple[list[BytesIO], list[str]]:
    """
    Download output videos from Firefly API response and capture presigned URLs.

    Args:
        outputs: List of output objects with video references
        timeout: Download timeout in seconds
        unique_id: Node unique ID for progress updates

    Returns:
        Tuple of (bytesio_list, url_list) containing downloaded videos and their presigned URLs
    """
    all_bytesio = []
    all_urls = []

    for output in outputs:
        if hasattr(output, 'video') and output.video and output.video.url:
            url = output.video.url
            all_urls.append(url)
            all_bytesio.append(await download_url_to_bytesio(url, timeout=timeout))

    return all_bytesio, all_urls


# ============================================================================
# Constant Definitions
# ============================================================================

FIREFLY_ASPECT_RATIOS = [
    "2048x2048 (1:1)",
    "2688x1536 (16:9)",
    "2304x1792 (4:3)",
    "1792x2304 (3:4)",
    "1344x768 (16:9)",
    "1152x896 (4:3)",
    "1024x1024 (1:1)",
    "896x1152 (3:4)",
    "2688x3456 (9:16)",
    "3456x2688 (16:9)",
]

# Mapping of human-readable names to Firefly API enum values
FIREFLY_STYLE_PRESET_MAP = {
    "None": "none",
    # Colors & Tones
    "Wireframe": "wireframe",
    "Vector look": "vector_look",
    "Black and white": "bw",
    "Cool tone": "cool_colors",
    "Golden": "golden",
    "Monochromatic": "monochromatic",
    "Muted color": "muted_color",
    "Pastel color": "pastel_color",
    "Toned image": "toned_image",
    "Vibrant colors": "vibrant_colors",
    "Warm tone": "warm_tone",
    # Photography Styles
    "Closeup": "closeup",
    "Knolling": "knolling",
    "Landscape photography": "landscape_photography",
    "Macrophotography": "macrophotography",
    "Photographed through window": "photographed_through_window",
    "Shallow depth of field": "shallow_depth_of_field",
    "Shot from above": "shot_from_above",
    "Shot from below": "shot_from_below",
    "Surface detail": "surface_detail",
    "Wide angle": "wide_angle",
    # Moods & Themes
    "Beautiful": "beautiful",
    "Bohemian": "bohemian",
    "Chaotic": "chaotic",
    "Dais": "dais",
    "Divine": "divine",
    "Eclectic": "eclectic",
    "Futuristic": "futuristic",
    "Kitschy": "kitschy",
    "Nostalgic": "nostalgic",
    "Simple": "simple",
    # Effects
    "Antique photo": "antique_photo",
    "Bioluminescent": "bioluminescent",
    "Bokeh effect": "bokeh",
    "Color explosion": "color_explosion",
    "Dark": "dark",
    "Faded image": "faded_image",
    "Fisheye": "fisheye",
    "Gomori photography": "gomori_photography",
    "Grainy film": "grainy_film",
    "Iridescent": "iridescent",
    "Isometric": "isometric",
    "Misty": "misty",
    "Neon": "neon",
    "Otherworldly depiction": "otherworldly_depiction",
    "Ultraviolet": "ultraviolet",
    "Underwater": "underwater",
    # Lighting
    "Backlighting": "backlighting",
    "Dramatic light": "dramatic_light",
    "Golden hour": "golden_hour",
    "Harsh light": "harsh_light",
    "Long-time exposure": "long",
    "Low lighting": "low_lighting",
    "Multiexposure": "multiexposure",
    "Studio light": "studio_light",
    "Surreal lighting": "surreal_lighting",
    # Materials & Textures
    "3d patterns": "3d_patterns",
    "Charcoal": "charcoal",
    "Claymation": "claymation",
    "Fabric": "fabric",
    "Fur": "fur",
    "Guilloche patterns": "guilloche_patterns",
    "Layered paper": "layered_paper",
    "Marble": "marble_sculpture",
    "Metal": "made_of_metal",
    "Origami": "origami",
    "Paper mache": "paper_mache",
    "Polka-dot pattern": "polka",
    "Strange patterns": "strange_patterns",
    "Wood carving": "wood_carving",
    "Yarn": "yarn",
    # Art Movements & Styles
    "Art deco": "art_deco",
    "Art nouveau": "art_nouveau",
    "Baroque": "baroque",
    "Bauhaus": "bauhaus",
    "Constructivism": "constructivism",
    "Cubism": "cubism",
    "Cyberpunk": "cyberpunk",
    "Fantasy": "fantasy",
    "Fauvism": "fauvism",
    "Film noir": "film_noir",
    "Glitch art": "glitch_art",
    "Impressionism": "impressionism",
    "Industrial": "industrialism",
    "Maximalism": "maximalism",
    "Minimalism": "minimalism",
    "Modern art": "modern_art",
    "Modernism": "modernism",
    "Neo-expressionism": "neo",
    "Pointillism": "pointillism",
    "Psychedelic": "psychedelic",
    "Science fiction": "science_fiction",
    "Steampunk": "steampunk",
    "Surrealism": "surrealism",
    "Synthetism": "synthetism",
    "Synthwave": "synthwave",
    "Vaporwave": "vaporwave",
    # Art Techniques
    "Acrylic paint": "acrylic_paint",
    "Bold lines": "bold_lines",
    "Chiaroscuro": "chiaroscuro",
    "Color shift art": "color_shift_art",
    "Daguerreotype": "daguerreotype",
    "Digital fractal": "digital_fractal",
    "Doodle drawing": "doodle_drawing",
    "Double exposure": "double_exposure_portrait",
    "Fresco": "fresco",
    "Geometric pen": "geometric_pen",
    "Halftone": "halftone",
    "Ink": "ink",
    "Light painting": "light_painting",
    "Line drawing": "line_drawing",
    "Linocut": "linocut",
    "Oil paint": "oil_paint",
    "Paint Spattering": "paint_spattering",
    "Painting": "painting",
    "Palette knife": "palette_knife",
    "Photo manipulation": "photo_manipulation",
    "Scribble texture": "scribble_texture",
    "Sketch": "sketch",
    "Splattering": "splattering",
    "Stippling": "stippling_drawing",
    "Watercolor": "watercolor",
    # Digital & Graphic Styles
    "3d": "3d",
    "Anime": "anime",
    "Cartoon": "cartoon",
    "Cinematic": "cinematic",
    "Comic book": "comic_book",
    "Concept art": "concept_art",
    "Cyber matrix": "cyber_matrix",
    "Digital art": "digital_art",
    "Flat design": "flat_design",
    "Geometric": "geometric",
    "Glassmorphism": "glassmorphism",
    "Glitch graphic": "glitch_graphic",
    "Graffiti": "graffiti",
    "Hyper realistic": "hyper_realistic",
    "Interior design": "interior_design",
    "Line gradient": "line_gradient",
    "Low poly": "low_poly",
    "Newspaper collage": "newspaper_collage",
    "Optical illusion": "optical_illusion",
    "Pattern pixel": "pattern_pixel",
    "Pixel art": "pixel_art",
    "Pop art": "pop_art",
    "Product photo": "product_photo",
    "Psychedelic background": "psychedelic_background",
    "Psychedelic wonderland": "psychedelic_wonderland",
    "Scandinavian": "scandinavian",
    "Splash images": "splash_images",
    "Stamp": "stamp",
    "Trompe l'oeil": "trompe_loeil",
}

# List of human-readable names for the dropdown
FIREFLY_STYLE_PRESETS = list(FIREFLY_STYLE_PRESET_MAP.keys())

# Video generation constants
# Allowed dimensions per API: (960,540), (540,960), (540,540), (1280,720), (720,1280), (720,720), (1920,1080), (1080,1920), (1080,1080)
FIREFLY_VIDEO_SIZES = [
    "540x540 (1:1)",
    "720x720 (1:1)",
    "1080x1080 (1:1)",
    "540x960 (9:16)",
    "720x1280 (9:16)",
    "1080x1920 (9:16)",
    "960x540 (16:9)",
    "1280x720 (16:9)",
    "1920x1080 (16:9)",
]

FIREFLY_CAMERA_MOTIONS = [
    "None",
    "camera pan left",
    "camera pan right",
    "camera zoom in",
    "camera zoom out",
    "camera tilt up",
    "camera tilt down",
    "camera locked down",
    "camera handheld",
]

FIREFLY_VIDEO_STYLES = [
    "None",
    "anime",
    "3d",
    "fantasy",
    "cinematic",
    "claymation",
    "line art",
    "stop motion",
    "2d",
    "vector art",
    "black and white",
]

FIREFLY_SHOT_ANGLES = [
    "None",
    "aerial shot",
    "eye_level shot",
    "high angle shot",
    "low angle shot",
    "top-down shot",
]

FIREFLY_SHOT_SIZES = [
    "None",
    "close-up shot",
    "extreme close-up",
    "medium shot",
    "long shot",
    "extreme long shot",
]


# ============================================================================
# Node Implementation
# ============================================================================

# Note: While we explored EasyNodes, the async nature of Firefly API and complex
# optional parameters are better served by traditional node implementation with
# improved structure. This version focuses on clean architecture and better
# organization rather than framework-specific features.

class FireflyUploadImageNode:
    """
    Upload an image to Firefly storage and return the upload ID.
    This is a helper node for other Firefly nodes that require image references.

    Features:
    - Cleaner code structure
    - Better error handling
    - Improved documentation
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("upload_id",)
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/firefly"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    async def api_call(self, image: torch.Tensor):
        """Upload first image in batch to Firefly storage."""
        upload_id = await upload_image_to_firefly(
            image=image[0] if len(image.shape) == 4 else image,
        )
        return (upload_id,)


class FireflyTextToImageNode:
    """
    Generate images from text prompts using Adobe Firefly.

    Features:
    - Cleaner parameter organization
    - Better default values
    - Improved documentation
    - Enhanced debug logging
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "image_url_2", "image_url_3", "image_url_4", "debug_log")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/firefly"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                # Text inputs (at top)
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for image generation (max 1024 characters).",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Negative prompt to exclude unwanted elements.",
                    },
                ),
                "prompt_suffix": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text to append to the main prompt.",
                    },
                ),

                # Seed (comma-separated for multiple values)
                "seed": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Seed(s) for reproducibility. Can be single (e.g. '123') or comma-separated (e.g. '1,2,3,4') for multiple variations.",
                    },
                ),

                # Primary controls
                "model_version": (
                    ["image3", "image3_custom", "image4_standard", "image4_ultra", "image4_custom"],
                    {
                        "default": "image4_standard",
                        "tooltip": "Firefly model version.",
                    },
                ),
                "model_version_string": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "String override for model_version (takes precedence if connected).",
                    },
                ),
                "size": (
                    FIREFLY_ASPECT_RATIOS,
                    {
                        "default": "2048x2048 (1:1)",
                        "tooltip": "Image size and aspect ratio.",
                    },
                ),
                "num_variations": (
                    "INT",
                    {
                        "min": 1,
                        "max": 4,
                        "default": 1,
                        "tooltip": "Number of image variations to generate.",
                    },
                ),
                "content_class": (
                    ["photo", "art"],
                    {
                        "default": "photo",
                        "tooltip": "Content class: 'photo' for photorealistic, 'art' for artistic style.",
                    },
                ),
                "style_preset": (
                    FIREFLY_STYLE_PRESETS,
                    {
                        "default": "None",
                        "tooltip": "Style preset to apply to generation.",
                    },
                ),
                "upsampler_type": (
                    ["default", "low_creativity"],
                    {
                        "default": "default",
                        "tooltip": "Upsampler type (only for image4_custom model).",
                    },
                ),
                "visual_intensity": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Visual intensity of the generation (2-10). Leave empty for default.",
                    },
                ),

                # Reference inputs
                "structure_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Structure reference image upload ID or presigned URL from another node.",
                    },
                ),
                "style_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Style reference image upload ID or presigned URL from another node.",
                    },
                ),

                # Image uploads
                "structure_image": (
                    "IMAGE",
                    {
                        "tooltip": "Structure reference image (auto-uploads to Firefly storage).",
                    },
                ),
                "style_image": (
                    "IMAGE",
                    {
                        "tooltip": "Style reference image (auto-uploads to Firefly storage).",
                    },
                ),

                # Strength controls
                "structure_strength": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Structure reference strength (0-100). Leave empty for default.",
                    },
                ),
                "style_strength": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Style reference strength (0-100). Leave empty for default.",
                    },
                ),

                # Advanced options
                "custom_model_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Custom model identifier (for *_custom model versions).",
                    },
                ),
                "prompt_biasing_locale": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Language/locale code (e.g., 'en-US', 'ja-JP').",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        prompt_suffix: str = "",
        custom_model_id: str = "",
        prompt_biasing_locale: str = "",
        model_version: str = "image4_standard",
        model_version_string: str = "",
        content_class: str = "photo",
        size: str = "2048x2048 (1:1)",
        num_variations: int = 1,
        seed: str = "",
        visual_intensity: str = "",
        style_preset: str = "None",
        style_image: Optional[torch.Tensor] = None,
        style_reference: str = "",
        style_strength: str = "",
        structure_image: Optional[torch.Tensor] = None,
        structure_reference: str = "",
        structure_strength: str = "",
        upsampler_type: str = "default",
        unique_id: Optional[str] = None,
    ):
        """Generate images using Adobe Firefly API."""

        # Apply string override for model_version
        if model_version_string:
            model_version = model_version_string

        # Convert human-readable style preset to API enum value
        style_preset_enum = FIREFLY_STYLE_PRESET_MAP.get(style_preset, "none")

        # Concatenate prompt with suffix if provided
        full_prompt = (prompt + " " + prompt_suffix).strip() if prompt_suffix else prompt

        # Validate prompt
        validate_string(full_prompt, strip_whitespace=False, max_length=1024)

        # Parse size
        size_parts = size.split(" ")[0].split("x")
        width = int(size_parts[0])
        height = int(size_parts[1])

        # Parse seeds
        seeds_list = None
        if seed and seed.strip():
            try:
                seeds_list = [int(s.strip()) for s in seed.split(",") if s.strip()]
            except ValueError:
                raise ValueError(f"Invalid seed format: '{seed}'. Use integers separated by commas (e.g., '1,2,3,4').")

        # Handle empty upsampler_type (default to "default")
        if not upsampler_type or upsampler_type == "":
            upsampler_type = "default"

        # Parse visual_intensity
        visual_intensity_int = None
        if visual_intensity and visual_intensity.strip():
            try:
                visual_intensity_int = int(visual_intensity.strip())
            except ValueError:
                raise ValueError(f"Invalid visual_intensity: '{visual_intensity}'. Must be an integer between 2-10.")

        # Validate style inputs (allow at most one)
        style_inputs_provided = sum([
            style_image is not None,
            bool(style_reference),
        ])
        if style_inputs_provided > 1:
            raise ValueError("Cannot provide multiple style inputs - choose only one: 'style_image' or 'style_reference'")

        # Validate structure inputs (allow at most one)
        structure_inputs_provided = sum([
            structure_image is not None,
            bool(structure_reference),
        ])
        if structure_inputs_provided > 1:
            raise ValueError("Cannot provide multiple structure inputs - choose only one: 'structure_image' or 'structure_reference'")

        # Create Adobe API client
        client = await create_adobe_client(model_version=model_version)

        try:
            # Build style configuration
            style_config = None

            # Check if any style parameters are provided
            has_style_preset = style_preset_enum and style_preset_enum != "none"
            has_style_image_or_ref = style_image is not None or style_reference
            has_style_strength = style_strength and style_strength.strip()

            if has_style_preset or has_style_image_or_ref or has_style_strength:
                # Build style_kwargs conditionally
                style_kwargs = {}

                # Add imageReference if provided
                if style_image is not None or style_reference:
                    if style_image is not None:
                        # Upload to Firefly storage and get upload ID
                        upload_id = await upload_image_to_firefly(
                            image=style_image[0] if len(style_image.shape) == 4 else style_image,
                        )
                        style_ref = FireflyStyleImageReferenceV3(
                            source=FireflyPublicBinaryInput(uploadId=upload_id)
                        )
                    else:
                        # Use provided upload ID or presigned URL from node connection
                        if style_reference.lower().startswith("http"):
                            style_ref = FireflyStyleImageReferenceV3(
                                source=FireflyPublicBinaryInput(url=style_reference)
                            )
                        else:
                            style_ref = FireflyStyleImageReferenceV3(
                                source=FireflyPublicBinaryInput(uploadId=style_reference)
                            )
                    style_kwargs["imageReference"] = style_ref

                # Add presets if provided
                if has_style_preset:
                    style_kwargs["presets"] = [style_preset_enum]

                # Add strength if provided
                if has_style_strength:
                    try:
                        style_strength_int = int(style_strength.strip())
                        style_kwargs["strength"] = style_strength_int
                    except ValueError:
                        raise ValueError(f"Invalid style_strength: '{style_strength}'. Must be an integer between 0-100.")

                # Create style config if we have any parameters
                if style_kwargs:
                    style_config = FireflyStyles(**style_kwargs)

            # Build structure configuration
            structure_config = None
            if structure_image is not None or structure_reference:
                # Determine structure image source
                if structure_image is not None:
                    # Upload to Firefly storage and get upload ID
                    upload_id = await upload_image_to_firefly(
                        image=structure_image[0] if len(structure_image.shape) == 4 else structure_image,
                    )
                    structure_ref = FireflyStructureImageReferenceV3(
                        source=FireflyPublicBinaryInput(uploadId=upload_id)
                    )
                else:
                    # Use provided upload ID or presigned URL from node connection
                    if structure_reference.lower().startswith("http"):
                        structure_ref = FireflyStructureImageReferenceV3(
                            source=FireflyPublicBinaryInput(url=structure_reference)
                        )
                    else:
                        structure_ref = FireflyStructureImageReferenceV3(
                            source=FireflyPublicBinaryInput(uploadId=structure_reference)
                        )

                # Parse structure_strength
                structure_strength_int = None
                if structure_strength and structure_strength.strip():
                    try:
                        structure_strength_int = int(structure_strength.strip())
                    except ValueError:
                        raise ValueError(f"Invalid structure_strength: '{structure_strength}'. Must be an integer between 0-100.")

                structure_config = FireflyStructure(
                    imageReference=structure_ref,
                    strength=structure_strength_int,
                )

            # Prepare request
            request = GenerateImagesRequest(
                prompt=full_prompt,
                contentClass=FireflyContentClass(content_class),
                customModelId=custom_model_id if (custom_model_id and model_version.endswith("_custom")) else None,
                size=FireflySize(width=width, height=height),
                numVariations=num_variations,
                seeds=seeds_list,
                negativePrompt=negative_prompt if negative_prompt else None,
                promptBiasingLocaleCode=prompt_biasing_locale if prompt_biasing_locale else None,
                style=style_config,
                structure=structure_config,
                visualIntensity=visual_intensity_int if (visual_intensity_int is not None and model_version != "image4_custom") else None,
                upsamplerType=FireflyUpsamplerType(upsampler_type) if model_version == "image4_custom" else None,
            )

            # Build debug log
            console_log = self._build_debug_log(
                model_version=model_version,
                prompt=full_prompt,
                prompt_suffix=prompt_suffix,
                content_class=content_class,
                width=width,
                height=height,
                num_variations=num_variations,
                seed=seed,
                visual_intensity=visual_intensity,
                negative_prompt=negative_prompt,
                custom_model_id=custom_model_id,
                prompt_biasing_locale=prompt_biasing_locale,
                style_config=style_config,
                style_image=style_image,
                style_reference=style_reference,
                style_strength=style_strength,
                style_preset=style_preset,
                structure_config=structure_config,
                structure_image=structure_image,
                structure_reference=structure_reference,
                structure_strength=structure_strength,
                upsampler_type=upsampler_type,
            )

            # Submit async job
            submit_endpoint = ApiEndpoint(
                path="/v3/images/generate-async",
                method=HttpMethod.POST,
                request_model=GenerateImagesRequest,
                response_model=AsyncAcceptResponse,
            )

            submit_op = SynchronousOperation(
                endpoint=submit_endpoint,
                request=request,
                api_base="https://firefly-api.adobe.io",
            )

            submit_response: AsyncAcceptResponse = await submit_op.execute(client=client)

            console_log += f"\nResponse: 202 Accepted\n"
            console_log += f"  jobId: {submit_response.jobId}\n"

            # Poll for completion
            console_log += f"\n{'='*55}\n"
            console_log += f"GET /v3/status/{submit_response.jobId.split(':')[-1][:8]}...\n"
            console_log += f"{'-'*55}\n"
            console_log += f"Polling for job completion...\n"

            poll_endpoint = ApiEndpoint(
                path=f"/v3/status/{submit_response.jobId}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=AsyncTaskResponse,
            )

            poll_op = PollingOperation(
                poll_endpoint=poll_endpoint,
                request=EmptyRequest(),
                completed_statuses=["succeeded"],
                failed_statuses=["failed", "canceled"],
                status_extractor=lambda x: x.status,
                api_base="https://firefly-api.adobe.io",
                poll_interval=2.0,
                max_poll_attempts=120,
                node_id=unique_id,
            )

            result: AsyncTaskResponse = await poll_op.execute(client=client)

            console_log += f"\nResponse: 200 OK\n"
            console_log += f"  status: {result.status}\n"
            console_log += f"  jobId: {result.jobId}\n"
            console_log += f"  outputs: {len(result.outputs) if result.outputs else 0} image(s)\n"

            # Validate outputs
            if not result.outputs:
                console_log += f"\n{'='*55}\n"
                console_log += f"ERROR: No outputs in response\n"
                console_log += f"  status: {result.status}\n"
                raise Exception(f"No outputs returned from Firefly API. Status: {result.status}")

            # Download outputs
            console_log += f"\n{'='*55}\n"
            console_log += f"Downloading {len(result.outputs)} output(s)...\n"

            output_bytesio, presigned_urls = await download_firefly_outputs(
                result.outputs,
                unique_id=unique_id,
            )

            console_log += f"[OK] Downloaded {len(output_bytesio)} image(s)\n"
            console_log += f"{'='*55}\n"

            # Add presigned URLs to console log
            console_log += f"\nPresigned URLs (valid for 1 hour):\n"
            for i, url in enumerate(presigned_urls, 1):
                console_log += f"  [{i}] {url}\n"
            console_log += f"{'='*55}\n"

            # Convert to tensors
            images = []
            for bytesio in output_bytesio:
                image = bytesio_to_image_tensor(bytesio)
                if len(image.shape) < 4:
                    image = image.unsqueeze(0)
                images.append(image)

            output_image = torch.cat(images, dim=0)

            # Return all 4 URLs (pad with empty strings if fewer generated)
            image_url = presigned_urls[0] if len(presigned_urls) > 0 else ""
            image_url_2 = presigned_urls[1] if len(presigned_urls) > 1 else ""
            image_url_3 = presigned_urls[2] if len(presigned_urls) > 2 else ""
            image_url_4 = presigned_urls[3] if len(presigned_urls) > 3 else ""

            return (output_image, image_url, image_url_2, image_url_3, image_url_4, console_log)

        finally:
            await client.close()

    def _build_debug_log(
        self,
        model_version: str,
        prompt: str,
        prompt_suffix: str,
        content_class: str,
        width: int,
        height: int,
        num_variations: int,
        seed: str,
        visual_intensity: str,
        negative_prompt: str,
        custom_model_id: str,
        prompt_biasing_locale: str,
        style_config: Any,
        style_image: Optional[torch.Tensor],
        style_reference: str,
        style_strength: str,
        style_preset: str,
        structure_config: Any,
        structure_image: Optional[torch.Tensor],
        structure_reference: str,
        structure_strength: str,
        upsampler_type: str,
    ) -> str:
        """Build formatted debug log for console output."""
        log = "=" * 55 + "\n"
        log += "POST /v3/images/generate-async\n"
        log += "-" * 55 + "\n"
        log += f"Headers:\n"
        log += f"  x-model-version: {model_version}\n"
        log += f"\nRequest Body:\n"
        log += f"  prompt: {prompt[:50]}...\n" if len(prompt) > 50 else f"  prompt: {prompt}\n"
        if prompt_suffix:
            log += f"  (suffix: '{prompt_suffix[:30]}...')\n" if len(prompt_suffix) > 30 else f"  (suffix: '{prompt_suffix}')\n"
        log += f"  contentClass: {content_class}\n"
        log += f"  size: {width}x{height}\n"
        log += f"  numVariations: {num_variations}\n"
        if seed and seed.strip():
            log += f"  seeds: [{seed}]\n"
        if visual_intensity and visual_intensity.strip() and model_version != "image4_custom":
            log += f"  visualIntensity: {visual_intensity}\n"
        if custom_model_id:
            log += f"  customModelId: {custom_model_id}\n"
        if prompt_biasing_locale:
            log += f"  promptBiasingLocaleCode: {prompt_biasing_locale}\n"
        if negative_prompt:
            log += f"  negativePrompt: {negative_prompt[:30]}...\n" if len(negative_prompt) > 30 else f"  negativePrompt: {negative_prompt}\n"

        if style_config:
            log += f"  style:\n"
            if style_image is not None:
                log += f"    uploadId: [UPLOADED IMAGE]\n"
            else:
                log += f"    uploadId: {style_reference}\n"
            if style_strength and style_strength.strip():
                log += f"    strength: {style_strength}\n"
            if style_preset and style_preset != "none":
                log += f"    preset: {style_preset}\n"

        if structure_config:
            log += f"  structure:\n"
            if structure_image is not None:
                log += f"    uploadId: [UPLOADED IMAGE]\n"
            else:
                log += f"    uploadId: {structure_reference}\n"
            if structure_strength and structure_strength.strip():
                log += f"    strength: {structure_strength}\n"

        if model_version == "image4_custom":
            log += f"  upsamplerType: {upsampler_type}\n"

        return log


class FireflyGenerativeFillNode:
    """
    Fill/inpaint masked areas of an image using Adobe Firefly.

    Features:
    - Cleaner code structure
    - Better batch processing
    - Enhanced error handling
    - Dual input support for images and masks
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url_1", "image_url_2", "image_url_3", "image_url_4", "debug_log")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/firefly"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Image to fill (auto-uploads to Firefly storage).",
                    },
                ),
                "image_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Image upload ID or presigned URL from another node.",
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": "Mask for fill area (auto-uploads to Firefly storage).",
                    },
                ),
                "mask_image": (
                    "IMAGE",
                    {
                        "tooltip": "Mask image for fill area (auto-uploads to Firefly storage).",
                    },
                ),
                "mask_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Mask upload ID or presigned URL from another node.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional prompt to guide the fill.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Negative prompt to exclude unwanted elements.",
                    },
                ),
                "size": (
                    FIREFLY_ASPECT_RATIOS,
                    {
                        "default": "2048x2048 (1:1)",
                        "tooltip": "Output image size and aspect ratio.",
                    },
                ),
                "prompt_biasing_locale": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Locale for region-specific generation (e.g., 'en-US', 'de-DE', 'ja-JP'). Leave empty for AUTO.",
                    },
                ),
            },
            "required": {
                "num_variations": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Number of variations to generate.",
                    },
                ),
                "seed": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Seed(s) for reproducibility. Leave empty for random, or provide single seed (e.g. '12345') or multiple comma-separated seeds (e.g. '1,2,3,4').",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        num_variations: int,
        seed: str = "",
        image: Optional[torch.Tensor] = None,
        image_reference: str = "",
        mask: Optional[torch.Tensor] = None,
        mask_image: Optional[torch.Tensor] = None,
        mask_reference: str = "",
        prompt: str = "",
        negative_prompt: str = "",
        size: str = "2048x2048 (1:1)",
        prompt_biasing_locale: str = "",
        unique_id: Optional[str] = None,
    ):
        """Fill masked areas using Firefly API."""
        # Validate inputs
        if image is None and not image_reference:
            raise ValueError("Must provide either 'image' or 'image_reference'")
        if image is not None and image_reference:
            raise ValueError("Cannot provide both 'image' and 'image_reference' - choose only one")

        # Validate mask inputs (allow exactly one of three options)
        mask_inputs_provided = sum([
            mask is not None,
            mask_image is not None,
            bool(mask_reference),
        ])
        if mask_inputs_provided == 0:
            raise ValueError("Must provide one of: 'mask', 'mask_image', or 'mask_reference'")
        if mask_inputs_provided > 1:
            raise ValueError("Cannot provide multiple mask inputs - choose only one: 'mask', 'mask_image', or 'mask_reference'")

        # Parse seeds
        seeds_list = None
        if seed and seed.strip():
            try:
                seeds_list = [int(s.strip()) for s in seed.split(",") if s.strip()]
            except ValueError:
                raise ValueError(f"Invalid seed format: '{seed}'. Use integers separated by commas (e.g., '1,2,3,4').")

        # Parse size to extract width and height
        size_parts = size.split("x")
        width = int(size_parts[0])
        height = int(size_parts[1].split()[0])  # Extract height before the aspect ratio part

        client = await create_adobe_client()

        try:
            # Prepare mask if provided as tensor
            if mask is not None:
                if image is not None:
                    # Resize mask to match image dimensions
                    mask = resize_mask_to_image(mask, image, allow_gradient=False, add_channel_dim=True)
                else:
                    # Convert mask to RGB format (expand [B,H,W] to [B,H,W,3])
                    if len(mask.shape) == 3:
                        mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
                    elif len(mask.shape) == 2:
                        mask = mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)

            images = []
            all_urls = []
            total = image.shape[0] if image is not None else 1
            pbar = ProgressBar(total)

            for i in range(total):
                # Determine image source
                if image is not None:
                    # Upload to Firefly storage and get upload ID
                    upload_id = await upload_image_to_firefly(image=image[i])
                    image_source = FireflyPublicBinaryInput(uploadId=upload_id)
                else:
                    # Use provided upload ID or presigned URL
                    if image_reference.lower().startswith("http"):
                        image_source = FireflyPublicBinaryInput(url=image_reference)
                    else:
                        image_source = FireflyPublicBinaryInput(uploadId=image_reference)

                # Determine mask source
                if mask is not None:
                    # Upload MASK tensor (now in RGB format) to Firefly storage and get upload ID
                    mask_upload_id = await upload_image_to_firefly(image=mask[i:i+1])
                    mask_source = FireflyPublicBinaryInput(uploadId=mask_upload_id)
                elif mask_image is not None:
                    # Upload IMAGE tensor to Firefly storage and get upload ID
                    mask_upload_id = await upload_image_to_firefly(image=mask_image[i])
                    mask_source = FireflyPublicBinaryInput(uploadId=mask_upload_id)
                else:
                    # Use provided upload ID or presigned URL
                    if mask_reference.lower().startswith("http"):
                        mask_source = FireflyPublicBinaryInput(url=mask_reference)
                    else:
                        mask_source = FireflyPublicBinaryInput(uploadId=mask_reference)

                # Prepare request
                request = FillImageRequest(
                    image=FireflyInputImage(
                        source=image_source
                    ),
                    mask=FireflyInputMask(
                        source=mask_source
                    ),
                    prompt=prompt if prompt else None,
                    negativePrompt=negative_prompt if negative_prompt else None,
                    size=FireflySize(width=width, height=height),
                    numVariations=num_variations,
                    seeds=seeds_list,
                    promptBiasingLocaleCode=prompt_biasing_locale if prompt_biasing_locale else None,
                )

                # Submit and poll
                submit_endpoint = ApiEndpoint(
                    path="/v3/images/fill-async",
                    method=HttpMethod.POST,
                    request_model=FillImageRequest,
                    response_model=AsyncAcceptResponse,
                )

                submit_op = SynchronousOperation(
                    endpoint=submit_endpoint,
                    request=request,
                    api_base="https://firefly-api.adobe.io",
                )

                submit_response = await submit_op.execute(client=client)

                # Poll for completion
                poll_endpoint = ApiEndpoint(
                    path=f"/v3/status/{submit_response.jobId}",
                    method=HttpMethod.GET,
                    request_model=EmptyRequest,
                    response_model=AsyncTaskResponse,
                )

                poll_op = PollingOperation(
                    poll_endpoint=poll_endpoint,
                    request=EmptyRequest(),
                    completed_statuses=["succeeded"],
                    failed_statuses=["failed", "canceled"],
                    status_extractor=lambda x: x.status,
                    api_base="https://firefly-api.adobe.io",
                    poll_interval=5.0,
                    max_poll_attempts=120,
                    node_id=unique_id,
                )

                result = await poll_op.execute(client=client)

                # Download outputs
                if not result.outputs:
                    raise Exception("No outputs returned from Firefly API")

                output_bytesio, presigned_urls = await download_firefly_outputs(
                    result.outputs,
                    unique_id=unique_id,
                )

                all_urls.extend(presigned_urls)

                # Convert to tensors
                batch_images = []
                for bytesio in output_bytesio:
                    img = bytesio_to_image_tensor(bytesio)
                    if len(img.shape) < 4:
                        img = img.unsqueeze(0)
                    batch_images.append(img)

                images.append(torch.cat(batch_images, dim=0))
                pbar.update(1)

            images_tensor = torch.cat(images, dim=0)

            # Build comprehensive debug log
            console_log = self._build_debug_log(
                num_variations=num_variations,
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                size=size,
                prompt_biasing_locale=prompt_biasing_locale,
                image=image,
                image_reference=image_reference,
                mask=mask,
                mask_image=mask_image,
                mask_reference=mask_reference,
                total_processed=total,
                total_urls=len(all_urls),
            )

            # Add presigned URLs to console log
            console_log += f"\nPresigned URLs (valid for 1 hour):\n"
            for idx, url in enumerate(all_urls, 1):
                console_log += f"  [{idx}] {url}\n"
            console_log += f"{'='*55}\n"

            # Split URLs into individual outputs (up to 4)
            image_url_1 = all_urls[0] if len(all_urls) > 0 else ""
            image_url_2 = all_urls[1] if len(all_urls) > 1 else ""
            image_url_3 = all_urls[2] if len(all_urls) > 2 else ""
            image_url_4 = all_urls[3] if len(all_urls) > 3 else ""

            return (images_tensor, image_url_1, image_url_2, image_url_3, image_url_4, console_log)

        finally:
            await client.close()

    def _build_debug_log(
        self,
        num_variations: int,
        seed: str,
        prompt: str,
        negative_prompt: str,
        size: str,
        prompt_biasing_locale: str,
        image: Optional[torch.Tensor],
        image_reference: str,
        mask: Optional[torch.Tensor],
        mask_image: Optional[torch.Tensor],
        mask_reference: str,
        total_processed: int,
        total_urls: int,
    ) -> str:
        """Build formatted debug log for console output."""
        log = "=" * 55 + "\n"
        log += "POST /v3/images/fill-async\n"
        log += "-" * 55 + "\n"
        log += f"Headers:\n"
        log += f"  x-model-version: image3\n"
        log += f"\nRequest Body:\n"

        # Image source
        if image is not None:
            log += f"  image: [UPLOADED IMAGE]\n"
        else:
            log += f"  image: {image_reference}\n"

        # Mask source
        if mask is not None:
            log += f"  mask: [UPLOADED MASK]\n"
        elif mask_image is not None:
            log += f"  mask: [UPLOADED MASK IMAGE]\n"
        else:
            log += f"  mask: {mask_reference}\n"

        # Parse and display size
        size_parts = size.split("x")
        width = size_parts[0]
        height = size_parts[1].split()[0]
        log += f"  size: {width}x{height}\n"

        log += f"  numVariations: {num_variations}\n"

        if seed and seed.strip():
            log += f"  seeds: [{seed}]\n"

        if prompt and prompt.strip():
            log += f"  prompt: {prompt[:50]}...\n" if len(prompt) > 50 else f"  prompt: {prompt}\n"

        if negative_prompt and negative_prompt.strip():
            log += f"  negativePrompt: {negative_prompt[:30]}...\n" if len(negative_prompt) > 30 else f"  negativePrompt: {negative_prompt}\n"

        if prompt_biasing_locale and prompt_biasing_locale.strip():
            log += f"  promptBiasingLocaleCode: {prompt_biasing_locale}\n"

        log += f"\nProcessed: {total_processed} image(s)\n"
        log += f"Generated: {total_urls} output(s)\n"
        log += "=" * 55 + "\n"

        return log


class FireflyTextToVideoNode:
    """
    Generate videos from text prompts using Adobe Firefly.

    Features:
    - Text-to-video generation (5 seconds)
    - Camera controls (pan, zoom, tilt)
    - Style controls (anime, 3d, cinematic, etc.)
    - Shot controls (angle, size)
    - Keyframe image support
    - Async job polling
    """

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "debug_log")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/firefly"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                # Text input
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for video generation. The longer the prompt, the better.",
                    },
                ),

                # Model version
                "model_version": (
                    ["video1_standard"],
                    {
                        "default": "video1_standard",
                        "tooltip": "Firefly video model version.",
                    },
                ),
                "model_version_string": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "String override for model_version (takes precedence if connected).",
                    },
                ),

                # Video settings
                "size": (
                    FIREFLY_VIDEO_SIZES,
                    {
                        "default": "540x540 (1:1)",
                        "tooltip": "Video dimensions and aspect ratio.",
                    },
                ),
                "bit_rate_factor": (
                    "INT",
                    {
                        "min": 0,
                        "max": 63,
                        "default": 18,
                        "tooltip": "Constant rate factor for encoding. 0=lossless (highest quality, largest size), 63=worst quality (smallest size). Suggested: 17-23.",
                    },
                ),
                "seed": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Seed for reproducibility. Leave empty for random. Currently only 1 seed is supported.",
                    },
                ),

                # Camera and shot controls
                "camera_motion": (
                    FIREFLY_CAMERA_MOTIONS,
                    {
                        "default": "None",
                        "tooltip": "Camera motion control.",
                    },
                ),
                "video_style": (
                    FIREFLY_VIDEO_STYLES,
                    {
                        "default": "None",
                        "tooltip": "Style of the generated video.",
                    },
                ),
                "shot_angle": (
                    FIREFLY_SHOT_ANGLES,
                    {
                        "default": "None",
                        "tooltip": "Shot angle control.",
                    },
                ),
                "shot_size": (
                    FIREFLY_SHOT_SIZES,
                    {
                        "default": "None",
                        "tooltip": "Shot size control.",
                    },
                ),

                # Keyframe image (first/last frame)
                "keyframe_image": (
                    "IMAGE",
                    {
                        "tooltip": "Keyframe image for video generation (auto-uploads to Firefly storage). Used as first or last frame.",
                    },
                ),
                "keyframe_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Keyframe image upload ID or presigned URL from another node.",
                    },
                ),
                "keyframe_position": (
                    "FLOAT",
                    {
                        "min": 0.0,
                        "max": 1.0,
                        "default": 0.0,
                        "step": 0.1,
                        "tooltip": "Position of keyframe on timeline. 0=first frame, 1=last frame.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        prompt: str = "",
        model_version: str = "video1_standard",
        model_version_string: str = "",
        size: str = "540x540 (1:1)",
        bit_rate_factor: int = 18,
        seed: str = "",
        camera_motion: str = "None",
        video_style: str = "None",
        shot_angle: str = "None",
        shot_size: str = "None",
        keyframe_image: Optional[torch.Tensor] = None,
        keyframe_reference: str = "",
        keyframe_position: float = 0.0,
        unique_id: Optional[str] = None,
    ):
        """Generate video using Adobe Firefly API."""

        # Apply string override for model_version
        if model_version_string:
            model_version = model_version_string

        # Validate that at least one of prompt or keyframe is provided
        has_prompt = prompt and prompt.strip()
        has_keyframe = keyframe_image is not None or keyframe_reference

        if not has_prompt and not has_keyframe:
            raise ValueError("Must provide at least one of 'prompt' or keyframe image ('keyframe_image' or 'keyframe_reference')")

        # Parse size
        size_parts = size.split(" ")[0].split("x")
        width = int(size_parts[0])
        height = int(size_parts[1])

        # Parse seed
        seeds_list = None
        if seed and seed.strip():
            try:
                seeds_list = [int(seed.strip())]
            except ValueError:
                raise ValueError(f"Invalid seed format: '{seed}'. Must be an integer.")

        # Build video settings
        video_settings = None
        has_camera_motion = camera_motion and camera_motion != "None"
        has_video_style = video_style and video_style != "None"
        has_shot_angle = shot_angle and shot_angle != "None"
        has_shot_size = shot_size and shot_size != "None"

        if has_camera_motion or has_video_style or has_shot_angle or has_shot_size:
            video_settings_kwargs = {}
            if has_camera_motion:
                video_settings_kwargs["cameraMotion"] = camera_motion
            if has_video_style:
                video_settings_kwargs["promptStyle"] = video_style
            if has_shot_angle:
                video_settings_kwargs["shotAngle"] = shot_angle
            if has_shot_size:
                video_settings_kwargs["shotSize"] = shot_size
            video_settings = FireflyVideoSettings(**video_settings_kwargs)

        # Build keyframe conditions
        conditions = None
        if has_keyframe:
            # Determine keyframe source
            if keyframe_image is not None:
                # Upload to Firefly storage
                upload_id = await upload_image_to_firefly(
                    image=keyframe_image[0] if len(keyframe_image.shape) == 4 else keyframe_image,
                )
                keyframe_source = FireflyPublicBinaryInput(uploadId=upload_id)
            else:
                # Use provided upload ID or presigned URL
                if keyframe_reference.lower().startswith("http"):
                    keyframe_source = FireflyPublicBinaryInput(url=keyframe_reference)
                else:
                    keyframe_source = FireflyPublicBinaryInput(uploadId=keyframe_reference)

            # Create condition
            condition = FireflyVideoImageCondition(
                placement=FireflyVideoImagePlacement(position=keyframe_position),
                source=keyframe_source,
            )
            conditions = [condition]

        # Create Adobe API client
        client = await create_adobe_client(model_version=model_version)

        try:
            # Build request
            request = GenerateVideoRequest(
                prompt=prompt if has_prompt else None,
                bitRateFactor=bit_rate_factor,
                conditions=conditions,
                seeds=seeds_list,
                sizes=[FireflyVideoSize(width=width, height=height)],
                videoSettings=video_settings,
            )

            # Build debug log
            console_log = self._build_debug_log(
                model_version=model_version,
                prompt=prompt,
                width=width,
                height=height,
                bit_rate_factor=bit_rate_factor,
                seed=seed,
                camera_motion=camera_motion,
                video_style=video_style,
                shot_angle=shot_angle,
                shot_size=shot_size,
                keyframe_image=keyframe_image,
                keyframe_reference=keyframe_reference,
                keyframe_position=keyframe_position,
            )

            # Submit async job
            submit_endpoint = ApiEndpoint(
                path="/v3/videos/generate",
                method=HttpMethod.POST,
                request_model=GenerateVideoRequest,
                response_model=AsyncAcceptResponse,
            )

            submit_op = SynchronousOperation(
                endpoint=submit_endpoint,
                request=request,
                api_base="https://firefly-api.adobe.io",
            )

            submit_response: AsyncAcceptResponse = await submit_op.execute(client=client)

            console_log += f"\nResponse: 202 Accepted\n"
            console_log += f"  jobId: {submit_response.jobId}\n"

            # Poll for completion
            console_log += f"\n{'='*55}\n"
            console_log += f"GET /v3/status/{submit_response.jobId.split(':')[-1][:8]}...\n"
            console_log += f"{'-'*55}\n"
            console_log += f"Polling for job completion...\n"

            poll_endpoint = ApiEndpoint(
                path=f"/v3/status/{submit_response.jobId}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=AsyncVideoTaskResponse,
            )

            poll_op = PollingOperation(
                poll_endpoint=poll_endpoint,
                request=EmptyRequest(),
                completed_statuses=["succeeded"],
                failed_statuses=["failed", "canceled"],
                status_extractor=lambda x: x.status,
                api_base="https://firefly-api.adobe.io",
                poll_interval=5.0,
                max_poll_attempts=60,
                node_id=unique_id,
            )

            result: AsyncVideoTaskResponse = await poll_op.execute(client=client)

            console_log += f"\nResponse: 200 OK\n"
            console_log += f"  status: {result.status}\n"
            console_log += f"  jobId: {result.jobId}\n"
            console_log += f"  outputs: {len(result.outputs) if result.outputs else 0} video(s)\n"

            # Validate outputs
            if not result.outputs:
                console_log += f"\n{'='*55}\n"
                console_log += f"ERROR: No outputs in response\n"
                console_log += f"  status: {result.status}\n"
                raise Exception(f"No outputs returned from Firefly API. Status: {result.status}")

            # Download video
            console_log += f"\n{'='*55}\n"
            console_log += f"Downloading video output...\n"

            output_bytesio, presigned_urls = await download_firefly_video_outputs(
                result.outputs,
                unique_id=unique_id,
            )

            console_log += f"[OK] Downloaded video\n"
            console_log += f"{'='*55}\n"

            # Add presigned URL to console log
            video_url = presigned_urls[0] if len(presigned_urls) > 0 else ""
            console_log += f"\nPresigned URL (valid for 1 hour):\n"
            console_log += f"  {video_url}\n"
            console_log += f"{'='*55}\n"

            # Convert video BytesIO to VIDEO output
            video_output = await download_url_to_video_output(video_url)

            return (video_output, video_url, console_log)

        finally:
            await client.close()

    def _build_debug_log(
        self,
        model_version: str,
        prompt: str,
        width: int,
        height: int,
        bit_rate_factor: int,
        seed: str,
        camera_motion: str,
        video_style: str,
        shot_angle: str,
        shot_size: str,
        keyframe_image: Optional[torch.Tensor],
        keyframe_reference: str,
        keyframe_position: float,
    ) -> str:
        """Build formatted debug log for console output."""
        log = "=" * 55 + "\n"
        log += "POST /v3/videos/generate\n"
        log += "-" * 55 + "\n"
        log += f"Headers:\n"
        log += f"  x-model-version: {model_version}\n"
        log += f"\nRequest Body:\n"

        if prompt and prompt.strip():
            log += f"  prompt: {prompt[:50]}...\n" if len(prompt) > 50 else f"  prompt: {prompt}\n"

        log += f"  size: {width}x{height}\n"
        log += f"  bitRateFactor: {bit_rate_factor}\n"

        if seed and seed.strip():
            log += f"  seeds: [{seed}]\n"

        # Video settings
        has_video_settings = (
            (camera_motion and camera_motion != "None") or
            (video_style and video_style != "None") or
            (shot_angle and shot_angle != "None") or
            (shot_size and shot_size != "None")
        )
        if has_video_settings:
            log += f"  videoSettings:\n"
            if camera_motion and camera_motion != "None":
                log += f"    cameraMotion: {camera_motion}\n"
            if video_style and video_style != "None":
                log += f"    promptStyle: {video_style}\n"
            if shot_angle and shot_angle != "None":
                log += f"    shotAngle: {shot_angle}\n"
            if shot_size and shot_size != "None":
                log += f"    shotSize: {shot_size}\n"

        # Keyframe info
        if keyframe_image is not None or keyframe_reference:
            log += f"  conditions:\n"
            log += f"    - placement.position: {keyframe_position}\n"
            if keyframe_image is not None:
                log += f"      source.uploadId: [UPLOADED IMAGE]\n"
            else:
                log += f"      source: {keyframe_reference}\n"

        log += "=" * 55 + "\n"

        return log


class FireflyListCustomModelsNode:
    """
    List custom models from Adobe Firefly.

    Retrieves a paginated list of custom models available to the user.
    Useful for discovering available custom model IDs to use in generation.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("models_json",)
    FUNCTION = "list_models"
    OUTPUT_NODE = True
    CATEGORY = "api node/firefly"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "sort_by": (
                    ["None", "modifiedDate", "createdDate", "assetName", "-modifiedDate", "-createdDate", "-assetName"],
                    {
                        "default": "None",
                        "tooltip": "Sort option. Use - prefix for reverse sort (e.g., -modifiedDate). None uses API default.",
                    }
                ),
                "start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "First result to include in paginated response",
                }),
                "limit": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "tooltip": "Number of models to return (1-50)",
                }),
                "published_state": (
                    ["all", "ready", "published", "unpublished", "queued", "training", "failed", "cancelled"],
                    {
                        "default": "published",
                        "tooltip": "Filter models by published state",
                    }
                ),
            }
        }

    async def list_models(
        self,
        sort_by: str = "None",
        start: int = 0,
        limit: int = 10,
        published_state: str = "published",
    ):
        """List custom models from Firefly API."""

        # Create Adobe client
        client = await create_adobe_client()

        try:
            # Build query parameters - only include sortBy if not None
            params = {
                "start": str(start),
                "limit": str(limit),
                "publishedState": published_state,
            }

            # Only add sortBy if it's not "None"
            if sort_by and sort_by != "None":
                params["sortBy"] = sort_by

            # Create endpoint for listing custom models
            list_endpoint = ApiEndpoint(
                path="/v3/custom-models",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=CustomModelsResponse,
                query_params=params,
            )

            # Create operation
            list_op = SynchronousOperation(
                endpoint=list_endpoint,
                request=EmptyRequest(),
            )

            # Execute request
            response = await list_op.execute(client=client)

            # Format output as JSON string for display
            import json
            output_data = {
                "total_count": response.total_count or 0,
                "models": []
            }

            for model in response.custom_models:
                model_info = {
                    "assetId": model.assetId,
                    "assetName": model.assetName,
                    "displayName": model.displayName,
                    "trainingMode": model.trainingMode,
                    "conceptId": model.conceptId,
                    "publishedState": model.publishedState,
                    "createdDate": model.createdDate,
                    "modifiedDate": model.modifiedDate,
                    "samplePrompt": model.samplePrompt,
                    "baseModel": {
                        "name": model.baseModel.name if model.baseModel else None,
                        "version": model.baseModel.version if model.baseModel else None,
                    } if model.baseModel else None,
                }
                output_data["models"].append(model_info)

            # Add pagination info if available
            if response.links:
                output_data["_links"] = {}
                if response.links.page:
                    output_data["_links"]["page"] = {"href": response.links.page.href}
                if response.links.next:
                    output_data["_links"]["next"] = {"href": response.links.next.href}

            models_json = json.dumps(output_data, indent=2)

            # Print summary to console
            print(f"\n{'='*60}")
            print(f"FIREFLY CUSTOM MODELS LIST")
            print(f"{'='*60}")
            print(f"Total models: {output_data['total_count']}")
            print(f"Returned: {len(output_data['models'])}")
            print(f"Filter: publishedState={published_state}")
            print(f"Sort: {sort_by}")
            print(f"{'='*60}\n")

            for i, model in enumerate(output_data["models"], 1):
                print(f"{i}. {model['displayName'] or model['assetName']}")
                print(f"   ID: {model['assetId']}")
                print(f"   Training Mode: {model['trainingMode']}")
                if model['conceptId']:
                    print(f"   Concept ID: {model['conceptId']}")
                print(f"   State: {model['publishedState']}")
                print(f"   Created: {model['createdDate']}")
                if model['samplePrompt']:
                    print(f"   Sample Prompt: {model['samplePrompt'][:80]}...")
                print()

            return {"ui": {"text": [models_json]}, "result": (models_json,)}

        except Exception as e:
            error_msg = f"Error listing custom models: {str(e)}"
            print(f"\n[ERROR] {error_msg}\n")
            return {"ui": {"text": [error_msg]}, "result": (error_msg,)}

        finally:
            await client.close()


class FireflyGenerativeExpandNode:
    """
    Expand/outpaint an image to a larger size using Adobe Firefly.

    Features:
    - Cleaner code structure
    - Better batch processing
    - Enhanced error handling
    - Dual input support for images and masks
    - Placement controls for image positioning
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "debug_log")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/firefly"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Image to expand (auto-uploads to Firefly storage).",
                    },
                ),
                "image_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Image upload ID or presigned URL from another node.",
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": "Mask image to guide expansion (auto-uploads to Firefly storage). Cannot use with placement.",
                    },
                ),
                "mask_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Mask upload ID or presigned URL from another node. Cannot use with placement.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Optional prompt to guide the expansion.",
                    },
                ),
                "alignment_horizontal": (
                    ["center", "left", "right"],
                    {
                        "default": "center",
                        "tooltip": "Horizontal alignment of source image in output. Cannot use with mask.",
                    },
                ),
                "alignment_vertical": (
                    ["center", "top", "bottom"],
                    {
                        "default": "center",
                        "tooltip": "Vertical alignment of source image in output. Cannot use with mask.",
                    },
                ),
                "inset_left": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Left margin in pixels (0-3999). Leave empty for 0. Cannot use with mask.",
                    },
                ),
                "inset_top": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Top margin in pixels (0-3999). Leave empty for 0. Cannot use with mask.",
                    },
                ),
                "inset_right": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Right margin in pixels (0-3999). Leave empty for 0. Cannot use with mask.",
                    },
                ),
                "inset_bottom": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Bottom margin in pixels (0-3999). Leave empty for 0. Cannot use with mask.",
                    },
                ),
            },
            "required": {
                "output_width": (
                    "STRING",
                    {
                        "default": "2048",
                        "tooltip": "Width of expanded image in pixels (1-3999).",
                    },
                ),
                "output_height": (
                    "STRING",
                    {
                        "default": "2048",
                        "tooltip": "Height of expanded image in pixels (1-3999).",
                    },
                ),
                "num_variations": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Number of variations to generate.",
                    },
                ),
                "seed": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Seed(s) for reproducibility. Leave empty for random, or provide single seed (e.g. '12345') or multiple comma-separated seeds (e.g. '1,2,3,4').",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        output_width: str,
        output_height: str,
        num_variations: int,
        seed: str = "",
        image: Optional[torch.Tensor] = None,
        image_reference: str = "",
        mask: Optional[torch.Tensor] = None,
        mask_reference: str = "",
        prompt: str = "",
        alignment_horizontal: str = "center",
        alignment_vertical: str = "center",
        inset_left: str = "",
        inset_top: str = "",
        inset_right: str = "",
        inset_bottom: str = "",
        unique_id: Optional[str] = None,
    ):
        """Expand image using Firefly API."""
        # Parse width and height
        try:
            width_int = int(output_width.strip()) if output_width and output_width.strip() else 2048
            height_int = int(output_height.strip()) if output_height and output_height.strip() else 2048
        except ValueError:
            raise ValueError(f"Invalid output dimensions. Width: '{output_width}', Height: '{output_height}'. Must be integers between 1-3999.")

        # Parse inset values
        try:
            inset_left_int = int(inset_left.strip()) if inset_left and inset_left.strip() else 0
            inset_top_int = int(inset_top.strip()) if inset_top and inset_top.strip() else 0
            inset_right_int = int(inset_right.strip()) if inset_right and inset_right.strip() else 0
            inset_bottom_int = int(inset_bottom.strip()) if inset_bottom and inset_bottom.strip() else 0
        except ValueError:
            raise ValueError(f"Invalid inset values. Must be integers between 0-3999.")

        # Validate inputs
        if image is None and not image_reference:
            raise ValueError("Must provide either 'image' or 'image_reference'")
        if image is not None and image_reference:
            raise ValueError("Cannot provide both 'image' and 'image_reference' - choose only one")

        # Validate mask inputs
        has_mask = (mask is not None) or bool(mask_reference)
        has_placement = (alignment_horizontal != "center" or alignment_vertical != "center" or
                        inset_left_int > 0 or inset_top_int > 0 or inset_right_int > 0 or inset_bottom_int > 0)

        if has_mask and has_placement:
            raise ValueError("Cannot use both mask and placement parameters - choose only one")
        if mask is not None and mask_reference:
            raise ValueError("Cannot provide both 'mask' and 'mask_reference' - choose only one")

        # Parse seeds
        seeds_list = None
        if seed and seed.strip():
            try:
                seeds_list = [int(s.strip()) for s in seed.split(",") if s.strip()]
            except ValueError:
                raise ValueError(f"Invalid seed format: '{seed}'. Use integers separated by commas (e.g., '1,2,3,4').")

        client = await create_adobe_client()

        try:
            # Convert mask to RGB format if provided
            if mask is not None:
                if len(mask.shape) == 3:
                    mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
                elif len(mask.shape) == 2:
                    mask = mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)

            images = []
            all_urls = []
            total = image.shape[0] if image is not None else 1
            pbar = ProgressBar(total)

            # Build debug log for first iteration
            console_log = self._build_debug_log(
                width=width_int,
                height=height_int,
                num_variations=num_variations,
                seed=seed,
                prompt=prompt,
                image=image,
                image_reference=image_reference,
                mask=mask,
                mask_reference=mask_reference,
                has_mask=has_mask,
                has_placement=has_placement,
                alignment_horizontal=alignment_horizontal,
                alignment_vertical=alignment_vertical,
                inset_left_int=inset_left_int,
                inset_top_int=inset_top_int,
                inset_right_int=inset_right_int,
                inset_bottom_int=inset_bottom_int,
            )

            for i in range(total):
                # Determine image source
                if image is not None:
                    # Upload to Firefly storage and get upload ID
                    upload_id = await upload_image_to_firefly(image=image[i])
                    image_source = FireflyPublicBinaryInput(uploadId=upload_id)
                else:
                    # Use provided upload ID or presigned URL
                    if image_reference.lower().startswith("http"):
                        image_source = FireflyPublicBinaryInput(url=image_reference)
                    else:
                        image_source = FireflyPublicBinaryInput(uploadId=image_reference)

                # Determine mask source if provided
                mask_input = None
                if has_mask:
                    if mask is not None:
                        # Upload to Firefly storage and get upload ID
                        mask_upload_id = await upload_image_to_firefly(image=mask[i:i+1] if mask.shape[0] > 1 else mask)
                        mask_source = FireflyPublicBinaryInput(uploadId=mask_upload_id)
                    else:
                        # Use provided upload ID or presigned URL
                        if mask_reference.lower().startswith("http"):
                            mask_source = FireflyPublicBinaryInput(url=mask_reference)
                        else:
                            mask_source = FireflyPublicBinaryInput(uploadId=mask_reference)

                    from comfy_api_nodes.apis.firefly_api import FireflyInputMask
                    mask_input = FireflyInputMask(source=mask_source)

                # Build placement if specified and no mask
                # Note: alignment and inset are mutually exclusive
                placement = None
                if has_placement and not has_mask:
                    from comfy_api_nodes.apis.firefly_api import FireflyPlacement, FireflyPlacementAlignment, FireflyPlacementInset

                    # If ANY inset value is specified, use inset ONLY (omit alignment)
                    # Set all 4 inset values (unspecified ones default to 0)
                    if inset_left_int > 0 or inset_top_int > 0 or inset_right_int > 0 or inset_bottom_int > 0:
                        inset = FireflyPlacementInset(
                            left=inset_left_int,
                            top=inset_top_int,
                            right=inset_right_int,
                            bottom=inset_bottom_int
                        )
                        placement = FireflyPlacement(inset=inset)
                    else:
                        # No inset specified, use alignment ONLY
                        alignment = FireflyPlacementAlignment(
                            horizontal=alignment_horizontal,
                            vertical=alignment_vertical
                        )
                        placement = FireflyPlacement(alignment=alignment)

                # Prepare request
                request = ExpandImageRequest(
                    image=FireflyInputImage(
                        source=image_source
                    ),
                    size=FireflySize(width=width_int, height=height_int),
                    prompt=prompt if prompt else None,
                    mask=mask_input,
                    placement=placement,
                    numVariations=num_variations,
                    seeds=seeds_list,
                )

                # Submit and poll
                submit_endpoint = ApiEndpoint(
                    path="/v3/images/expand-async",
                    method=HttpMethod.POST,
                    request_model=ExpandImageRequest,
                    response_model=AsyncAcceptResponse,
                )

                submit_op = SynchronousOperation(
                    endpoint=submit_endpoint,
                    request=request,
                    api_base="https://firefly-api.adobe.io",
                )

                submit_response = await submit_op.execute(client=client)

                # Log response for first iteration only
                if i == 0:
                    console_log += f"\nResponse: 202 Accepted\n"
                    console_log += f"  jobId: {submit_response.jobId}\n"

                # Log polling section for first iteration only
                if i == 0:
                    console_log += f"\n{'='*55}\n"
                    console_log += f"GET /v3/status/{submit_response.jobId.split(':')[-1][:8]}...\n"
                    console_log += f"{'-'*55}\n"
                    console_log += f"Polling for job completion...\n"

                # Poll for completion
                poll_endpoint = ApiEndpoint(
                    path=f"/v3/status/{submit_response.jobId}",
                    method=HttpMethod.GET,
                    request_model=EmptyRequest,
                    response_model=AsyncTaskResponse,
                )

                poll_op = PollingOperation(
                    poll_endpoint=poll_endpoint,
                    request=EmptyRequest(),
                    completed_statuses=["succeeded"],
                    failed_statuses=["failed", "canceled"],
                    status_extractor=lambda x: x.status,
                    api_base="https://firefly-api.adobe.io",
                    poll_interval=5.0,
                    max_poll_attempts=120,
                    node_id=unique_id,
                )

                result = await poll_op.execute(client=client)

                # Log result for first iteration only
                if i == 0:
                    console_log += f"\nResponse: 200 OK\n"
                    console_log += f"  status: {result.status}\n"
                    console_log += f"  jobId: {result.jobId}\n"
                    console_log += f"  outputs: {len(result.outputs) if result.outputs else 0} image(s)\n"

                # Validate outputs
                if not result.outputs:
                    if i == 0:
                        console_log += f"\n{'='*55}\n"
                        console_log += f"ERROR: No outputs in response\n"
                        console_log += f"  status: {result.status}\n"
                    raise Exception("No outputs returned from Firefly API")

                # Log download start for first iteration only
                if i == 0:
                    console_log += f"\n{'='*55}\n"
                    console_log += f"Downloading {len(result.outputs)} output(s)...\n"

                output_bytesio, presigned_urls = await download_firefly_outputs(
                    result.outputs,
                    unique_id=unique_id,
                )

                # Log download completion for first iteration only
                if i == 0:
                    console_log += f"[OK] Downloaded {len(output_bytesio)} image(s)\n"
                    console_log += f"{'='*55}\n"

                all_urls.extend(presigned_urls)

                # Convert to tensors
                batch_images = []
                for bytesio in output_bytesio:
                    img = bytesio_to_image_tensor(bytesio)
                    if len(img.shape) < 4:
                        img = img.unsqueeze(0)
                    batch_images.append(img)

                images.append(torch.cat(batch_images, dim=0))
                pbar.update(1)

            images_tensor = torch.cat(images, dim=0)

            # Add presigned URLs to console log
            console_log += f"\nPresigned URLs (valid for 1 hour):\n"
            for idx, url in enumerate(all_urls, 1):
                console_log += f"  [{idx}] {url}\n"
            console_log += f"{'='*55}\n"

            # Return first URL (typically only 1 image is generated)
            image_url = all_urls[0] if len(all_urls) > 0 else ""

            return (images_tensor, image_url, console_log)

        finally:
            await client.close()

    def _build_debug_log(
        self,
        width: int,
        height: int,
        num_variations: int,
        seed: str,
        prompt: str,
        image: Optional[torch.Tensor],
        image_reference: str,
        mask: Optional[torch.Tensor],
        mask_reference: str,
        has_mask: bool,
        has_placement: bool,
        alignment_horizontal: str,
        alignment_vertical: str,
        inset_left_int: int,
        inset_top_int: int,
        inset_right_int: int,
        inset_bottom_int: int,
    ) -> str:
        """Build formatted debug log for console output."""
        log = "=" * 55 + "\n"
        log += "POST /v3/images/expand-async\n"
        log += "-" * 55 + "\n"
        log += f"Headers:\n"
        log += f"  x-model-version: image3\n"  # Expand uses default image3
        log += f"\nRequest Body:\n"

        # Image source
        if image is not None:
            log += f"  image: [UPLOADED IMAGE]\n"
        else:
            log += f"  image: {image_reference}\n"

        log += f"  size: {width}x{height}\n"
        log += f"  numVariations: {num_variations}\n"

        if seed and seed.strip():
            log += f"  seeds: [{seed}]\n"

        if prompt and prompt.strip():
            log += f"  prompt: {prompt[:50]}...\n" if len(prompt) > 50 else f"  prompt: {prompt}\n"

        # Mask info
        if has_mask:
            if mask is not None:
                log += f"  mask: [UPLOADED MASK]\n"
            else:
                log += f"  mask: {mask_reference}\n"

        # Placement info
        if has_placement:
            log += f"  placement:\n"
            log += f"    alignment:\n"
            log += f"      horizontal: {alignment_horizontal}\n"
            log += f"      vertical: {alignment_vertical}\n"
            if inset_left_int > 0 or inset_top_int > 0 or inset_right_int > 0 or inset_bottom_int > 0:
                log += f"    inset:\n"
                if inset_left_int > 0:
                    log += f"      left: {inset_left_int}\n"
                if inset_top_int > 0:
                    log += f"      top: {inset_top_int}\n"
                if inset_right_int > 0:
                    log += f"      right: {inset_right_int}\n"
                if inset_bottom_int > 0:
                    log += f"      bottom: {inset_bottom_int}\n"

        return log


class FireflyGenerateSimilarNode:
    """
    Generate similar images based on a reference image using Adobe Firefly.

    Features:
    - Cleaner code structure
    - Better batch processing
    - Enhanced error handling
    - Dual input support for images
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "image_url_2", "image_url_3", "image_url_4", "debug_log")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/firefly"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Reference image (auto-uploads to Firefly storage).",
                    },
                ),
                "image_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Image upload ID or presigned URL from another node.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional prompt to guide the generation.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Negative prompt to exclude unwanted elements.",
                    },
                ),
            },
            "required": {
                "num_variations": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Number of variations to generate.",
                    },
                ),
                "seed": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Seed(s) for reproducibility. Leave empty for random, or provide single seed (e.g. '12345') or multiple comma-separated seeds (e.g. '1,2,3,4').",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        num_variations: int,
        seed: str = "",
        image: Optional[torch.Tensor] = None,
        image_reference: str = "",
        prompt: str = "",
        negative_prompt: str = "",
        unique_id: Optional[str] = None,
    ):
        """Generate similar images using Firefly API."""
        # Validate inputs
        if image is None and not image_reference:
            raise ValueError("Must provide either 'image' or 'image_reference'")
        if image is not None and image_reference:
            raise ValueError("Cannot provide both 'image' and 'image_reference' - choose only one")

        # Parse seeds
        seeds_list = None
        if seed and seed.strip():
            try:
                seeds_list = [int(s.strip()) for s in seed.split(",") if s.strip()]
            except ValueError:
                raise ValueError(f"Invalid seed format: '{seed}'. Use integers separated by commas (e.g., '1,2,3,4').")

        client = await create_adobe_client()

        try:
            images = []
            all_urls = []
            total = image.shape[0] if image is not None else 1
            pbar = ProgressBar(total)

            # Build debug log for first iteration
            console_log = self._build_debug_log(
                num_variations=num_variations,
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                image_reference=image_reference,
            )

            for i in range(total):
                # Determine image source
                if image is not None:
                    # Upload to Firefly storage and get upload ID
                    upload_id = await upload_image_to_firefly(image=image[i])
                    image_source = FireflyPublicBinaryInput(uploadId=upload_id)
                else:
                    # Use provided upload ID or presigned URL
                    if image_reference.lower().startswith("http"):
                        image_source = FireflyPublicBinaryInput(url=image_reference)
                    else:
                        image_source = FireflyPublicBinaryInput(uploadId=image_reference)

                # Prepare request
                request = GenerateSimilarImagesRequest(
                    image=FireflyInputImage(
                        source=image_source
                    ),
                    prompt=prompt if prompt else None,
                    negativePrompt=negative_prompt if negative_prompt else None,
                    numVariations=num_variations,
                    seeds=seeds_list,
                )

                # Submit and poll
                submit_endpoint = ApiEndpoint(
                    path="/v3/images/generate-similar-async",
                    method=HttpMethod.POST,
                    request_model=GenerateSimilarImagesRequest,
                    response_model=AsyncAcceptResponse,
                )

                submit_op = SynchronousOperation(
                    endpoint=submit_endpoint,
                    request=request,
                    api_base="https://firefly-api.adobe.io",
                )

                submit_response = await submit_op.execute(client=client)

                # Log response for first iteration only
                if i == 0:
                    console_log += f"\nResponse: 202 Accepted\n"
                    console_log += f"  jobId: {submit_response.jobId}\n"

                # Log polling section for first iteration only
                if i == 0:
                    console_log += f"\n{'='*55}\n"
                    console_log += f"GET /v3/status/{submit_response.jobId.split(':')[-1][:8]}...\n"
                    console_log += f"{'-'*55}\n"
                    console_log += f"Polling for job completion...\n"

                # Poll for completion
                poll_endpoint = ApiEndpoint(
                    path=f"/v3/status/{submit_response.jobId}",
                    method=HttpMethod.GET,
                    request_model=EmptyRequest,
                    response_model=AsyncTaskResponse,
                )

                poll_op = PollingOperation(
                    poll_endpoint=poll_endpoint,
                    request=EmptyRequest(),
                    completed_statuses=["succeeded"],
                    failed_statuses=["failed", "canceled"],
                    status_extractor=lambda x: x.status,
                    api_base="https://firefly-api.adobe.io",
                    poll_interval=5.0,
                    max_poll_attempts=120,
                    node_id=unique_id,
                )

                result = await poll_op.execute(client=client)

                # Log result for first iteration only
                if i == 0:
                    console_log += f"\nResponse: 200 OK\n"
                    console_log += f"  status: {result.status}\n"
                    console_log += f"  jobId: {result.jobId}\n"
                    console_log += f"  outputs: {len(result.outputs) if result.outputs else 0} image(s)\n"

                # Validate outputs
                if not result.outputs:
                    if i == 0:
                        console_log += f"\n{'='*55}\n"
                        console_log += f"ERROR: No outputs in response\n"
                        console_log += f"  status: {result.status}\n"
                    raise Exception("No outputs returned from Firefly API")

                # Log download start for first iteration only
                if i == 0:
                    console_log += f"\n{'='*55}\n"
                    console_log += f"Downloading {len(result.outputs)} output(s)...\n"

                output_bytesio, presigned_urls = await download_firefly_outputs(
                    result.outputs,
                    unique_id=unique_id,
                )

                # Log download completion for first iteration only
                if i == 0:
                    console_log += f"[OK] Downloaded {len(output_bytesio)} image(s)\n"
                    console_log += f"{'='*55}\n"

                all_urls.extend(presigned_urls)

                # Convert to tensors
                batch_images = []
                for bytesio in output_bytesio:
                    img = bytesio_to_image_tensor(bytesio)
                    if len(img.shape) < 4:
                        img = img.unsqueeze(0)
                    batch_images.append(img)

                images.append(torch.cat(batch_images, dim=0))
                pbar.update(1)

            images_tensor = torch.cat(images, dim=0)

            # Add presigned URLs to console log
            console_log += f"\nPresigned URLs (valid for 1 hour):\n"
            for idx, url in enumerate(all_urls, 1):
                console_log += f"  [{idx}] {url}\n"
            console_log += f"{'='*55}\n"

            # Return all 4 URLs (pad with empty strings if fewer generated)
            image_url = all_urls[0] if len(all_urls) > 0 else ""
            image_url_2 = all_urls[1] if len(all_urls) > 1 else ""
            image_url_3 = all_urls[2] if len(all_urls) > 2 else ""
            image_url_4 = all_urls[3] if len(all_urls) > 3 else ""

            return (images_tensor, image_url, image_url_2, image_url_3, image_url_4, console_log)

        finally:
            await client.close()

    def _build_debug_log(
        self,
        num_variations: int,
        seed: str,
        prompt: str,
        negative_prompt: str,
        image: Optional[torch.Tensor],
        image_reference: str,
    ) -> str:
        """Build formatted debug log for console output."""
        log = "=" * 55 + "\n"
        log += "POST /v3/images/generate-similar-async\n"
        log += "-" * 55 + "\n"
        log += f"Headers:\n"
        log += f"  x-model-version: image3\n"  # Generate Similar uses default image3
        log += f"\nRequest Body:\n"

        # Image source
        if image is not None:
            log += f"  image: [UPLOADED IMAGE]\n"
        else:
            log += f"  image: {image_reference}\n"

        log += f"  numVariations: {num_variations}\n"

        if seed and seed.strip():
            log += f"  seeds: [{seed}]\n"

        if prompt and prompt.strip():
            log += f"  prompt: {prompt[:50]}...\n" if len(prompt) > 50 else f"  prompt: {prompt}\n"

        if negative_prompt and negative_prompt.strip():
            log += f"  negativePrompt: {negative_prompt[:30]}...\n" if len(negative_prompt) > 30 else f"  negativePrompt: {negative_prompt}\n"

        return log


class FireflyGenerateObjectCompositeNode:
    """
    Generate and composite an object into a background scene using Adobe Firefly.

    Features:
    - Cleaner code structure
    - Better batch processing
    - Enhanced error handling
    - Dual input support for images and masks
    """

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url_1", "image_url_2", "image_url_3", "image_url_4", "debug_log")
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/firefly"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Background image (auto-uploads to Firefly storage).",
                    },
                ),
                "image_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Image upload ID or presigned URL from another node.",
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": "Mask for object placement (auto-uploads to Firefly storage).",
                    },
                ),
                "mask_image": (
                    "IMAGE",
                    {
                        "tooltip": "Mask image for object placement (auto-uploads to Firefly storage).",
                    },
                ),
                "mask_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Mask upload ID or presigned URL from another node.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt describing the object to generate.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Negative prompt to exclude unwanted elements.",
                    },
                ),
                "model_version": (
                    ["image3", "image3_fast"],
                    {
                        "default": "image3",
                        "tooltip": "Firefly model version (Object Composite only supports image3 and image3_fast).",
                    },
                ),
                "content_class": (
                    ["photo", "art"],
                    {
                        "default": "photo",
                        "tooltip": "Content class: 'photo' for photorealistic, 'art' for artistic style.",
                    },
                ),
                "size": (
                    FIREFLY_ASPECT_RATIOS,
                    {
                        "default": "2048x2048 (1:1)",
                        "tooltip": "Image size and aspect ratio.",
                    },
                ),
                "alignment_horizontal": (
                    ["none", "center", "left", "right"],
                    {
                        "default": "none",
                        "tooltip": "Horizontal alignment of the placed object.",
                    },
                ),
                "alignment_vertical": (
                    ["none", "center", "top", "bottom"],
                    {
                        "default": "none",
                        "tooltip": "Vertical alignment of the placed object.",
                    },
                ),
                "inset_left": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Left margin in pixels. Leave empty for 0.",
                    },
                ),
                "inset_top": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Top margin in pixels. Leave empty for 0.",
                    },
                ),
                "inset_right": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Right margin in pixels. Leave empty for 0.",
                    },
                ),
                "inset_bottom": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Bottom margin in pixels. Leave empty for 0.",
                    },
                ),
                "style_preset": (
                    FIREFLY_STYLE_PRESETS,
                    {
                        "default": "None",
                        "tooltip": "Style preset to apply to generation.",
                    },
                ),
                "style_image": (
                    "IMAGE",
                    {
                        "tooltip": "Style reference image (auto-uploads to Firefly storage).",
                    },
                ),
                "style_reference": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Style reference image upload ID or presigned URL from another node.",
                    },
                ),
                "style_strength": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Style reference strength (0-100). Leave empty for default.",
                    },
                ),
            },
            "required": {
                "num_variations": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Number of variations to generate.",
                    },
                ),
                "seed": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Seed(s) for reproducibility. Leave empty for random, or provide single seed (e.g. '12345') or multiple comma-separated seeds (e.g. '1,2,3,4').",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    async def api_call(
        self,
        prompt: str,
        num_variations: int,
        seed: str = "",
        image: Optional[torch.Tensor] = None,
        image_reference: str = "",
        mask: Optional[torch.Tensor] = None,
        mask_image: Optional[torch.Tensor] = None,
        mask_reference: str = "",
        negative_prompt: str = "",
        model_version: str = "image3",
        content_class: str = "photo",
        size: str = "2048x2048 (1:1)",
        alignment_horizontal: str = "none",
        alignment_vertical: str = "none",
        inset_left: str = "",
        inset_top: str = "",
        inset_right: str = "",
        inset_bottom: str = "",
        style_preset: str = "None",
        style_reference: str = "",
        style_image: Optional[torch.Tensor] = None,
        style_strength: str = "",
        unique_id: Optional[str] = None,
    ):
        """Generate object composite using Firefly API."""
        validate_string(prompt, strip_whitespace=False, max_length=1024)

        # Convert human-readable style preset to API enum value
        style_preset_enum = FIREFLY_STYLE_PRESET_MAP.get(style_preset, "none")

        # Parse size
        size_parts = size.split(" ")[0].split("x")
        width = int(size_parts[0])
        height = int(size_parts[1])

        # Parse inset values
        try:
            inset_left_int = int(inset_left.strip()) if inset_left and inset_left.strip() else 0
            inset_top_int = int(inset_top.strip()) if inset_top and inset_top.strip() else 0
            inset_right_int = int(inset_right.strip()) if inset_right and inset_right.strip() else 0
            inset_bottom_int = int(inset_bottom.strip()) if inset_bottom and inset_bottom.strip() else 0
        except ValueError:
            raise ValueError(f"Invalid inset values. Must be integers.")

        # Validate inputs
        if image is None and not image_reference:
            raise ValueError("Must provide either 'image' or 'image_reference'")
        if image is not None and image_reference:
            raise ValueError("Cannot provide both 'image' and 'image_reference' - choose only one")

        # Validate mask inputs (allow exactly one of three options)
        mask_inputs_provided = sum([
            mask is not None,
            mask_image is not None,
            bool(mask_reference),
        ])
        if mask_inputs_provided == 0:
            raise ValueError("Must provide one of: 'mask', 'mask_image', or 'mask_reference'")
        if mask_inputs_provided > 1:
            raise ValueError("Cannot provide multiple mask inputs - choose only one: 'mask', 'mask_image', or 'mask_reference'")

        # Validate style inputs (allow at most one)
        style_inputs_provided = sum([
            style_image is not None,
            bool(style_reference),
        ])
        if style_inputs_provided > 1:
            raise ValueError("Cannot provide multiple style inputs - choose only one: 'style_image' or 'style_reference'")

        # Parse seeds
        seeds_list = None
        if seed and seed.strip():
            try:
                seeds_list = [int(s.strip()) for s in seed.split(",") if s.strip()]
            except ValueError:
                raise ValueError(f"Invalid seed format: '{seed}'. Use integers separated by commas (e.g., '1,2,3,4').")

        client = await create_adobe_client(model_version=model_version)

        try:
            # Prepare mask if provided as tensor
            if mask is not None:
                if image is not None:
                    # Resize mask to match image dimensions
                    mask = resize_mask_to_image(mask, image, allow_gradient=False, add_channel_dim=True)
                else:
                    # Convert mask to RGB format (expand [B,H,W] to [B,H,W,3])
                    if len(mask.shape) == 3:
                        mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
                    elif len(mask.shape) == 2:
                        mask = mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)

            images = []
            all_urls = []
            total = image.shape[0] if image is not None else 1
            pbar = ProgressBar(total)

            for i in range(total):
                # Determine image source
                if image is not None:
                    # Upload to Firefly storage and get upload ID
                    upload_id = await upload_image_to_firefly(image=image[i])
                    image_source = FireflyPublicBinaryInput(uploadId=upload_id)
                else:
                    # Use provided upload ID or presigned URL
                    if image_reference.lower().startswith("http"):
                        image_source = FireflyPublicBinaryInput(url=image_reference)
                    else:
                        image_source = FireflyPublicBinaryInput(uploadId=image_reference)

                # Determine mask source
                if mask is not None:
                    # Upload MASK tensor (now in RGB format) to Firefly storage and get upload ID
                    mask_upload_id = await upload_image_to_firefly(image=mask[i:i+1])
                    mask_source = FireflyPublicBinaryInput(uploadId=mask_upload_id)
                elif mask_image is not None:
                    # Upload IMAGE tensor to Firefly storage and get upload ID
                    mask_upload_id = await upload_image_to_firefly(image=mask_image[i])
                    mask_source = FireflyPublicBinaryInput(uploadId=mask_upload_id)
                else:
                    # Use provided upload ID or presigned URL
                    if mask_reference.lower().startswith("http"):
                        mask_source = FireflyPublicBinaryInput(url=mask_reference)
                    else:
                        mask_source = FireflyPublicBinaryInput(uploadId=mask_reference)

                # Build style configuration
                style_config = None

                # Check if any style parameters are provided (excluding "None" and "none")
                has_style_preset = style_preset_enum and style_preset_enum != "none" and style_preset != "None"
                has_style_image_or_ref = style_image is not None or style_reference
                has_style_strength = style_strength and style_strength.strip()

                if has_style_preset or has_style_image_or_ref or has_style_strength:
                    # Build style_kwargs conditionally
                    style_kwargs = {}

                    # Add imageReference if provided
                    if style_image is not None or style_reference:
                        if style_image is not None:
                            # Upload to Firefly storage and get upload ID
                            style_upload_id = await upload_image_to_firefly(
                                image=style_image[0] if len(style_image.shape) == 4 else style_image,
                            )
                            style_ref = FireflyStyleImageReferenceV3(
                                source=FireflyPublicBinaryInput(uploadId=style_upload_id)
                            )
                        else:
                            # Use provided upload ID or presigned URL from node connection
                            if style_reference.lower().startswith("http"):
                                style_ref = FireflyStyleImageReferenceV3(
                                    source=FireflyPublicBinaryInput(url=style_reference)
                                )
                            else:
                                style_ref = FireflyStyleImageReferenceV3(
                                    source=FireflyPublicBinaryInput(uploadId=style_reference)
                                )
                        style_kwargs["imageReference"] = style_ref

                    # Add presets if provided (only if not "None" or "none")
                    if has_style_preset:
                        style_kwargs["presets"] = [style_preset_enum]

                    # Add strength if provided
                    if has_style_strength:
                        try:
                            style_strength_int = int(style_strength.strip())
                            style_kwargs["strength"] = style_strength_int
                        except ValueError:
                            raise ValueError(f"Invalid style_strength: '{style_strength}'. Must be an integer between 0-100.")

                    # Create style config if we have any parameters
                    if style_kwargs:
                        style_config = FireflyStyles(**style_kwargs)

                # Build placement configuration
                placement_config = None
                has_alignment = (alignment_horizontal != "none" or alignment_vertical != "none")
                has_inset = (inset_left_int > 0 or inset_top_int > 0 or inset_right_int > 0 or inset_bottom_int > 0)
                has_placement = has_alignment or has_inset

                if has_placement:
                    placement_dict = {}

                    # Only create alignment if not "none"
                    if has_alignment:
                        alignment_dict = {}
                        if alignment_horizontal != "none":
                            alignment_dict["horizontal"] = alignment_horizontal
                        if alignment_vertical != "none":
                            alignment_dict["vertical"] = alignment_vertical
                        if alignment_dict:
                            placement_dict["alignment"] = alignment_dict

                    # Only include inset if any value is non-zero
                    if has_inset:
                        inset_dict = {}
                        if inset_left_int > 0:
                            inset_dict["left"] = inset_left_int
                        if inset_top_int > 0:
                            inset_dict["top"] = inset_top_int
                        if inset_right_int > 0:
                            inset_dict["right"] = inset_right_int
                        if inset_bottom_int > 0:
                            inset_dict["bottom"] = inset_bottom_int
                        if inset_dict:
                            placement_dict["inset"] = inset_dict

                    placement_config = placement_dict if placement_dict else None

                # Prepare request
                request = GenerateObjectCompositeRequest(
                    image=FireflyInputImage(
                        source=image_source
                    ),
                    mask=FireflyInputMask(
                        source=mask_source
                    ),
                    prompt=prompt,
                    negativePrompt=negative_prompt if negative_prompt else None,
                    contentClass=FireflyContentClass(content_class),
                    size=FireflySize(width=width, height=height),
                    style=style_config,
                    placement=placement_config,
                    numVariations=num_variations,
                    seeds=seeds_list,
                )

                # Submit and poll
                submit_endpoint = ApiEndpoint(
                    path="/v3/images/generate-object-composite-async",
                    method=HttpMethod.POST,
                    request_model=GenerateObjectCompositeRequest,
                    response_model=AsyncAcceptResponse,
                )

                submit_op = SynchronousOperation(
                    endpoint=submit_endpoint,
                    request=request,
                    api_base="https://firefly-api.adobe.io",
                )

                submit_response = await submit_op.execute(client=client)

                # Poll for completion
                poll_endpoint = ApiEndpoint(
                    path=f"/v3/status/{submit_response.jobId}",
                    method=HttpMethod.GET,
                    request_model=EmptyRequest,
                    response_model=AsyncTaskResponse,
                )

                poll_op = PollingOperation(
                    poll_endpoint=poll_endpoint,
                    request=EmptyRequest(),
                    completed_statuses=["succeeded"],
                    failed_statuses=["failed", "canceled"],
                    status_extractor=lambda x: x.status,
                    api_base="https://firefly-api.adobe.io",
                    poll_interval=5.0,
                    max_poll_attempts=120,
                    node_id=unique_id,
                )

                result = await poll_op.execute(client=client)

                # Download outputs
                if not result.outputs:
                    raise Exception("No outputs returned from Firefly API")

                output_bytesio, presigned_urls = await download_firefly_outputs(
                    result.outputs,
                    unique_id=unique_id,
                )

                all_urls.extend(presigned_urls)

                # Convert to tensors
                batch_images = []
                for bytesio in output_bytesio:
                    img = bytesio_to_image_tensor(bytesio)
                    if len(img.shape) < 4:
                        img = img.unsqueeze(0)
                    batch_images.append(img)

                images.append(torch.cat(batch_images, dim=0))
                pbar.update(1)

            images_tensor = torch.cat(images, dim=0)

            # Build comprehensive debug log
            console_log = self._build_debug_log(
                model_version=model_version,
                prompt=prompt,
                num_variations=num_variations,
                seed=seed,
                negative_prompt=negative_prompt,
                content_class=content_class,
                width=width,
                height=height,
                image=image,
                image_reference=image_reference,
                mask=mask,
                mask_image=mask_image,
                mask_reference=mask_reference,
                alignment_horizontal=alignment_horizontal,
                alignment_vertical=alignment_vertical,
                inset_left_int=inset_left_int,
                inset_top_int=inset_top_int,
                inset_right_int=inset_right_int,
                inset_bottom_int=inset_bottom_int,
                style_config=style_config,
                style_image=style_image,
                style_reference=style_reference,
                style_strength=style_strength,
                style_preset=style_preset,
                total_processed=total,
                total_urls=len(all_urls),
            )

            # Add presigned URLs to console log
            console_log += f"\nPresigned URLs (valid for 1 hour):\n"
            for idx, url in enumerate(all_urls, 1):
                console_log += f"  [{idx}] {url}\n"
            console_log += f"{'='*55}\n"

            # Split URLs into individual outputs (up to 4)
            image_url_1 = all_urls[0] if len(all_urls) > 0 else ""
            image_url_2 = all_urls[1] if len(all_urls) > 1 else ""
            image_url_3 = all_urls[2] if len(all_urls) > 2 else ""
            image_url_4 = all_urls[3] if len(all_urls) > 3 else ""

            return (images_tensor, image_url_1, image_url_2, image_url_3, image_url_4, console_log)

        finally:
            await client.close()

    def _build_debug_log(
        self,
        model_version: str,
        prompt: str,
        num_variations: int,
        seed: str,
        negative_prompt: str,
        content_class: str,
        width: int,
        height: int,
        image: Optional[torch.Tensor],
        image_reference: str,
        mask: Optional[torch.Tensor],
        mask_image: Optional[torch.Tensor],
        mask_reference: str,
        alignment_horizontal: str,
        alignment_vertical: str,
        inset_left_int: int,
        inset_top_int: int,
        inset_right_int: int,
        inset_bottom_int: int,
        style_config: Any,
        style_image: Optional[torch.Tensor],
        style_reference: str,
        style_strength: str,
        style_preset: str,
        total_processed: int,
        total_urls: int,
    ) -> str:
        """Build formatted debug log for console output."""
        log = "=" * 55 + "\n"
        log += "POST /v3/images/generate-object-composite-async\n"
        log += "-" * 55 + "\n"
        log += f"Headers:\n"
        log += f"  x-model-version: {model_version}\n"
        log += f"\nRequest Body:\n"

        log += f"  prompt: {prompt[:50]}...\n" if len(prompt) > 50 else f"  prompt: {prompt}\n"

        # Image source
        if image is not None:
            log += f"  image: [UPLOADED IMAGE]\n"
        else:
            log += f"  image: {image_reference}\n"

        # Mask source
        if mask is not None:
            log += f"  mask: [UPLOADED MASK]\n"
        elif mask_image is not None:
            log += f"  mask: [UPLOADED MASK IMAGE]\n"
        else:
            log += f"  mask: {mask_reference}\n"

        log += f"  contentClass: {content_class}\n"
        log += f"  size: {width}x{height}\n"
        log += f"  numVariations: {num_variations}\n"

        if seed and seed.strip():
            log += f"  seeds: [{seed}]\n"

        if negative_prompt and negative_prompt.strip():
            log += f"  negativePrompt: {negative_prompt[:30]}...\n" if len(negative_prompt) > 30 else f"  negativePrompt: {negative_prompt}\n"

        # Placement info
        has_alignment = (alignment_horizontal != "none" or alignment_vertical != "none")
        has_inset = (inset_left_int > 0 or inset_top_int > 0 or inset_right_int > 0 or inset_bottom_int > 0)
        has_placement = has_alignment or has_inset

        if has_placement:
            log += f"  placement:\n"
            if has_alignment:
                log += f"    alignment:\n"
                if alignment_horizontal != "none":
                    log += f"      horizontal: {alignment_horizontal}\n"
                if alignment_vertical != "none":
                    log += f"      vertical: {alignment_vertical}\n"
            if has_inset:
                log += f"    inset:\n"
                if inset_left_int > 0:
                    log += f"      left: {inset_left_int}\n"
                if inset_top_int > 0:
                    log += f"      top: {inset_top_int}\n"
                if inset_right_int > 0:
                    log += f"      right: {inset_right_int}\n"
                if inset_bottom_int > 0:
                    log += f"      bottom: {inset_bottom_int}\n"

        # Style info
        if style_config:
            log += f"  style:\n"
            if style_image is not None:
                log += f"    uploadId: [UPLOADED IMAGE]\n"
            elif style_reference:
                log += f"    uploadId: {style_reference}\n"
            if style_strength and style_strength.strip():
                log += f"    strength: {style_strength}\n"
            if style_preset and style_preset != "None" and style_preset != "none":
                log += f"    preset: {style_preset}\n"

        log += f"\nProcessed: {total_processed} image(s)\n"
        log += f"Generated: {total_urls} output(s)\n"
        log += "=" * 55 + "\n"
