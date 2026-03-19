"""
Adobe Firefly API models and types.

This module contains Pydantic models for interacting with Adobe Firefly API v3.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class FireflyContentClass(str, Enum):
    """Content class for image generation"""
    PHOTO = "photo"
    ART = "art"


class FireflyTaskStatus(str, Enum):
    """Status of async tasks"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class FireflyPromptBiasingLocale(str, Enum):
    """Locale codes for prompt biasing"""
    EN_US = "en-US"
    DE_DE = "de-DE"
    ES_ES = "es-ES"
    FR_FR = "fr-FR"
    IT_IT = "it-IT"
    JA_JP = "ja-JP"
    PT_BR = "pt-BR"
    AUTO = "AUTO"


class FireflyImageFormat(str, Enum):
    """Image format for outputs"""
    IMAGE_JPEG = "image/jpeg"
    IMAGE_PNG = "image/png"


class FireflyVideoFormat(str, Enum):
    """Video format for outputs"""
    VIDEO_MP4 = "video/mp4"


class FireflyStyleImageReference(str, Enum):
    """Style type for image references"""
    AUTO = "auto"
    IMAGE = "image"
    TEXT = "text"


class FireflyAlignment(str, Enum):
    """Alignment options for fill and expand"""
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    CENTER = "center"


class FireflyUpsamplerType(str, Enum):
    """Upsampler type for image4_custom"""
    DEFAULT = "default"
    LOW_CREATIVITY = "low_creativity"


# ============================================================================
# Common Models
# ============================================================================

class FireflySize(BaseModel):
    """Image size specification"""
    width: int = Field(..., description="Width of the image in pixels", ge=1, le=3999)
    height: int = Field(..., description="Height of the image in pixels", ge=1, le=3999)


class FireflyPublicBinaryInput(BaseModel):
    """Reference to an image via presigned URL or upload ID"""
    uploadId: Optional[str] = Field(None, description="Upload ID from storage API")
    url: Optional[str] = Field(None, description="Presigned URL to the image")


class FireflyInputImage(BaseModel):
    """Input image with source reference"""
    source: FireflyPublicBinaryInput = Field(..., description="Reference to the image")


class FireflyInputMask(BaseModel):
    """Input mask with source reference"""
    source: FireflyPublicBinaryInput = Field(..., description="Reference to the mask image")


class FireflyStyleReference(BaseModel):
    """Style reference image for generation"""
    imageReference: Optional[FireflyStyleImageReference] = Field(None, description="Style type")
    strength: Optional[int] = Field(None, description="Style strength", ge=0, le=100)


class FireflyStructureReference(BaseModel):
    """Structure reference image for generation"""
    strength: Optional[int] = Field(None, description="Structure strength", ge=0, le=100)


class FireflyStyleImageReferenceV3(BaseModel):
    """Style image reference for V3"""
    source: FireflyPublicBinaryInput = Field(..., description="Style image source")


class FireflyStyles(BaseModel):
    """Style configuration"""
    imageReference: Optional[FireflyStyleImageReferenceV3] = Field(None, description="Style image reference")
    presets: Optional[List[str]] = Field(None, description="Style presets")
    strength: Optional[int] = Field(None, description="Style strength", ge=0, le=100)


class FireflyStructureImageReferenceV3(BaseModel):
    """Structure image reference for V3"""
    source: FireflyPublicBinaryInput = Field(..., description="Structure image source")


class FireflyStructure(BaseModel):
    """Structure configuration"""
    imageReference: Optional[FireflyStructureImageReferenceV3] = Field(None, description="Structure reference")
    strength: Optional[int] = Field(None, description="Structure strength", ge=0, le=100)


class FireflyPlacementAlignment(BaseModel):
    """Alignment configuration for placement"""
    horizontal: Optional[str] = Field(None, description="Horizontal alignment (left, right, center)")
    vertical: Optional[str] = Field(None, description="Vertical alignment (top, bottom, center)")


class FireflyPlacementInset(BaseModel):
    """Inset configuration for placement"""
    left: Optional[int] = Field(None, description="Left inset in pixels", ge=0)
    top: Optional[int] = Field(None, description="Top inset in pixels", ge=0)
    right: Optional[int] = Field(None, description="Right inset in pixels", ge=0)
    bottom: Optional[int] = Field(None, description="Bottom inset in pixels", ge=0)


class FireflyPlacement(BaseModel):
    """Placement configuration for expand and object composite"""
    alignment: Optional[FireflyPlacementAlignment] = Field(None, description="Alignment configuration")
    inset: Optional[FireflyPlacementInset] = Field(None, description="Inset configuration")


class FireflyOutputImage(BaseModel):
    """Output image with seed and URL"""
    seed: Optional[int] = Field(None, description="Seed used for generation")
    image: Optional[FireflyPublicBinaryInput] = Field(None, description="Generated image reference")


class FireflyOutputVideo(BaseModel):
    """Output video with seed and URL"""
    seed: Optional[int] = Field(None, description="Seed used for generation")
    video: Optional[FireflyPublicBinaryInput] = Field(None, description="Generated video reference")


# ============================================================================
# Request Models - Text to Image
# ============================================================================

class GenerateImagesRequest(BaseModel):
    """Request for text-to-image generation"""
    prompt: str = Field(..., description="Text prompt for generation", max_length=1024)
    contentClass: Optional[FireflyContentClass] = Field(FireflyContentClass.PHOTO, description="Content class")
    customModelId: Optional[str] = Field(None, description="Custom model ID for custom model versions")
    size: Optional[FireflySize] = Field(None, description="Output size")
    numVariations: Optional[int] = Field(1, description="Number of variations", ge=1, le=4)
    seeds: Optional[List[int]] = Field(None, description="Seeds for generation")
    negativePrompt: Optional[str] = Field(None, description="Negative prompt", max_length=1024)
    promptBiasingLocaleCode: Optional[str] = Field(None, description="Locale for prompt (e.g., en-US)")
    style: Optional[FireflyStyles] = Field(None, description="Style reference")
    structure: Optional[FireflyStructure] = Field(None, description="Structure reference")
    visualIntensity: Optional[int] = Field(None, description="Visual intensity", ge=2, le=10)
    upsamplerType: Optional[FireflyUpsamplerType] = Field(None, description="Upsampler type for image4_custom")


# ============================================================================
# Request Models - Generative Fill
# ============================================================================

class FillImageRequest(BaseModel):
    """Request for generative fill"""
    image: FireflyInputImage = Field(..., description="Input image")
    mask: FireflyInputMask = Field(..., description="Mask for fill area")
    prompt: Optional[str] = Field(None, description="Text prompt for fill", max_length=1024)
    negativePrompt: Optional[str] = Field(None, description="Negative prompt", max_length=1024)
    size: Optional[FireflySize] = Field(None, description="Output size")
    numVariations: Optional[int] = Field(1, description="Number of variations", ge=1, le=4)
    seeds: Optional[List[int]] = Field(None, description="Seeds for generation")
    promptBiasingLocaleCode: Optional[FireflyPromptBiasingLocale] = Field(None, description="Locale for prompt")


# ============================================================================
# Request Models - Generative Expand
# ============================================================================

class ExpandImageRequest(BaseModel):
    """Request for generative expand"""
    image: FireflyInputImage = Field(..., description="Input image")
    size: FireflySize = Field(..., description="Output size")
    prompt: Optional[str] = Field(None, description="Text prompt for expansion", max_length=1024)
    negativePrompt: Optional[str] = Field(None, description="Negative prompt", max_length=1024)
    mask: Optional[FireflyInputMask] = Field(None, description="Optional mask for expansion")
    placement: Optional[FireflyPlacement] = Field(None, description="Placement configuration")
    numVariations: Optional[int] = Field(1, description="Number of variations", ge=1, le=4)
    seeds: Optional[List[int]] = Field(None, description="Seeds for generation")
    promptBiasingLocaleCode: Optional[FireflyPromptBiasingLocale] = Field(None, description="Locale for prompt")


# ============================================================================
# Request Models - Generate Similar
# ============================================================================

class GenerateSimilarImagesRequest(BaseModel):
    """Request for generating similar images"""
    image: FireflyInputImage = Field(..., description="Reference image")
    prompt: Optional[str] = Field(None, description="Text prompt", max_length=1024)
    negativePrompt: Optional[str] = Field(None, description="Negative prompt", max_length=1024)
    size: Optional[FireflySize] = Field(None, description="Output size")
    numVariations: Optional[int] = Field(1, description="Number of variations", ge=1, le=4)
    seeds: Optional[List[int]] = Field(None, description="Seeds for generation")
    promptBiasingLocaleCode: Optional[FireflyPromptBiasingLocale] = Field(None, description="Locale for prompt")


# ============================================================================
# Request Models - Generate Object Composite
# ============================================================================

class GenerateObjectCompositeRequest(BaseModel):
    """Request for generating object composite"""
    image: FireflyInputImage = Field(..., description="Background scene image")
    mask: FireflyInputMask = Field(..., description="Mask for object placement")
    prompt: str = Field(..., description="Text prompt for object", max_length=1024)
    negativePrompt: Optional[str] = Field(None, description="Negative prompt", max_length=1024)
    contentClass: Optional[FireflyContentClass] = Field(None, description="Content class")
    size: Optional[FireflySize] = Field(None, description="Output size")
    style: Optional[FireflyStyles] = Field(None, description="Style configuration")
    placement: Optional[FireflyPlacement] = Field(None, description="Placement configuration")
    numVariations: Optional[int] = Field(1, description="Number of variations", ge=1, le=4)
    seeds: Optional[List[int]] = Field(None, description="Seeds for generation")
    promptBiasingLocaleCode: Optional[FireflyPromptBiasingLocale] = Field(None, description="Locale for prompt")


# ============================================================================
# Request Models - Video Generation
# ============================================================================

class FireflyVideoSize(BaseModel):
    """Video dimensions"""
    width: int = Field(..., description="Width in pixels", ge=1, le=8192)
    height: int = Field(..., description="Height in pixels", ge=1, le=8192)


class FireflyVideoSettings(BaseModel):
    """Camera and shot control settings for video"""
    cameraMotion: Optional[str] = Field(None, description="Camera motion: 'camera pan left', 'camera pan right', 'camera zoom in', 'camera zoom out', 'camera tilt up', 'camera tilt down', 'camera locked down', 'camera handheld'")
    promptStyle: Optional[str] = Field(None, description="Style: 'anime', '3d', 'fantasy', 'cinematic', 'claymation', 'line art', 'stop motion', '2d', 'vector art', 'black and white'")
    shotAngle: Optional[str] = Field(None, description="Shot angle: 'aerial shot', 'eye_level shot', 'high angle shot', 'low angle shot', 'top-down shot'")
    shotSize: Optional[str] = Field(None, description="Shot size: 'close-up shot', 'extreme close-up', 'medium shot', 'long shot', 'extreme long shot'")


class FireflyVideoImagePlacement(BaseModel):
    """Timeline placement of keyframe image"""
    position: float = Field(..., description="Position on timeline: 0 = first frame, 1 = last frame", ge=0, le=1)


class FireflyVideoImageCondition(BaseModel):
    """Keyframe image for video generation"""
    placement: FireflyVideoImagePlacement = Field(..., description="Timeline placement details")
    source: FireflyPublicBinaryInput = Field(..., description="Image source details")


class GenerateVideoRequest(BaseModel):
    """Request for text-to-video generation"""
    prompt: Optional[str] = Field(None, description="Text prompt for video generation")
    bitRateFactor: Optional[int] = Field(18, description="Constant rate factor (0=lossless, 63=worst quality)", ge=0, le=63)
    image: Optional[FireflyInputImage] = Field(None, description="Keyframe image for video generation")
    conditions: Optional[List[FireflyVideoImageCondition]] = Field(None, description="Keyframe images for video generation")
    seeds: Optional[List[int]] = Field(None, description="Seed values (currently only 1 supported)")
    sizes: Optional[List[FireflyVideoSize]] = Field(None, description="Video dimensions")
    videoSettings: Optional[FireflyVideoSettings] = Field(None, description="Camera and shot control settings")


# ============================================================================
# Request Models - Upload Image
# ============================================================================

class UploadImageRequest(BaseModel):
    """Request for uploading an image"""
    name: str = Field(..., description="Filename")
    type: FireflyImageFormat = Field(..., description="MIME type of the image")


# ============================================================================
# Response Models - Async Operations
# ============================================================================

class AsyncAcceptResponse(BaseModel):
    """Initial response from async operations"""
    jobId: str = Field(..., description="Job ID for status polling")
    statusUrl: str = Field(..., description="URL to poll for status")
    cancelUrl: str = Field(..., description="URL to cancel the job")


class GenerateImagesResponse(BaseModel):
    """Response wrapper for generated images"""
    outputs: List[FireflyOutputImage] = Field(..., description="Output images")
    size: Optional[FireflySize] = Field(None, description="Output size")
    contentClass: Optional[FireflyContentClass] = Field(None, description="Content class")


# V5-specific response models for debug data
class FireflyV5LLMDescription(BaseModel):
    """LLM-generated description breakdown from V5 API"""
    scene: Optional[str] = Field(None, description="Scene description")
    type: Optional[str] = Field(None, description="Type (e.g., 'photo, natural, outdoor')")
    lighting: Optional[str] = Field(None, description="Lighting description")
    background: Optional[str] = Field(None, description="Background description")
    composition: Optional[str] = Field(None, description="Composition description")
    details: Optional[str] = Field(None, description="Detail description")
    camera: Optional[str] = Field(None, description="Camera settings")
    entity: Optional[str] = Field(None, description="Entity descriptions")
    version: Optional[str] = Field(None, description="Version")

    class Config:
        extra = "allow"  # Allow extra fields


class FireflyV5LLMResponse(BaseModel):
    """LLM response from V5 API debug data"""
    workflow: Optional[str] = Field(None, description="Workflow type")
    contentType: Optional[str] = Field(None, description="Content type (photo, art, etc.)")
    description: Optional[FireflyV5LLMDescription] = Field(None, description="Description breakdown")
    rendererPrompt: Optional[str] = Field(None, description="Expanded renderer prompt")

    class Config:
        extra = "allow"


class FireflyV5DebugData(BaseModel):
    """Debug data from V5 API response"""
    llm_response: Optional[FireflyV5LLMResponse] = Field(None, description="LLM response data")
    prompt_reasoner: Optional[str] = Field(None, description="Prompt reasoner used")
    is_user_text_input_nsfw: Optional[bool] = Field(None, description="NSFW flag")

    class Config:
        extra = "allow"


class GenerateImagesV5Response(BaseModel):
    """V5-specific response wrapper with debug data"""
    outputs: List[FireflyOutputImage] = Field(..., description="Output images")
    size: Optional[FireflySize] = Field(None, description="Output size")
    modelId: Optional[str] = Field(None, description="Model ID used")
    modelVersion: Optional[str] = Field(None, description="Model version used")
    debugData: Optional[FireflyV5DebugData] = Field(None, description="Debug data including LLM response")

    class Config:
        extra = "allow"


class AsyncTaskV5Response(BaseModel):
    """V5-specific response from status polling endpoint with debug data"""
    jobId: str = Field(..., description="Job ID")
    status: FireflyTaskStatus = Field(..., description="Current status")
    result: Optional[GenerateImagesV5Response] = Field(None, description="Result when succeeded")
    errorCode: Optional[str] = Field(None, description="Error code if failed")
    errorMessage: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        extra = "allow"

    @property
    def outputs(self) -> Optional[List[FireflyOutputImage]]:
        """Helper property to access outputs directly"""
        return self.result.outputs if self.result else None

    @property
    def renderer_prompt(self) -> Optional[str]:
        """Helper property to access renderer prompt"""
        if self.result and self.result.debugData and self.result.debugData.llm_response:
            return self.result.debugData.llm_response.rendererPrompt
        return None

    @property
    def content_type(self) -> Optional[str]:
        """Helper property to access content type (falls back to description.type for instruct-edit)"""
        if self.result and self.result.debugData and self.result.debugData.llm_response:
            # Try contentType first (available in text-to-image)
            if self.result.debugData.llm_response.contentType:
                return self.result.debugData.llm_response.contentType
            # Fall back to description.type (available in instruct-edit)
            if self.result.debugData.llm_response.description and self.result.debugData.llm_response.description.type:
                return self.result.debugData.llm_response.description.type
        return None

    @property
    def description(self) -> Optional[FireflyV5LLMDescription]:
        """Helper property to access description breakdown"""
        if self.result and self.result.debugData and self.result.debugData.llm_response:
            return self.result.debugData.llm_response.description
        return None


class GenerateVideoResponse(BaseModel):
    """Response wrapper for generated videos"""
    outputs: List[FireflyOutputVideo] = Field(..., description="Output videos")


class AsyncTaskResponse(BaseModel):
    """Response from status polling endpoint"""
    jobId: str = Field(..., description="Job ID")
    status: FireflyTaskStatus = Field(..., description="Current status")
    result: Optional[GenerateImagesResponse] = Field(None, description="Result when succeeded")
    errorCode: Optional[str] = Field(None, description="Error code if failed")
    errorMessage: Optional[str] = Field(None, description="Error message if failed")

    @property
    def outputs(self) -> Optional[List[FireflyOutputImage]]:
        """Helper property to access outputs directly"""
        return self.result.outputs if self.result else None


class AsyncVideoTaskResponse(BaseModel):
    """Response from video status polling endpoint"""
    jobId: str = Field(..., description="Job ID")
    status: FireflyTaskStatus = Field(..., description="Current status")
    result: Optional[GenerateVideoResponse] = Field(None, description="Result when succeeded")
    errorCode: Optional[str] = Field(None, description="Error code if failed")
    errorMessage: Optional[str] = Field(None, description="Error message if failed")

    @property
    def outputs(self) -> Optional[List[FireflyOutputVideo]]:
        """Helper property to access outputs directly"""
        return self.result.outputs if self.result else None


# ============================================================================
# Response Models - Upload Image
# ============================================================================

class UploadImageResponse(BaseModel):
    """Response from upload image request"""
    uploadId: str = Field(..., description="Upload ID for the image")
    uploadUrl: str = Field(..., description="Presigned URL to upload the image")


# ============================================================================
# Response Models - Custom Models
# ============================================================================

class HypermediaLink(BaseModel):
    """Standard representation of a hypermedia link"""
    href: Optional[str] = Field(None, description="Fully qualified URI or relative path for the link")
    rel: Optional[str] = Field(None, description="Relationship or function of the link (e.g. next, page, self)")
    templated: Optional[bool] = Field(None, description="Whether the href supports URI template parameters")


class CustomModelsLinks(BaseModel):
    """Collection of hypermedia links for pagination, navigation, etc."""
    page: Optional[HypermediaLink] = Field(None, description="Current page link")
    next: Optional[HypermediaLink] = Field(None, description="Next page link")


class CustomModelBaseModel(BaseModel):
    """Base model information"""
    name: Optional[str] = Field(None, description="Base model name")
    version: Optional[str] = Field(None, description="Base model version")


class CustomModel(BaseModel):
    """Custom model information"""
    version: Optional[str] = Field(None, description="Model version")
    assetName: Optional[str] = Field(None, description="Model name")
    size: Optional[int] = Field(None, description="Storage size used")
    etag: Optional[str] = Field(None, description="Version identifier")
    trainingMode: Optional[str] = Field(None, description="Training mode: 'subject' or 'style'")
    assetId: Optional[str] = Field(None, description="Unique identifier")
    mediaType: Optional[str] = Field(None, description="Media type specific to the asset")
    createdDate: Optional[str] = Field(None, description="Creation date (ISO 8601)")
    modifiedDate: Optional[str] = Field(None, description="Modification date (ISO 8601)")
    publishedState: Optional[str] = Field(None, description="Status: 'never', 'published', or 'unpublished'")
    baseModel: Optional[CustomModelBaseModel] = Field(None, description="Underlying GenAI model used for training")
    samplePrompt: Optional[str] = Field(None, description="Example string provided by trainer")
    displayName: Optional[str] = Field(None, description="User-provided asset name from training set")
    conceptId: Optional[str] = Field(None, description="Concept ID for subject mode prompts")


class CustomModelsResponse(BaseModel):
    """Response from list custom models"""
    custom_models: List[CustomModel] = Field(default_factory=list, description="List of custom models")
    links: Optional[CustomModelsLinks] = Field(None, description="Hypermedia links for pagination", alias="_links")
    total_count: Optional[int] = Field(None, description="Total number of models")


# ============================================================================
# V5 (Image5) Models
# ============================================================================

class FireflyV5AspectRatio(str, Enum):
    """Aspect ratio options for V5 image generation"""
    SQUARE = "1:1"
    LANDSCAPE_4_3 = "4:3"
    PORTRAIT_3_4 = "3:4"
    WIDESCREEN_16_9 = "16:9"
    PORTRAIT_9_16 = "9:16"


class FireflyV5ReferenceBlobUsage(str, Enum):
    """Usage type for reference blobs in V5"""
    GENERAL = "general"


class FireflyV5ReferenceBlob(BaseModel):
    """Reference blob for V5 image generation"""
    source: FireflyPublicBinaryInput = Field(..., description="Source location of the reference image")
    usage: Optional[FireflyV5ReferenceBlobUsage] = Field(
        FireflyV5ReferenceBlobUsage.GENERAL,
        description="Usage of the reference blob"
    )


class FireflyV5ModelSpecificPayload(BaseModel):
    """Model-specific parameters for V5"""
    localeCode: Optional[str] = Field(
        None,
        description="Locale code (RFC 5646 format, e.g., 'en-US') for region-relevant content"
    )


class GenerateImagesV5Request(BaseModel):
    """Request for V5 (Image5) text-to-image generation"""
    prompt: str = Field(..., description="Text prompt for generation", min_length=1, max_length=1500)
    aspectRatio: Optional[FireflyV5AspectRatio] = Field(None, description="Aspect ratio (mutually exclusive with size)")
    size: Optional[FireflySize] = Field(None, description="Custom size (mutually exclusive with aspectRatio)")
    # Note: modelId and modelVersion are set via x-model-version header, not in body
    numVariations: Optional[int] = Field(1, description="Number of variations", ge=1, le=4)
    seeds: Optional[List[int]] = Field(None, description="Seeds for reproducibility (1-4 items)")
    referenceBlobs: Optional[List[FireflyV5ReferenceBlob]] = Field(
        default_factory=list,
        description="Reference blobs for additional input"
    )
    modelSpecificPayload: Optional[FireflyV5ModelSpecificPayload] = Field(
        None,
        description="Additional model-specific parameters"
    )


class FireflyV5HypermediaLink(BaseModel):
    """Hypermedia link in V5 responses"""
    href: str = Field(..., description="URL for the link")


class FireflyV5Links(BaseModel):
    """Links in V5 async response"""
    cancel: Optional[FireflyV5HypermediaLink] = Field(None, description="URL to cancel the job")
    result: Optional[FireflyV5HypermediaLink] = Field(None, description="URL to poll for results")


class AsyncAcceptResponseV5(BaseModel):
    """Initial response from V5 async operations"""
    links: FireflyV5Links = Field(..., description="Hypermedia links for job control")
    progress: Optional[int] = Field(0, description="Generation progress")
