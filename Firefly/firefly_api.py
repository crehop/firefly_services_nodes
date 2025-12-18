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
    width: int = Field(..., description="Width of the image in pixels", ge=1, le=2688)
    height: int = Field(..., description="Height of the image in pixels", ge=1, le=2688)


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
