"""
Adobe Photoshop API models and types.

This module contains Pydantic models for interacting with Adobe Photoshop API.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, List, Union
from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class PhotoshopRemoveBgMode(str, Enum):
    """Mode for background removal"""
    CUTOUT = "cutout"
    MASK = "mask"
    PSD = "psd"


class PhotoshopOutputMediaType(str, Enum):
    """Media type for output images"""
    IMAGE_JPEG = "image/jpeg"
    IMAGE_PNG = "image/png"
    IMAGE_WEBP = "image/webp"
    IMAGE_PSD = "image/vnd.adobe.photoshop"


class PhotoshopStorageType(str, Enum):
    """Storage type for input/output images"""
    EXTERNAL = "external"
    AZURE = "azure"
    DROPBOX = "dropbox"


class PhotoshopJobStatusEnum(str, Enum):
    """Status of Photoshop API jobs"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# ============================================================================
# Request Models
# ============================================================================

class PhotoshopImageSource(BaseModel):
    """Image source with pre-signed URL"""
    url: str = Field(..., description="Pre-signed URL to the image")
    storage: Optional[PhotoshopStorageType] = Field(None, description="Storage type (external for S3/HTTP URLs, optional)")


class PhotoshopImageInput(BaseModel):
    """Input image specification"""
    source: PhotoshopImageSource = Field(..., description="Image source")


class PhotoshopBackgroundColor(BaseModel):
    """Background color specification"""
    red: int = Field(..., description="Red value (0-255)", ge=0, le=255)
    green: int = Field(..., description="Green value (0-255)", ge=0, le=255)
    blue: int = Field(..., description="Blue value (0-255)", ge=0, le=255)
    alpha: float = Field(..., description="Alpha value (0-1)", ge=0, le=1)


class PhotoshopOutputOptions(BaseModel):
    """Output options for processed image"""
    mediaType: Optional[PhotoshopOutputMediaType] = Field(None, description="Output media type")
    trim: Optional[bool] = Field(False, description="Trim transparent pixels")


class RemoveBackgroundRequest(BaseModel):
    """Request for remove background operation"""
    image: PhotoshopImageInput = Field(..., description="Input image")
    mode: Optional[PhotoshopRemoveBgMode] = Field(PhotoshopRemoveBgMode.CUTOUT, description="Removal mode")
    output: Optional[PhotoshopOutputOptions] = Field(None, description="Output options")
    backgroundColor: Optional[PhotoshopBackgroundColor] = Field(None, description="Background color")
    colorDecontamination: Optional[float] = Field(None, description="Color decontamination strength (0-1)", ge=0, le=1)


class RefineMaskRequest(BaseModel):
    """Request for refine mask operation"""
    image: PhotoshopImageInput = Field(..., description="Input image")
    mask: PhotoshopImageInput = Field(..., description="Mask to refine")
    colorDecontamination: Optional[bool] = Field(False, description="Return RGBA image with refined mask applied (true) or just refined mask (false)")


class MaskBodyPartsRequest(BaseModel):
    """Request for mask-body-parts operation"""
    image: PhotoshopImageInput = Field(..., description="Input image")
    mask: PhotoshopImageInput = Field(..., description="Input mask")


class FillMaskedAreasRequest(BaseModel):
    """Request for fill-masked-areas operation"""
    image: PhotoshopImageInput = Field(..., description="Input image")
    masks: List[PhotoshopImageInput] = Field(..., description="List of masks to fill")


# ============================================================================
# Response Models
# ============================================================================

class RemoveBackgroundResponse(BaseModel):
    """Initial response from remove background submission"""
    jobId: str = Field(..., description="Job ID for polling")
    statusUrl: str = Field(..., description="URL to check job status")
    status: Optional[PhotoshopJobStatusEnum] = Field(None, description="Initial status (if provided)")


class PhotoshopDestination(BaseModel):
    """Destination information for output"""
    url: str = Field(..., description="Pre-signed URL to download result")
    mediaType: Optional[str] = Field(None, description="Media type of output")


class PhotoshopOutput(BaseModel):
    """Output specification"""
    destination: PhotoshopDestination = Field(..., description="Output destination")


class PhotoshopJobResult(BaseModel):
    """Result of completed job"""
    outputs: List[PhotoshopOutput] = Field(..., description="List of output images")


class PhotoshopJobStatus(BaseModel):
    """Status response from polling"""
    status: PhotoshopJobStatusEnum = Field(..., description="Current job status")
    jobId: str = Field(..., description="Job ID")
    result: Optional[PhotoshopJobResult] = Field(None, description="Job result (when succeeded)")


# ============================================================================
# Refine Mask Response Models (v1 API)
# ============================================================================

class RefineMaskBoundingBox(BaseModel):
    """Bounding box for mask"""
    x: int = Field(..., description="X coordinate")
    y: int = Field(..., description="Y coordinate")
    width: int = Field(..., description="Width")
    height: int = Field(..., description="Height")


class RefineMaskDestination(BaseModel):
    """Destination for refined mask"""
    url: str = Field(..., description="URL to download refined mask")


class RefineMaskImageResult(BaseModel):
    """Image result from refine-mask"""
    boundingBox: RefineMaskBoundingBox = Field(..., description="Bounding box")
    mediaType: str = Field(..., description="Media type (e.g., image/png)")
    destination: RefineMaskDestination = Field(..., description="Download destination")


class RefineMaskStatusResponse(BaseModel):
    """Status response from refine-mask polling (v1 API) - ACTUAL structure"""
    jobId: str = Field(..., description="Job ID")
    status: PhotoshopJobStatusEnum = Field(..., description="Current job status")
    image: Optional[RefineMaskImageResult] = Field(None, description="Refined image result (when colorDecontamination=true)")
    mask: Optional[RefineMaskImageResult] = Field(None, description="Refined mask result (when colorDecontamination=false)")
    Retry_After: Optional[str] = Field(None, alias="Retry-After", description="Retry after seconds")


# ============================================================================
# Mask Objects Models (v1 API)
# ============================================================================

class MaskObjectsRequest(BaseModel):
    """Request for mask-objects operation"""
    image: PhotoshopImageInput = Field(..., description="Input image")


class MaskObjectsBoundingBox(BaseModel):
    """Bounding box for detected object/background mask (normalized coordinates 0-1)"""
    x: float = Field(..., description="X coordinate (normalized 0-1)")
    y: float = Field(..., description="Y coordinate (normalized 0-1)")
    width: float = Field(..., description="Width (normalized 0-1)")
    height: float = Field(..., description="Height (normalized 0-1)")


class MaskObjectsDestination(BaseModel):
    """Destination for mask download"""
    url: str = Field(..., description="URL to download mask")


class MaskObjectsMaskItem(BaseModel):
    """Individual mask item (semantic or background)"""
    mediaType: str = Field(..., description="Media type (e.g., image/png)")
    destination: MaskObjectsDestination = Field(..., description="Download destination")
    label: str = Field(..., description="Detected object/background label")
    boundingBox: MaskObjectsBoundingBox = Field(..., description="Bounding box")
    score: float = Field(..., description="Confidence score (0-1)")


class MaskObjectsStatusResponse(BaseModel):
    """Status response from mask-objects polling (v1 API)"""
    jobId: str = Field(..., description="Job ID")
    status: PhotoshopJobStatusEnum = Field(..., description="Current job status")
    semanticMasks: Optional[List[MaskObjectsMaskItem]] = Field(None, description="Detected object masks")
    backgroundMasks: Optional[List[MaskObjectsMaskItem]] = Field(None, description="Detected background masks")
    Retry_After: Optional[str] = Field(None, alias="Retry-After", description="Retry after seconds")


class MaskBodyPartsStatusResponse(BaseModel):
    """Status response from mask-body-parts polling (v1 API)"""
    jobId: str = Field(..., description="Job ID")
    status: PhotoshopJobStatusEnum = Field(..., description="Current job status")
    masks: Optional[List[MaskObjectsMaskItem]] = Field(None, description="All detected masks (body parts and background)")
    Retry_After: Optional[str] = Field(None, alias="Retry-After", description="Retry after seconds")


class FillMaskedAreasImageResult(BaseModel):
    """Image result from fill-masked-areas (simpler than refine-mask)"""
    mediaType: str = Field(..., description="Media type (e.g., image/png)")
    destination: RefineMaskDestination = Field(..., description="Download destination")


class FillMaskedAreasStatusResponse(BaseModel):
    """Status response from fill-masked-areas polling (v1 API)"""
    jobId: str = Field(..., description="Job ID")
    status: PhotoshopJobStatusEnum = Field(..., description="Current job status")
    image: Optional[FillMaskedAreasImageResult] = Field(None, description="Filled image result")
    Retry_After: Optional[str] = Field(None, alias="Retry-After", description="Retry after seconds")


# ============================================================================
# Photoshop Actions Models
# ============================================================================

class PhotoshopActionsInput(BaseModel):
    """Input specification for Photoshop Actions"""
    href: str = Field(..., description="URL to input file")
    storage: str = Field(default="external", description="Storage type (external, azure, dropbox)")


class PhotoshopActionsOutput(BaseModel):
    """Output specification for Photoshop Actions"""
    href: str = Field(..., description="URL for output file")
    storage: str = Field(default="external", description="Storage type")
    type: str = Field(default="image/vnd.adobe.photoshop", description="Output MIME type")
    overwrite: Optional[bool] = Field(True, description="Overwrite existing file")
    quality: Optional[int] = Field(None, description="Quality level (1-12 for JPEG)", ge=1, le=12)
    compression: Optional[str] = Field(None, description="Compression level for PNG (small, medium, large)")


class PhotoshopAction(BaseModel):
    """Action file specification"""
    storage: str = Field(default="external", description="Storage type")
    href: str = Field(..., description="URL to action file (.atn)")
    actionName: str = Field(..., description="Name of the action to execute")


class PhotoshopAsset(BaseModel):
    """Asset specification (patterns, fonts, brushes)"""
    href: str = Field(..., description="URL to asset file")
    storage: str = Field(default="external", description="Storage type")


class PhotoshopActionsOptions(BaseModel):
    """Options for Photoshop Actions execution"""
    actions: List[PhotoshopAction] = Field(..., description="List of actions to execute")
    patterns: Optional[List[PhotoshopAsset]] = Field(None, description="Pattern files (.pat)")
    fonts: Optional[List[PhotoshopAsset]] = Field(None, description="Font files")
    brushes: Optional[List[PhotoshopAsset]] = Field(None, description="Brush files (.abr)")


class PhotoshopActionsRequest(BaseModel):
    """Request for Photoshop Actions execution"""
    inputs: List[PhotoshopActionsInput] = Field(..., description="Input files")
    outputs: List[PhotoshopActionsOutput] = Field(..., description="Output specifications")
    options: PhotoshopActionsOptions = Field(..., description="Execution options")


class PhotoshopActionsResponse(BaseModel):
    """Response from Photoshop Actions submission"""
    jobId: str = Field(..., description="Job ID for polling")
    statusUrl: str = Field(..., description="URL to check job status")
    status: Optional[PhotoshopJobStatusEnum] = Field(None, description="Initial status")


class PhotoshopActionsOutputStatus(BaseModel):
    """Status of a single output in the outputs array for Actions API"""
    input: str = Field(..., description="Input URL that was processed")
    status: PhotoshopJobStatusEnum = Field(..., description="Status of this output")
    created: str = Field(..., description="Creation timestamp")
    modified: str = Field(..., description="Last modification timestamp")
    links: Optional[dict] = Field(None, alias="_links", description="Links including renditions")

    class Config:
        populate_by_name = True

    @property
    def destination(self):
        """Extract destination URL from renditions"""
        if self.links and "renditions" in self.links:
            renditions = self.links["renditions"]
            if isinstance(renditions, list) and len(renditions) > 0:
                # Return an object with a 'url' property for compatibility
                class Destination:
                    def __init__(self, href):
                        self.url = href
                return Destination(renditions[0]["href"])
        return None


class PhotoshopActionsJobStatus(BaseModel):
    """Status response from Photoshop Actions polling"""
    jobId: str = Field(..., description="Job ID")
    outputs: List[PhotoshopActionsOutputStatus] = Field(..., description="List of outputs with their statuses")
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def status(self) -> PhotoshopJobStatusEnum:
        """Get overall status from first output (for compatibility with polling)"""
        if self.outputs and len(self.outputs) > 0:
            return self.outputs[0].status
        return PhotoshopJobStatusEnum.PENDING

    @property
    def result(self):
        """Build a result object for compatibility with existing code"""
        if not self.outputs or len(self.outputs) == 0:
            return None

        # Check if any output has succeeded
        succeeded_outputs = [o for o in self.outputs if o.status == PhotoshopJobStatusEnum.SUCCEEDED]
        if not succeeded_outputs:
            return None

        # Build PhotoshopJobResult-like object
        class JobResult:
            def __init__(self, outputs):
                self.outputs = outputs

        return JobResult(succeeded_outputs)


# ============================================================================
# Photoshop Depth Blur Models
# ============================================================================

class FocalSelector(BaseModel):
    """Focal selector coordinates"""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")


class DepthBlurOptions(BaseModel):
    """Options for Depth Blur operation"""
    focalDistance: Optional[int] = Field(None, description="Distance of the point to be in focus (0=nearest, 100=furthest)", ge=0, le=100)
    focalRange: Optional[int] = Field(None, description="Range of the focal point", ge=0, le=100)
    focalSelector: Optional[FocalSelector] = Field(None, description="Focal selector coordinates")
    focusSubject: Optional[bool] = Field(None, description="Use select subject to automatically select prominent subject for focus")
    blurStrength: Optional[int] = Field(None, description="Amount of blur to apply", ge=0, le=100)
    haze: Optional[int] = Field(None, description="Amount of haze to apply", ge=0, le=100)
    temp: Optional[int] = Field(None, description="Temperature to apply (-50=coldest, 50=warmest)", ge=-50, le=50)
    tint: Optional[int] = Field(None, description="Amount of tint to apply", ge=-50, le=50)
    saturation: Optional[int] = Field(None, description="Amount of saturation to apply", ge=-50, le=50)
    brightness: Optional[int] = Field(None, description="Amount of brightness to apply", ge=-50, le=50)
    grain: Optional[int] = Field(None, description="Amount of graining to add", ge=0, le=100)


class DepthBlurRequest(BaseModel):
    """Request for Depth Blur operation"""
    inputs: List[PhotoshopActionsInput] = Field(..., description="Input files (only one supported)")
    outputs: List[PhotoshopActionsOutput] = Field(..., description="Output specifications")
    options: Optional[DepthBlurOptions] = Field(None, description="Depth blur options")


class DepthBlurResponse(BaseModel):
    """Response from Depth Blur submission"""
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def jobId(self) -> Optional[str]:
        """Extract job ID from _links if available"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            href = self.links['self']['href']
            parts = href.split('/')
            if len(parts) > 0:
                return parts[-1]
        return None

    @property
    def statusUrl(self) -> Optional[str]:
        """Extract status URL from _links"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            return self.links['self']['href']
        return None


class DepthBlurOutputStatus(BaseModel):
    """Status of a single output in the outputs array for Depth Blur API"""
    input: str = Field(..., description="Input URL that was processed")
    status: PhotoshopJobStatusEnum = Field(..., description="Status of this output")
    created: str = Field(..., description="Creation timestamp")
    modified: str = Field(..., description="Last modification timestamp")
    links: Optional[dict] = Field(None, alias="_links", description="Links including renditions")

    class Config:
        populate_by_name = True

    @property
    def destination(self):
        """Extract destination URL from renditions"""
        if self.links and "renditions" in self.links:
            renditions = self.links["renditions"]
            if isinstance(renditions, list) and len(renditions) > 0:
                class Destination:
                    def __init__(self, href):
                        self.url = href
                return Destination(renditions[0]["href"])
        return None


class DepthBlurJobStatus(BaseModel):
    """Status response from Depth Blur polling"""
    jobId: str = Field(..., description="Job ID")
    outputs: List[DepthBlurOutputStatus] = Field(..., description="List of outputs with their statuses")
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def status(self) -> PhotoshopJobStatusEnum:
        """Get overall status from first output (for compatibility with polling)"""
        if self.outputs and len(self.outputs) > 0:
            return self.outputs[0].status
        return PhotoshopJobStatusEnum.PENDING

    @property
    def result(self):
        """Build a result object for compatibility with existing code"""
        if not self.outputs or len(self.outputs) == 0:
            return None

        succeeded_outputs = [o for o in self.outputs if o.status == PhotoshopJobStatusEnum.SUCCEEDED]
        if not succeeded_outputs:
            return None

        class JobResult:
            def __init__(self, outputs):
                self.outputs = outputs

        return JobResult(succeeded_outputs)


# ============================================================================
# Photoshop Product Crop Models
# ============================================================================

class ProductCropOptions(BaseModel):
    """Options for Product Crop operation"""
    unit: str = Field(default="Pixels", description="Unit for width/height (Pixels, Percent)")
    width: int = Field(..., description="Width to be added as padding", ge=0)
    height: int = Field(..., description="Height to be added as padding", ge=0)


class ProductCropRequest(BaseModel):
    """Request for Product Crop operation"""
    inputs: List[PhotoshopActionsInput] = Field(..., description="Input files (only one supported)")
    outputs: List[PhotoshopActionsOutput] = Field(..., description="Output specifications")
    options: ProductCropOptions = Field(..., description="Product crop options")


class ProductCropResponse(BaseModel):
    """Response from Product Crop submission"""
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def jobId(self) -> Optional[str]:
        """Extract job ID from _links if available"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            # Extract job ID from status URL
            href = self.links['self']['href']
            # Example: https://image.adobe.io/pie/psdService/status/<jobId>
            parts = href.split('/')
            if len(parts) > 0:
                return parts[-1]
        return None

    @property
    def statusUrl(self) -> Optional[str]:
        """Extract status URL from _links"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            return self.links['self']['href']
        return None


class ProductCropOutputStatus(BaseModel):
    """Status of a single output in the outputs array for Product Crop API"""
    input: str = Field(..., description="Input URL that was processed")
    status: PhotoshopJobStatusEnum = Field(..., description="Status of this output")
    created: str = Field(..., description="Creation timestamp")
    modified: str = Field(..., description="Last modification timestamp")
    links: Optional[dict] = Field(None, alias="_links", description="Links including renditions")

    class Config:
        populate_by_name = True

    @property
    def destination(self):
        """Extract destination URL from renditions"""
        if self.links and "renditions" in self.links:
            renditions = self.links["renditions"]
            if isinstance(renditions, list) and len(renditions) > 0:
                # Return an object with a 'url' property for compatibility
                class Destination:
                    def __init__(self, href):
                        self.url = href
                return Destination(renditions[0]["href"])
        return None


class ProductCropJobStatus(BaseModel):
    """Status response from Product Crop polling"""
    jobId: str = Field(..., description="Job ID")
    outputs: List[ProductCropOutputStatus] = Field(..., description="List of outputs with their statuses")
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def status(self) -> PhotoshopJobStatusEnum:
        """Get overall status from first output (for compatibility with polling)"""
        if self.outputs and len(self.outputs) > 0:
            return self.outputs[0].status
        return PhotoshopJobStatusEnum.PENDING

    @property
    def result(self):
        """Build a result object for compatibility with existing code"""
        if not self.outputs or len(self.outputs) == 0:
            return None

        # Check if any output has succeeded
        succeeded_outputs = [o for o in self.outputs if o.status == PhotoshopJobStatusEnum.SUCCEEDED]
        if not succeeded_outputs:
            return None

        # Build PhotoshopJobResult-like object
        class JobResult:
            def __init__(self, outputs):
                self.outputs = outputs

        return JobResult(succeeded_outputs)


# ============================================================================
# Photoshop ActionJSON Models
# ============================================================================

class ActionJsonOptions(BaseModel):
    """Options for ActionJSON execution"""
    actionJSON: List[dict] = Field(..., description="Array of Photoshop JSON-formatted actions to play")
    patterns: Optional[List[PhotoshopAsset]] = Field(None, description="Custom pattern presets (.pat)")
    fonts: Optional[List[PhotoshopAsset]] = Field(None, description="Custom fonts")
    brushes: Optional[List[PhotoshopAsset]] = Field(None, description="Custom brushes (.abr)")
    additionalImages: Optional[List[PhotoshopAsset]] = Field(None, description="Additional images for actionJSON commands")


class ActionJsonRequest(BaseModel):
    """Request for ActionJSON execution"""
    inputs: List[PhotoshopActionsInput] = Field(..., description="Input files (only one supported)")
    outputs: List[PhotoshopActionsOutput] = Field(..., description="Output specifications")
    options: ActionJsonOptions = Field(..., description="ActionJSON execution options")


class ActionJsonResponse(BaseModel):
    """Response from ActionJSON submission"""
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def jobId(self) -> Optional[str]:
        """Extract job ID from _links if available"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            # Extract job ID from status URL
            href = self.links['self']['href']
            # Example: https://image.adobe.io/pie/psdService/status/<jobId>
            parts = href.split('/')
            if len(parts) > 0:
                return parts[-1]
        return None

    @property
    def statusUrl(self) -> Optional[str]:
        """Extract status URL from _links"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            return self.links['self']['href']
        return None


class ActionJsonRendition(BaseModel):
    """Rendition link in output response"""
    href: str = Field(..., description="URL to download the result")
    storage: str = Field(..., description="Storage type")
    type: str = Field(..., description="Output MIME type")
    compression: Optional[str] = Field(None, description="Compression level (for PNG)")
    quality: Optional[int] = Field(None, description="Quality level (for JPEG)")


class ActionJsonOutputStatus(BaseModel):
    """Status of a single output in the outputs array"""
    input: str = Field(..., description="Input URL that was processed")
    status: PhotoshopJobStatusEnum = Field(..., description="Status of this output")
    created: str = Field(..., description="Creation timestamp")
    modified: str = Field(..., description="Last modification timestamp")
    links: Optional[dict] = Field(None, alias="_links", description="Links including renditions")

    class Config:
        populate_by_name = True

    @property
    def destination(self):
        """Extract destination URL from renditions"""
        if self.links and "renditions" in self.links:
            renditions = self.links["renditions"]
            if isinstance(renditions, list) and len(renditions) > 0:
                # Return an object with a 'url' property for compatibility
                class Destination:
                    def __init__(self, href):
                        self.url = href
                return Destination(renditions[0]["href"])
        return None


class ActionJsonJobStatus(BaseModel):
    """Status response from ActionJSON polling"""
    jobId: str = Field(..., description="Job ID")
    outputs: List[ActionJsonOutputStatus] = Field(..., description="List of outputs with their statuses")
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def status(self) -> PhotoshopJobStatusEnum:
        """Get overall status from first output (for compatibility with polling)"""
        if self.outputs and len(self.outputs) > 0:
            return self.outputs[0].status
        return PhotoshopJobStatusEnum.PENDING

    @property
    def result(self):
        """Build a result object for compatibility with existing code"""
        if not self.outputs or len(self.outputs) == 0:
            return None

        # Check if any output has succeeded
        succeeded_outputs = [o for o in self.outputs if o.status == PhotoshopJobStatusEnum.SUCCEEDED]
        if not succeeded_outputs:
            return None

        # Build PhotoshopJobResult-like object
        class JobResult:
            def __init__(self, outputs):
                self.outputs = outputs

        return JobResult(succeeded_outputs)


# ============================================================================
# Photoshop Document Create Models
# ============================================================================

class DocumentSettings(BaseModel):
    """Document settings for PSD creation"""
    height: int = Field(..., description="Document height in pixels")
    width: int = Field(..., description="Document width in pixels")
    resolution: Optional[int] = Field(72, description="Document resolution (DPI)", ge=72, le=300)
    fill: Optional[str] = Field("transparent", description="Fill color (white, backgroundColor, transparent)")
    mode: Optional[str] = Field("rgb", description="Color mode (bitmap, greyscale, indexed, rgb, cmyk, hsl, hsb, multichannel, duotone, lab, xyz)")
    depth: Optional[int] = Field(8, description="Bit depth (8, 16, 32)")


class DocumentLayerBounds(BaseModel):
    """Bounds for layer positioning"""
    top: int = Field(..., description="Top position in pixels")
    left: int = Field(..., description="Left position in pixels")
    width: int = Field(..., description="Width in pixels")
    height: int = Field(..., description="Height in pixels")


class BrightnessContrast(BaseModel):
    """Brightness and contrast adjustment"""
    brightness: Optional[int] = Field(None, description="Brightness (-150 to 150)", ge=-150, le=150)
    contrast: Optional[int] = Field(None, description="Contrast (-150 to 150)", ge=-150, le=150)


class ExposureDetails(BaseModel):
    """Exposure adjustment"""
    exposure: Optional[float] = Field(0, description="Exposure (-20 to 20)", ge=-20, le=20)
    offset: Optional[float] = Field(0, description="Offset (-0.5 to 0.5)", ge=-0.5, le=0.5)
    gammaCorrection: Optional[float] = Field(1, description="Gamma correction (0.01 to 9.99)", ge=0.01, le=9.99)


class ChannelDetails(BaseModel):
    """Channel details for hue/saturation"""
    channel: str = Field(default="master", description="Channel type (master)")
    hue: Optional[int] = Field(None, description="Hue (-180 to 180)", ge=-180, le=180)
    saturation: Optional[int] = Field(None, description="Saturation (-100 to 100)", ge=-100, le=100)
    lightness: Optional[int] = Field(None, description="Lightness (-100 to 100)", ge=-100, le=100)


class HueSaturation(BaseModel):
    """Hue and saturation adjustment"""
    colorize: Optional[bool] = Field(None, description="Whether to colorize")
    channels: Optional[List[ChannelDetails]] = Field(None, description="Channel adjustments")


class ColorBalance(BaseModel):
    """Color balance adjustment"""
    preserveLuminosity: Optional[bool] = Field(None, description="Whether to preserve luminosity")
    shadowLevels: Optional[List[int]] = Field(None, description="Shadow levels")
    midtoneLevels: Optional[List[int]] = Field(None, description="Midtone levels")
    highlightLevels: Optional[List[int]] = Field(None, description="Highlight levels")


class AdjustmentDetails(BaseModel):
    """Adjustment layer information"""
    brightnessContrast: Optional[BrightnessContrast] = Field(None, description="Brightness/contrast adjustment")
    exposure: Optional[ExposureDetails] = Field(None, description="Exposure adjustment")
    hueSaturation: Optional[HueSaturation] = Field(None, description="Hue/saturation adjustment")
    colorBalance: Optional[ColorBalance] = Field(None, description="Color balance adjustment")


class BlendDetails(BaseModel):
    """Blend options for layers"""
    opacity: Optional[int] = Field(None, description="Opacity (0-100)", ge=0, le=100)
    blendMode: Optional[str] = Field(None, description="Blend mode (normal, multiply, screen, overlay, etc.)")


class Offset(BaseModel):
    """Offset for mask positioning"""
    x: Optional[int] = Field(None, description="Horizontal offset")
    y: Optional[int] = Field(None, description="Vertical offset")


class MaskDetails(BaseModel):
    """Mask details for layers"""
    input: Optional[PhotoshopActionsInput] = Field(None, description="Mask input image")
    clip: Optional[bool] = Field(None, description="Indicates if this is a clipped layer")
    enabled: Optional[bool] = Field(None, description="Indicates whether a mask is enabled")
    linked: Optional[bool] = Field(None, description="Indicates whether a mask is linked to the layer")
    offset: Optional[Offset] = Field(None, description="Mask offset")


class RgbColor(BaseModel):
    """RGB color specification"""
    red: int = Field(..., description="Red (0-255)", ge=0, le=255)
    green: int = Field(..., description="Green (0-255)", ge=0, le=255)
    blue: int = Field(..., description="Blue (0-255)", ge=0, le=255)


class SolidColor(BaseModel):
    """Solid color fill"""
    rgb: RgbColor = Field(..., description="RGB color")


class FillDetails(BaseModel):
    """Fill layer details"""
    solidColor: Optional[SolidColor] = Field(None, description="Solid color fill")


class TextLayerCharacterStyleDetails(BaseModel):
    """Character style details for text layers"""
    from_: Optional[int] = Field(None, alias="from", description="Starting character index")
    to: Optional[int] = Field(None, description="Ending character index")
    fontSize: Optional[float] = Field(None, description="Font size")
    fontName: Optional[str] = Field(None, description="Font PostScript name")
    color: Optional[RgbColor] = Field(None, description="Text color")

    class Config:
        populate_by_name = True


class TextLayerParagraphStyleDetails(BaseModel):
    """Paragraph style details for text layers"""
    from_: Optional[int] = Field(None, alias="from", description="Starting character index")
    to: Optional[int] = Field(None, description="Ending character index")
    alignment: Optional[str] = Field(None, description="Text alignment (left, center, right)")

    class Config:
        populate_by_name = True


class TextLayerDetails(BaseModel):
    """Text layer details"""
    content: Optional[str] = Field(None, description="Text content")
    characterStyles: Optional[List[TextLayerCharacterStyleDetails]] = Field(None, description="Character styles")
    paragraphStyles: Optional[List[TextLayerParagraphStyleDetails]] = Field(None, description="Paragraph styles")


class SmartObjectDetails(BaseModel):
    """Smart object details"""
    type: Optional[str] = Field(None, description="Desired image format")
    linked: Optional[bool] = Field(False, description="Indicates if the smart object is linked")
    name: Optional[str] = Field(None, description="Name of the smart object")
    path: Optional[str] = Field(None, description="Path for linked smart object")
    instanceId: Optional[str] = Field(None, description="Instance ID of embedded smart object")


class LayerMaskDetails(BaseModel):
    """Layer mask details"""
    clip: Optional[bool] = Field(None, description="Indicates if this is a clipped layer")
    enabled: Optional[bool] = Field(None, description="Indicates whether a mask is enabled")
    linked: Optional[bool] = Field(None, description="Indicates whether a mask is linked to the layer")
    offset: Optional[Offset] = Field(None, description="Mask offset")


# Forward reference for children layers
class ChildrenLayerDetails(BaseModel):
    """Child layer details for layer groups"""
    id: Optional[int] = Field(None, description="Layer ID")
    index: Optional[int] = Field(None, description="Layer index")
    thumbnail: Optional[str] = Field(None, description="Thumbnail URL if requested")
    type: Optional[str] = Field(None, description="Layer type")
    name: Optional[str] = Field(None, description="Layer name")
    locked: Optional[bool] = Field(None, description="Whether layer is locked")
    visible: Optional[bool] = Field(None, description="Whether layer is visible")
    adjustments: Optional[AdjustmentDetails] = Field(None, description="Adjustment layer information")
    bounds: Optional[DocumentLayerBounds] = Field(None, description="Layer bounds")
    blendOptions: Optional[BlendDetails] = Field(None, description="Blend options")
    mask: Optional[LayerMaskDetails] = Field(None, description="Mask details")
    smartObject: Optional[SmartObjectDetails] = Field(None, description="Smart object details")
    fill: Optional[FillDetails] = Field(None, description="Fill details")
    text: Optional[TextLayerDetails] = Field(None, description="Text layer details")


class DocumentLayerInput(BaseModel):
    """Layer specification for document creation"""
    type: str = Field(default="layer", description="Layer type (layer, textLayer, adjustmentLayer, smartObject, fillLayer, backgroundLayer, layerSection)")
    input: Optional[PhotoshopActionsInput] = Field(None, description="Input image for this layer")
    name: Optional[str] = Field(None, description="Layer name")
    locked: Optional[bool] = Field(False, description="Whether layer is locked")
    visible: Optional[bool] = Field(True, description="Whether layer is visible")
    bounds: Optional[DocumentLayerBounds] = Field(None, description="Layer bounds (position and size)")
    adjustments: Optional[AdjustmentDetails] = Field(None, description="Adjustment layer settings")
    children: Optional[List[ChildrenLayerDetails]] = Field(None, description="Child layers for layer groups")
    blendOptions: Optional[BlendDetails] = Field(None, description="Blend options")
    mask: Optional[MaskDetails] = Field(None, description="Mask details")
    smartObject: Optional[SmartObjectDetails] = Field(None, description="Smart object details")
    fill: Optional[FillDetails] = Field(None, description="Fill layer details")
    text: Optional[TextLayerDetails] = Field(None, description="Text layer details")


class DocumentCreateOptions(BaseModel):
    """Options for document creation"""
    manageMissingFonts: Optional[str] = Field(None, description="Action for missing fonts (useDefault, fail)")
    globalFont: Optional[str] = Field(None, description="PostScript name of global default font")
    fonts: Optional[List[PhotoshopAsset]] = Field(None, description="Custom fonts needed in this document")
    document: DocumentSettings = Field(..., description="Document settings")
    layers: Optional[List[DocumentLayerInput]] = Field(None, description="Layers to add to document")


class DocumentCreateOutput(BaseModel):
    """Output specification for document creation"""
    href: str = Field(..., description="Pre-signed URL for output")
    storage: str = Field(default="external", description="Storage type")
    type: str = Field(..., description="Output MIME type")
    overwrite: Optional[bool] = Field(True, description="Overwrite existing file")
    quality: Optional[int] = Field(None, description="Quality for JPEG output (1-12)", ge=1, le=12)
    compression: Optional[str] = Field(None, description="Compression for PNG (small, medium, large)")


class DocumentCreateRequest(BaseModel):
    """Request for document creation"""
    options: DocumentCreateOptions = Field(..., description="Document creation options")
    outputs: List[DocumentCreateOutput] = Field(..., description="Output specifications")


class DocumentCreateResponse(BaseModel):
    """Response from Document Create submission"""
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def jobId(self) -> Optional[str]:
        """Extract job ID from _links if available"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            href = self.links['self']['href']
            parts = href.split('/')
            if len(parts) > 0:
                return parts[-1]
        return None

    @property
    def statusUrl(self) -> Optional[str]:
        """Extract status URL from _links"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            return self.links['self']['href']
        return None


class DocumentCreateOutputStatus(BaseModel):
    """Status of a single output in the outputs array for Document Create API"""
    status: PhotoshopJobStatusEnum = Field(..., description="Status of this output")
    created: str = Field(..., description="Creation timestamp")
    modified: str = Field(..., description="Last modification timestamp")
    links: Optional[dict] = Field(None, alias="_links", description="Links including renditions")

    class Config:
        populate_by_name = True

    @property
    def destination(self):
        """Extract destination URL from renditions"""
        if self.links and "renditions" in self.links:
            renditions = self.links["renditions"]
            if isinstance(renditions, list) and len(renditions) > 0:
                class Destination:
                    def __init__(self, href):
                        self.url = href
                return Destination(renditions[0]["href"])
        return None


class DocumentCreateJobStatus(BaseModel):
    """Status response from Document Create polling"""
    jobId: str = Field(..., description="Job ID")
    outputs: List[DocumentCreateOutputStatus] = Field(..., description="List of outputs with their statuses")
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def status(self) -> PhotoshopJobStatusEnum:
        """Get overall status from first output (for compatibility with polling)"""
        if self.outputs and len(self.outputs) > 0:
            return self.outputs[0].status
        return PhotoshopJobStatusEnum.PENDING

    @property
    def result(self):
        """Build a result object for compatibility with existing code"""
        if not self.outputs or len(self.outputs) == 0:
            return None

        succeeded_outputs = [o for o in self.outputs if o.status == PhotoshopJobStatusEnum.SUCCEEDED]
        if not succeeded_outputs:
            return None

        class JobResult:
            def __init__(self, outputs):
                self.outputs = outputs

        return JobResult(succeeded_outputs)


# ============================================================================
# Photoshop Document Manifest Models
# ============================================================================

class DocumentManifestInput(BaseModel):
    """Input specification for Document Manifest"""
    href: str = Field(..., description="Pre-signed GET URL to PSD file")
    storage: str = Field(default="external", description="Storage type (external, azure, dropbox)")


class ThumbnailOptions(BaseModel):
    """Thumbnail options for document manifest"""
    type: Optional[str] = Field("image/png", description="Thumbnail format (image/png, image/jpeg)")


class DocumentManifestOptions(BaseModel):
    """Options for Document Manifest extraction"""
    thumbnails: Optional[ThumbnailOptions] = Field(None, description="Include thumbnail URLs for renderable layers")


class DocumentManifestRequest(BaseModel):
    """Request for Document Manifest extraction"""
    inputs: List[DocumentManifestInput] = Field(..., description="Input PSD files to extract manifest from")
    options: Optional[DocumentManifestOptions] = Field(None, description="Manifest extraction options")


class DocumentManifestResponse(BaseModel):
    """Response from Document Manifest submission"""
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def jobId(self) -> Optional[str]:
        """Extract job ID from _links if available"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            href = self.links['self']['href']
            parts = href.split('/')
            if len(parts) > 0:
                return parts[-1]
        return None

    @property
    def statusUrl(self) -> Optional[str]:
        """Extract status URL from _links"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            return self.links['self']['href']
        return None


class ManifestBounds(BaseModel):
    """Bounds for layers in manifest"""
    top: Optional[int] = Field(None, description="Top position")
    left: Optional[int] = Field(None, description="Left position")
    width: Optional[int] = Field(None, description="Width")
    height: Optional[int] = Field(None, description="Height")


class ManifestBlendOptions(BaseModel):
    """Blend options in manifest"""
    opacity: Optional[int] = Field(None, description="Layer opacity (0-100)")
    blendMode: Optional[str] = Field(None, description="Blend mode")


class ManifestMask(BaseModel):
    """Mask info in manifest"""
    enabled: Optional[bool] = Field(None, description="Whether mask is enabled")
    linked: Optional[bool] = Field(None, description="Whether mask is linked")
    offset: Optional[Offset] = Field(None, description="Mask offset")


class ManifestTextStyle(BaseModel):
    """Text style info in manifest"""
    fontName: Optional[str] = Field(None, description="Font name")
    fontSize: Optional[float] = Field(None, description="Font size")
    fontColor: Optional[dict] = Field(None, description="Font color")


class ManifestText(BaseModel):
    """Text layer info in manifest"""
    content: Optional[str] = Field(None, description="Text content")
    characterStyles: Optional[List[dict]] = Field(None, description="Character styles")
    paragraphStyles: Optional[List[dict]] = Field(None, description="Paragraph styles")


class ManifestSmartObject(BaseModel):
    """Smart object info in manifest"""
    type: Optional[str] = Field(None, description="Smart object type")
    linked: Optional[bool] = Field(None, description="Whether linked")
    name: Optional[str] = Field(None, description="Smart object name")
    instanceId: Optional[str] = Field(None, description="Instance ID")
    path: Optional[str] = Field(None, description="Path for linked smart object")


class ManifestLayerInfo(BaseModel):
    """Layer information in document manifest"""
    id: Optional[int] = Field(None, description="Layer ID")
    index: Optional[int] = Field(None, description="Layer index")
    type: Optional[str] = Field(None, description="Layer type (layer, textLayer, adjustmentLayer, layerSection, backgroundLayer, etc.)")
    name: Optional[str] = Field(None, description="Layer name")
    locked: Optional[bool] = Field(None, description="Whether layer is locked")
    visible: Optional[bool] = Field(None, description="Whether layer is visible")
    bounds: Optional[ManifestBounds] = Field(None, description="Layer bounds")
    blendOptions: Optional[ManifestBlendOptions] = Field(None, description="Blend options")
    mask: Optional[ManifestMask] = Field(None, description="Mask info")
    text: Optional[ManifestText] = Field(None, description="Text layer info")
    smartObject: Optional[ManifestSmartObject] = Field(None, description="Smart object info")
    adjustments: Optional[dict] = Field(None, description="Adjustment layer settings")
    fill: Optional[dict] = Field(None, description="Fill layer settings")
    children: Optional[List["ManifestLayerInfo"]] = Field(None, description="Child layers for groups")
    thumbnail: Optional[str] = Field(None, description="Thumbnail URL if requested")


class ManifestDocumentInfo(BaseModel):
    """Document information in manifest"""
    name: Optional[str] = Field(None, description="Document name")
    width: Optional[int] = Field(None, description="Document width in pixels")
    height: Optional[int] = Field(None, description="Document height in pixels")
    photoshopBuild: Optional[str] = Field(None, description="Photoshop build version")
    imageMode: Optional[str] = Field(None, description="Image mode (rgb, cmyk, etc.)")
    bitDepth: Optional[int] = Field(None, description="Bit depth (8, 16, 32)")


class ManifestOutput(BaseModel):
    """Output in manifest job status"""
    input: Optional[str] = Field(None, description="Input URL that was processed")
    status: Optional[str] = Field(None, description="Status of this output")
    created: Optional[str] = Field(None, description="Creation timestamp")
    modified: Optional[str] = Field(None, description="Modification timestamp")
    document: Optional[ManifestDocumentInfo] = Field(None, description="Document information")
    layers: Optional[List[ManifestLayerInfo]] = Field(None, description="Layer tree")
    errors: Optional[Union[dict, List[dict]]] = Field(None, description="Errors if any")


class DocumentManifestJobStatus(BaseModel):
    """Status response from Document Manifest polling"""
    jobId: Optional[str] = Field(None, description="Job ID")
    outputs: Optional[List[ManifestOutput]] = Field(None, description="Manifest outputs")
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def status(self) -> str:
        """Get overall status from first output (for compatibility with polling)"""
        if self.outputs and len(self.outputs) > 0:
            return self.outputs[0].status or "pending"
        return "pending"


# ============================================================================
# Photoshop Document Operations Models
# ============================================================================

class DocumentOperationsInput(BaseModel):
    """Input specification for Document Operations"""
    href: str = Field(..., description="Pre-signed GET URL to PSD file")
    storage: str = Field(default="external", description="Storage type (external, azure, dropbox)")


class DocumentOperationsOutput(BaseModel):
    """Output specification for Document Operations"""
    href: str = Field(..., description="Pre-signed PUT URL for output")
    storage: str = Field(default="external", description="Storage type")
    type: str = Field(default="image/vnd.adobe.photoshop", description="Output MIME type (image/vnd.adobe.photoshop, image/png, image/jpeg, image/tiff)")
    overwrite: Optional[bool] = Field(True, description="Overwrite existing file")
    quality: Optional[int] = Field(None, description="Quality for JPEG output (1-12)", ge=1, le=12)
    compression: Optional[str] = Field(None, description="Compression for PNG (small, medium, large)")


class DocumentOperationsRequest(BaseModel):
    """Request for Document Operations (modify existing PSD)"""
    inputs: List[DocumentOperationsInput] = Field(..., description="Input PSD files (only one supported)")
    outputs: List[DocumentOperationsOutput] = Field(..., description="Output specifications")
    options: dict = Field(..., description="Document operation options including layers array")


class DocumentOperationsResponse(BaseModel):
    """Response from Document Operations submission"""
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def jobId(self) -> Optional[str]:
        """Extract job ID from _links if available"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            href = self.links['self']['href']
            parts = href.split('/')
            if len(parts) > 0:
                return parts[-1]
        return None

    @property
    def statusUrl(self) -> Optional[str]:
        """Extract status URL from _links"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            return self.links['self']['href']
        return None


class DocumentOperationsOutputStatus(BaseModel):
    """Status of a single output in the outputs array for Document Operations API"""
    status: PhotoshopJobStatusEnum = Field(..., description="Status of this output")
    created: Optional[str] = Field(None, description="Creation timestamp")
    modified: Optional[str] = Field(None, description="Last modification timestamp")
    links: Optional[dict] = Field(None, alias="_links", description="Links including renditions")
    input: Optional[str] = Field(None, description="Input URL that was processed")
    errors: Optional[Union[dict, List[dict]]] = Field(None, description="Errors if any")

    class Config:
        populate_by_name = True

    @property
    def destination(self):
        """Extract destination URL from renditions"""
        if self.links and "renditions" in self.links:
            renditions = self.links["renditions"]
            if isinstance(renditions, list) and len(renditions) > 0:
                class Destination:
                    def __init__(self, href):
                        self.url = href
                return Destination(renditions[0]["href"])
        return None


class DocumentOperationsJobStatus(BaseModel):
    """Status response from Document Operations polling"""
    jobId: str = Field(..., description="Job ID")
    outputs: List[DocumentOperationsOutputStatus] = Field(..., description="List of outputs with their statuses")
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def status(self) -> PhotoshopJobStatusEnum:
        """Get overall status from first output (for compatibility with polling)"""
        if self.outputs and len(self.outputs) > 0:
            return self.outputs[0].status
        return PhotoshopJobStatusEnum.PENDING

    @property
    def result(self):
        """Build a result object for compatibility with existing code"""
        if not self.outputs or len(self.outputs) == 0:
            return None

        succeeded_outputs = [o for o in self.outputs if o.status == PhotoshopJobStatusEnum.SUCCEEDED]
        if not succeeded_outputs:
            return None

        class JobResult:
            def __init__(self, outputs):
                self.outputs = outputs

        return JobResult(succeeded_outputs)


# ============================================================================
# Photoshop Text Edit Models
# ============================================================================

class TextOptionsLayerText(BaseModel):
    """Text content and style options for a text layer"""
    content: Optional[str] = Field(None, description="The content of the text layer")
    orientation: Optional[str] = Field(None, description="Text orientation (horizontal, vertical)")
    antiAlias: Optional[str] = Field(None, description="Anti-alias type (antiAliasNone, antiAliasSharp, etc.)")
    characterStyles: Optional[List[dict]] = Field(None, description="Array of character style objects")
    paragraphStyles: Optional[List[dict]] = Field(None, description="Array of paragraph style objects")


class TextOptionsLayer(BaseModel):
    """Layer specification for text editing"""
    id: Optional[int] = Field(None, description="The layer ID (use either id or name)")
    name: Optional[str] = Field(None, description="The layer name (use either id or name)")
    locked: Optional[bool] = Field(None, description="Is the layer editable")
    visible: Optional[bool] = Field(None, description="Is the layer visible")
    text: Optional[TextOptionsLayerText] = Field(None, description="Text layer attributes")
    bounds: Optional[DocumentLayerBounds] = Field(None, description="Layer bounds")


class TextOptions(BaseModel):
    """Options for text layer editing"""
    layers: List[TextOptionsLayer] = Field(..., description="Array of text layer objects to edit")
    manageMissingFonts: Optional[str] = Field(None, description="Action for missing fonts (useDefault, fail)")
    globalFont: Optional[str] = Field(None, description="PostScript name of global default font")
    fonts: Optional[List[PhotoshopAsset]] = Field(None, description="Array of custom fonts")


class TextEditRequest(BaseModel):
    """Request for text layer editing"""
    inputs: List[PhotoshopActionsInput] = Field(..., description="Input PSD files (only one supported)")
    outputs: List[PhotoshopActionsOutput] = Field(..., description="Output specifications")
    options: TextOptions = Field(..., description="Text editing options")


class TextEditResponse(BaseModel):
    """Response from Text Edit submission"""
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def jobId(self) -> Optional[str]:
        """Extract job ID from _links if available"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            href = self.links['self']['href']
            parts = href.split('/')
            if len(parts) > 0:
                return parts[-1]
        return None

    @property
    def statusUrl(self) -> Optional[str]:
        """Extract status URL from _links"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            return self.links['self']['href']
        return None


class TextEditOutputStatus(BaseModel):
    """Status of a single output in the outputs array for Text Edit API"""
    input: str = Field(..., description="Input URL that was processed")
    status: PhotoshopJobStatusEnum = Field(..., description="Status of this output")
    created: str = Field(..., description="Creation timestamp")
    modified: str = Field(..., description="Last modification timestamp")
    links: Optional[dict] = Field(None, alias="_links", description="Links including renditions")
    errors: Optional[Union[dict, List[dict]]] = Field(None, description="Errors if any")

    class Config:
        populate_by_name = True

    @property
    def destination(self):
        """Extract destination URL from renditions"""
        if self.links and "renditions" in self.links:
            renditions = self.links["renditions"]
            if isinstance(renditions, list) and len(renditions) > 0:
                class Destination:
                    def __init__(self, href):
                        self.url = href
                return Destination(renditions[0]["href"])
        return None


class TextEditJobStatus(BaseModel):
    """Status response from Text Edit polling"""
    jobId: str = Field(..., description="Job ID")
    outputs: List[TextEditOutputStatus] = Field(..., description="List of outputs with their statuses")
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def status(self) -> PhotoshopJobStatusEnum:
        """Get overall status from first output (for compatibility with polling)"""
        if self.outputs and len(self.outputs) > 0:
            return self.outputs[0].status
        return PhotoshopJobStatusEnum.PENDING

    @property
    def result(self):
        """Build a result object for compatibility with existing code"""
        if not self.outputs or len(self.outputs) == 0:
            return None

        succeeded_outputs = [o for o in self.outputs if o.status == PhotoshopJobStatusEnum.SUCCEEDED]
        if not succeeded_outputs:
            return None

        class JobResult:
            def __init__(self, outputs):
                self.outputs = outputs

        return JobResult(succeeded_outputs)


# ============================================================================
# Photoshop Rendition Create Models
# ============================================================================

class LayerReference(BaseModel):
    """Reference to a layer by ID or name"""
    id: Optional[int] = Field(None, description="Layer ID")
    name: Optional[str] = Field(None, description="Layer name")


class RenditionOutput(BaseModel):
    """Output specification for rendition creation"""
    href: str = Field(..., description="Pre-signed PUT URL for output")
    storage: str = Field(default="external", description="Storage type")
    type: str = Field(..., description="Output MIME type (image/png, image/jpeg, image/tiff, image/vnd.adobe.photoshop)")
    overwrite: Optional[bool] = Field(True, description="Overwrite existing file")
    width: Optional[int] = Field(None, description="Width in pixels (0 for full size)")
    maxWidth: Optional[int] = Field(None, description="Max width in pixels")
    quality: Optional[int] = Field(None, description="Quality for JPEG (1-7)", ge=1, le=7)
    compression: Optional[str] = Field(None, description="Compression for PNG (small, medium, large)")
    trimToCanvas: Optional[bool] = Field(None, description="Trim to canvas bounds")
    layers: Optional[List[LayerReference]] = Field(None, description="Specific layers to render (omit for full document)")


class RenditionCreateRequest(BaseModel):
    """Request for rendition creation"""
    inputs: List[PhotoshopActionsInput] = Field(..., description="Input PSD files (only one supported)")
    outputs: List[RenditionOutput] = Field(..., description="Output specifications")


class RenditionCreateResponse(BaseModel):
    """Response from Rendition Create submission"""
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def jobId(self) -> Optional[str]:
        """Extract job ID from _links if available"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            href = self.links['self']['href']
            parts = href.split('/')
            if len(parts) > 0:
                return parts[-1]
        return None

    @property
    def statusUrl(self) -> Optional[str]:
        """Extract status URL from _links"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            return self.links['self']['href']
        return None


class RenditionOutputStatus(BaseModel):
    """Status of a single output in the outputs array for Rendition Create API"""
    input: Optional[str] = Field(None, description="Input URL that was processed")
    status: PhotoshopJobStatusEnum = Field(..., description="Status of this output")
    created: Optional[str] = Field(None, description="Creation timestamp")
    modified: Optional[str] = Field(None, description="Last modification timestamp")
    links: Optional[dict] = Field(None, alias="_links", description="Links including renditions")
    errors: Optional[Union[dict, List[dict]]] = Field(None, description="Errors if any")

    class Config:
        populate_by_name = True

    @property
    def destination(self):
        """Extract destination URL from renditions"""
        if self.links and "renditions" in self.links:
            renditions = self.links["renditions"]
            if isinstance(renditions, list) and len(renditions) > 0:
                class Destination:
                    def __init__(self, href):
                        self.url = href
                return Destination(renditions[0]["href"])
        return None


class RenditionCreateJobStatus(BaseModel):
    """Status response from Rendition Create polling"""
    jobId: str = Field(..., description="Job ID")
    outputs: List[RenditionOutputStatus] = Field(..., description="List of outputs with their statuses")
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def status(self) -> PhotoshopJobStatusEnum:
        """Get overall status from first output (for compatibility with polling)"""
        if self.outputs and len(self.outputs) > 0:
            return self.outputs[0].status
        return PhotoshopJobStatusEnum.PENDING

    @property
    def result(self):
        """Build a result object for compatibility with existing code"""
        if not self.outputs or len(self.outputs) == 0:
            return None

        succeeded_outputs = [o for o in self.outputs if o.status == PhotoshopJobStatusEnum.SUCCEEDED]
        if not succeeded_outputs:
            return None

        class JobResult:
            def __init__(self, outputs):
                self.outputs = outputs

        return JobResult(succeeded_outputs)


# ============================================================================
# Photoshop Smart Object Models
# ============================================================================

class SmartObjectLayerInput(BaseModel):
    """Input specification for smart object replacement"""
    href: str = Field(..., description="Pre-signed URL to replacement image")
    storage: str = Field(default="external", description="Storage type")


class SmartObjectLayerPosition(BaseModel):
    """Position specification for adding smart object layer"""
    insertAbove: Optional[dict] = Field(None, description="Insert above layer {id or name}")
    insertBelow: Optional[dict] = Field(None, description="Insert below layer {id or name}")
    insertInto: Optional[dict] = Field(None, description="Insert into layer group {id or name}")
    insertTop: Optional[bool] = Field(None, description="Insert at top of layer stack")
    insertBottom: Optional[bool] = Field(None, description="Insert at bottom of layer stack")


class SmartObjectLayer(BaseModel):
    """Smart object layer specification"""
    id: Optional[int] = Field(None, description="Layer ID to replace")
    name: Optional[str] = Field(None, description="Layer name to replace")
    input: SmartObjectLayerInput = Field(..., description="Replacement image input")
    bounds: Optional[DocumentLayerBounds] = Field(None, description="Layer bounds")
    locked: Optional[bool] = Field(False, description="Is the layer locked")
    visible: Optional[bool] = Field(True, description="Is the layer visible")
    add: Optional[SmartObjectLayerPosition] = Field(None, description="Position for adding new smart object")


class SmartObjectOptions(BaseModel):
    """Options for smart object replacement"""
    layers: List[SmartObjectLayer] = Field(..., description="Array of smart object layers")


class SmartObjectOutput(BaseModel):
    """Output specification for smart object replacement"""
    href: str = Field(..., description="Pre-signed PUT URL for output")
    storage: str = Field(default="external", description="Storage type")
    type: str = Field(default="image/vnd.adobe.photoshop", description="Output MIME type")
    overwrite: Optional[bool] = Field(True, description="Overwrite existing file")
    width: Optional[int] = Field(None, description="Width in pixels (0 for full size)")
    quality: Optional[int] = Field(None, description="Quality for JPEG (1-7)", ge=1, le=7)
    compression: Optional[str] = Field(None, description="Compression for PNG (small, medium, large)")


class SmartObjectRequest(BaseModel):
    """Request for smart object replacement"""
    inputs: List[PhotoshopActionsInput] = Field(..., description="Input PSD files (only one supported)")
    outputs: List[SmartObjectOutput] = Field(..., description="Output specifications")
    options: SmartObjectOptions = Field(..., description="Smart object replacement options")


class SmartObjectResponse(BaseModel):
    """Response from Smart Object submission"""
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def jobId(self) -> Optional[str]:
        """Extract job ID from _links if available"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            href = self.links['self']['href']
            parts = href.split('/')
            if len(parts) > 0:
                return parts[-1]
        return None

    @property
    def statusUrl(self) -> Optional[str]:
        """Extract status URL from _links"""
        if self.links and 'self' in self.links and 'href' in self.links['self']:
            return self.links['self']['href']
        return None


class SmartObjectOutputStatus(BaseModel):
    """Status of a single output in the outputs array for Smart Object API"""
    input: Optional[str] = Field(None, description="Input URL that was processed")
    status: PhotoshopJobStatusEnum = Field(..., description="Status of this output")
    created: Optional[str] = Field(None, description="Creation timestamp")
    modified: Optional[str] = Field(None, description="Last modification timestamp")
    links: Optional[dict] = Field(None, alias="_links", description="Links including renditions")
    errors: Optional[Union[dict, List[dict]]] = Field(None, description="Errors if any")

    class Config:
        populate_by_name = True

    @property
    def destination(self):
        """Extract destination URL from renditions"""
        if self.links and "renditions" in self.links:
            renditions = self.links["renditions"]
            if isinstance(renditions, list) and len(renditions) > 0:
                class Destination:
                    def __init__(self, href):
                        self.url = href
                return Destination(renditions[0]["href"])
        return None


class SmartObjectJobStatus(BaseModel):
    """Status response from Smart Object polling"""
    jobId: str = Field(..., description="Job ID")
    outputs: List[SmartObjectOutputStatus] = Field(..., description="List of outputs with their statuses")
    links: Optional[dict] = Field(None, alias="_links", description="Job status links")

    class Config:
        populate_by_name = True

    @property
    def status(self) -> PhotoshopJobStatusEnum:
        """Get overall status from first output (for compatibility with polling)"""
        if self.outputs and len(self.outputs) > 0:
            return self.outputs[0].status
        return PhotoshopJobStatusEnum.PENDING

    @property
    def result(self):
        """Build a result object for compatibility with existing code"""
        if not self.outputs or len(self.outputs) == 0:
            return None

        succeeded_outputs = [o for o in self.outputs if o.status == PhotoshopJobStatusEnum.SUCCEEDED]
        if not succeeded_outputs:
            return None

        class JobResult:
            def __init__(self, outputs):
                self.outputs = outputs

        return JobResult(succeeded_outputs)
