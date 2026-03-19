"""
Pydantic models for Adobe Substance 3D API requests and responses.

Base URL: https://s3d.adobe.io
Auth: Bearer JWT from Adobe IMS OAuth 2.0 (same as Firefly/Photoshop)
"""

from __future__ import annotations
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class S3DOutputFormat(str, Enum):
    GLB = "glb"
    GLTF = "gltf"
    FBX = "fbx"
    USDZ = "usdz"
    USDA = "usda"
    USDC = "usdc"
    OBJ = "obj"


class S3DContentClass(str, Enum):
    ART = "art"
    PHOTO = "photo"


class S3DModelVersion(str, Enum):
    IMAGE3_FAST = "image3_fast"
    IMAGE4_STANDARD = "image4_standard"
    IMAGE4_ULTRA = "image4_ultra"


class S3DAutoFramingAlgorithm(str, Enum):
    AUTO = "auto"
    BOUNDING_CYLINDER = "bounding_cylinder"
    FRUSTUM_FIT = "frustum_fit"


class S3DTurntableMode(str, Enum):
    ROTATE_CAMERA = "rotate_camera"
    ROTATE_MODEL = "rotate_model"
    ROTATE_ENVIRONMENT = "rotate_environment"


class S3DJobStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# ── Common Sub-models ──────────────────────────────────────────────────────────

class MountedSourceURL(BaseModel):
    url: str
    filename: Optional[str] = None


class MountedSource(BaseModel):
    url: Optional[MountedSourceURL] = None
    space: Optional[dict] = None
    mountPoint: Optional[str] = None


class OutputSize(BaseModel):
    width: int = Field(default=2688)
    height: int = Field(default=1536)


class SizeOptions(BaseModel):
    width: int = Field(default=1920, ge=16, le=3840)
    height: int = Field(default=1080, ge=16, le=2304)


class AutoFramingOptions(BaseModel):
    algorithm: Optional[str] = Field(default="auto")
    zoom: Optional[float] = None


class BackgroundOptions(BaseModel):
    color: Optional[List[float]] = None
    backgroundImage: Optional[str] = None
    showEnvironment: Optional[bool] = None


class GroundPlaneOptions(BaseModel):
    enable: Optional[bool] = Field(default=True)
    autoGroundScene: Optional[bool] = Field(default=True)
    shadows: Optional[bool] = Field(default=True)
    shadowsOpacity: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    reflections: Optional[bool] = Field(default=False)
    reflectionsOpacity: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    reflectionsRoughness: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)


class RenderExtraOutputs(BaseModel):
    exportMaterialIds: Optional[bool] = None
    exportObjectIds: Optional[bool] = None


class SceneCamera(BaseModel):
    focal: Optional[float] = Field(default=50.0, ge=10.0, le=1000.0)
    sensorWidth: Optional[float] = Field(default=36.0, ge=1.0, le=100.0)
    transform: Optional[dict] = None


class SceneEnvironment(BaseModel):
    file: Optional[str] = None
    exposure: Optional[float] = Field(default=None, ge=-10.0, le=10.0)
    rotation: Optional[float] = None
    preserveLights: Optional[bool] = None


class SbsMaterial(BaseModel):
    sbsar: str
    resolution: Optional[int] = Field(default=1024)


class MaterialAssign(BaseModel):
    materialName: str
    material: Optional[dict] = None
    nodeList: Optional[List[str]] = None
    assignByDefault: Optional[bool] = None


class SceneModelTransform(BaseModel):
    trs: Optional[dict] = None
    matrix: Optional[List[float]] = None


class SceneModelImport(BaseModel):
    file: str
    anchorName: Optional[str] = None
    transform: Optional[SceneModelTransform] = None
    materialOverrides: Optional[List[dict]] = None


class SceneModels(BaseModel):
    imports: Optional[List[SceneModelImport]] = None


class SceneDescription(BaseModel):
    baseFile: Optional[dict] = None
    camera: Optional[SceneCamera] = None
    environment: Optional[SceneEnvironment] = None
    materials: Optional[List[MaterialAssign]] = None
    models: Optional[SceneModels] = None
    metersPerUnit: Optional[float] = Field(default=0.01)


class SimpleSceneDescription(BaseModel):
    modelFile: str
    camera: Optional[SceneCamera] = None
    environment: Optional[SceneEnvironment] = None
    materials: Optional[List[MaterialAssign]] = None
    materialOverrides: Optional[List[dict]] = None
    metersPerUnit: Optional[float] = Field(default=0.01)


class ComposeEnvironment(BaseModel):
    file: str
    rotation: Optional[float] = None


class ComposeSceneDetails(BaseModel):
    camera: Optional[SceneCamera] = None


# ── Request Models ─────────────────────────────────────────────────────────────

class RenderBasicRequest(BaseModel):
    """POST /v1/scenes/render-basic"""
    scene: SimpleSceneDescription
    sources: List[MountedSource]
    autoFraming: Optional[AutoFramingOptions] = None
    background: Optional[BackgroundOptions] = None
    cameraName: Optional[str] = None
    extraOutputs: Optional[RenderExtraOutputs] = None
    groundPlane: Optional[GroundPlaneOptions] = None
    size: Optional[SizeOptions] = None


class RenderSceneRequest(BaseModel):
    """POST /v1/scenes/render"""
    scene: SceneDescription
    sources: List[MountedSource]
    autoFraming: Optional[AutoFramingOptions] = None
    background: Optional[BackgroundOptions] = None
    cameraName: Optional[str] = None
    extraOutputs: Optional[RenderExtraOutputs] = None
    groundPlane: Optional[GroundPlaneOptions] = None
    size: Optional[SizeOptions] = None


class CompositeRequest(BaseModel):
    """POST /v1/composites/compose"""
    sources: List[MountedSource]
    prompt: str
    heroAsset: str
    cameraName: Optional[str] = None
    contentClass: Optional[str] = Field(default="photo")
    enableGroundPlane: Optional[bool] = Field(default=False)
    environment: Optional[ComposeEnvironment] = None
    environmentExposure: Optional[float] = Field(default=None, ge=-10.0, le=10.0)
    lightingSeeds: Optional[List[int]] = None
    modelVersion: Optional[str] = Field(default="image4_ultra")
    numVariations: Optional[int] = Field(default=1, ge=1, le=4)
    scene: Optional[ComposeSceneDetails] = None
    sceneFile: Optional[str] = None
    seeds: Optional[List[int]] = None
    size: Optional[OutputSize] = None
    styleImage: Optional[str] = None


class ConvertRequest(BaseModel):
    """POST /v1/scenes/convert"""
    format: str
    sources: List[MountedSource]
    modelEntrypoint: Optional[str] = None


class AssembleRequest(BaseModel):
    """POST /v1/scenes/assemble"""
    scene: SceneDescription
    sources: List[MountedSource]
    encoding: str
    fileBaseName: str


class DescribeRequest(BaseModel):
    """POST /v1/scenes/describe"""
    sources: List[MountedSource]
    sceneFile: Optional[str] = None


# ── Response Models ────────────────────────────────────────────────────────────

class S3DSpaceFile(BaseModel):
    name: Optional[str] = None
    size: Optional[int] = None
    url: Optional[str] = None


class S3DSpace(BaseModel):
    id: Optional[str] = None
    url: Optional[str] = None
    expiry: Optional[str] = None
    files: Optional[List[S3DSpaceFile]] = None
    archiveUrl: Optional[str] = None


class ComposeOutputImage(BaseModel):
    url: Optional[str] = None


class ComposeOutput(BaseModel):
    image: Optional[ComposeOutputImage] = None
    backgroundImage: Optional[ComposeOutputImage] = None
    maskImage: Optional[ComposeOutputImage] = None
    seed: Optional[int] = None
    lightingSeed: Optional[int] = None


class ComposeJobResult(BaseModel):
    outputs: Optional[List[ComposeOutput]] = None
    outputSpace: Optional[S3DSpace] = None
    promptHasDeniedWords: Optional[bool] = None
    promptHasBlockedArtists: Optional[bool] = None
    warnings: Optional[List[dict]] = None


class RenderJobResult(BaseModel):
    renderUrl: Optional[str] = None
    framesUrls: Optional[List[str]] = None
    outputSpace: Optional[S3DSpace] = None
    warnings: Optional[List[dict]] = None


class ConvertJobResult(BaseModel):
    outputSpace: Optional[S3DSpace] = None


class AssembleJobResult(BaseModel):
    sceneUrl: Optional[str] = None
    outputSpace: Optional[S3DSpace] = None


class DescribeStatsNode(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    children: Optional[List[dict]] = None
    assignedMaterialName: Optional[str] = None


class DescribeStats(BaseModel):
    metersPerSceneUnit: Optional[float] = None
    sceneUpAxis: Optional[str] = None
    numVertices: Optional[int] = None
    numTriangles: Optional[int] = None
    numEquivalentTriangles: Optional[int] = None
    numFaces: Optional[int] = None
    numMeshes: Optional[int] = None
    numTextures: Optional[int] = None
    cameraNames: Optional[List[str]] = None
    materialNames: Optional[List[str]] = None
    nodesHierarchy: Optional[DescribeStatsNode] = None


class DescribeJobResult(BaseModel):
    stats: Optional[DescribeStats] = None


class S3DJobResponse(BaseModel):
    """Generic job response — used for all endpoints and polling."""
    id: Optional[str] = None
    status: str = "not_started"
    url: Optional[str] = None
    bugReportUrl: Optional[str] = None
    error: Optional[str] = None
    # Result fields vary by endpoint — we use Optional for all
    result: Optional[dict] = None
