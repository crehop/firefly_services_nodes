"""
Pydantic models for Adobe Audio-Video Reframe API requests and responses.

Base URL: https://audio-video-api.adobe.io
Endpoints:
  POST /v2/reframe          - Submit reframe job
  GET  /v2/status/{jobId}   - Poll job status
Auth: Bearer JWT from Adobe IMS OAuth 2.0 (same as Firefly/Photoshop)
"""

from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


# -- Request Models --

class SourceURL(BaseModel):
    url: str


class OverlayScale(BaseModel):
    width: int = 1080
    height: int = 1920


class OverlayPosition(BaseModel):
    anchorPoint: str = "top_left"
    offsetX: int = 0
    offsetY: int = 0


class Overlay(BaseModel):
    source: SourceURL
    startTime: Optional[str] = None
    duration: Optional[str] = None
    scale: Optional[OverlayScale] = None
    position: Optional[OverlayPosition] = None
    repeat: Optional[str] = None


class Composition(BaseModel):
    overlays: List[Overlay]


class RenditionResolution(BaseModel):
    width: int = 1080
    height: int = 1920


class Rendition(BaseModel):
    resolution: Optional[RenditionResolution] = None


class OutputFormat(BaseModel):
    media: str = "mp4"


class OutputSettings(BaseModel):
    format: Optional[OutputFormat] = None
    renditions: List[Rendition]


class VideoSource(BaseModel):
    source: SourceURL


class ReframeRequest(BaseModel):
    """POST /v2/reframe"""
    video: VideoSource
    output: OutputSettings
    composition: Optional[Composition] = None


# -- Response Models --

class ReframeSubmitResponse(BaseModel):
    """202 response from POST /v2/reframe"""
    jobId: Optional[str] = None
    statusUrl: Optional[str] = None


class ReframeOutputDestination(BaseModel):
    url: Optional[str] = None


class ReframeOutputError(BaseModel):
    error_code: Optional[str] = None
    message: Optional[str] = None


class ReframeJobOutput(BaseModel):
    destination: Optional[ReframeOutputDestination] = None
    mediaDestination: Optional[ReframeOutputDestination] = None
    error: Optional[ReframeOutputError] = None


class ReframeJobResponse(BaseModel):
    """Response from GET /v2/status/{jobId}"""
    jobId: Optional[str] = None
    status: str = "not_started"
    outputs: Optional[List[ReframeJobOutput]] = None
