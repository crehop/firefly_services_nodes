"""
Pydantic models for Adobe InDesign Data Merge API requests and responses.

Base URL: https://indesign.adobe.io
Endpoints:
  POST /v3/merge-data       - Submit data merge job
  GET  /v3/status/{id}      - Poll job status
Auth: Bearer JWT + x-api-key from Adobe IMS OAuth 2.0
"""

from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


# ── Request Sub-models ────────────────────────────────────────────────────────

class AssetSource(BaseModel):
    url: str


class Asset(BaseModel):
    destination: str
    source: AssetSource


class OutputDestination(BaseModel):
    url: str
    storageType: Optional[str] = Field(default="AWS")


class OutputAsset(BaseModel):
    """Output specification — tells the API where to PUT the result."""
    destination: OutputDestination
    source: str  # relative path from working dir (e.g. "output/image_1.png")


class DataMergeParams(BaseModel):
    dataSource: str
    outputFolderPath: str = Field(default="output")
    outputFileBaseString: str = Field(default="image_")
    outputMediaType: str = Field(default="image/png")
    targetDocument: str
    recordRange: Optional[str] = None
    removeBlankLines: Optional[bool] = None
    convertUrlToHyperlink: Optional[bool] = None


# ── Request Model ─────────────────────────────────────────────────────────────

class DataMergeRequest(BaseModel):
    """POST /v3/merge-data"""
    assets: List[Asset]
    params: DataMergeParams
    outputs: Optional[List[OutputAsset]] = None


# ── Response Models ───────────────────────────────────────────────────────────

class InDesignJobOutputDestination(BaseModel):
    url: Optional[str] = None


class InDesignJobOutput(BaseModel):
    destination: Optional[InDesignJobOutputDestination] = None
    source: Optional[str] = None


class InDesignJobResponse(BaseModel):
    """Generic InDesign API job response for submit and polling."""
    statusUrl: Optional[str] = None
    status: Optional[str] = None
    jobId: Optional[str] = None
    outputs: Optional[List[InDesignJobOutput]] = None
    error: Optional[dict] = None
    errorDetail: Optional[str] = None
    percentComplete: Optional[int] = None
    data: Optional[dict] = None
