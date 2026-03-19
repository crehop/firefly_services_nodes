"""
Substance 3D API Client Helper

Creates authenticated API clients for s3d.adobe.io and provides
common helpers for the submit+poll async workflow.
"""

from __future__ import annotations
import logging
import json
import asyncio
from typing import Optional

from ..client import (
    ApiClient,
    ApiEndpoint,
    HttpMethod,
    EmptyRequest,
    SynchronousOperation,
    PollingOperation,
)
from ..adobe_auth import get_adobe_auth_manager
from .substance3d_api import S3DJobResponse, MountedSource, MountedSourceURL

S3D_BASE_URL = "https://s3d.adobe.io"


async def create_s3d_client() -> ApiClient:
    """Create an ApiClient configured for Substance 3D API with Adobe OAuth."""
    auth_manager = get_adobe_auth_manager()
    auth_headers = await auth_manager.get_auth_headers()

    client = ApiClient(
        base_url=S3D_BASE_URL,
        verify_ssl=True,
        comfy_api_key="adobe_oauth",
        timeout=600.0,
    )

    client._adobe_headers = auth_headers

    original_get_headers = client.get_headers

    def get_headers_with_adobe():
        headers = original_get_headers()
        headers.update(client._adobe_headers)
        headers.pop("X-API-KEY", None)
        return headers

    client.get_headers = get_headers_with_adobe

    return client


def make_source(url: str, mount_point: str = "/") -> MountedSource:
    """Create a MountedSource from a presigned URL."""
    return MountedSource(
        url=MountedSourceURL(url=url),
        mountPoint=mount_point,
    )


def build_sources_list(*url_pairs: tuple[str, str]) -> list[MountedSource]:
    """
    Build a sources list from (url, filename) pairs.
    Skips any pairs where url is empty.
    """
    sources = []
    for url, filename in url_pairs:
        if url and url.strip():
            sources.append(MountedSource(
                url=MountedSourceURL(url=url),
                mountPoint="/",
            ))
    return sources


async def submit_and_poll_s3d(
    endpoint_path: str,
    request_model,
    request_data,
    node_id: Optional[str] = None,
    poll_interval: float = 5.0,
    max_poll_attempts: int = 120,
    estimated_duration: Optional[float] = None,
) -> S3DJobResponse:
    """
    Submit a job to S3D API and poll until completion.

    Args:
        endpoint_path: API path (e.g. "/v1/scenes/render-basic")
        request_model: Pydantic model class for the request
        request_data: Instance of the request model
        node_id: Optional ComfyUI node ID for progress display
        poll_interval: Seconds between poll attempts
        max_poll_attempts: Maximum number of poll attempts
        estimated_duration: Estimated job duration in seconds

    Returns:
        S3DJobResponse with status "succeeded" and result populated
    """
    client = await create_s3d_client()

    try:
        # Step 1: Submit the job
        submit_endpoint = ApiEndpoint(
            path=endpoint_path,
            method=HttpMethod.POST,
            request_model=request_model,
            response_model=S3DJobResponse,
        )

        submit_op = SynchronousOperation(
            endpoint=submit_endpoint,
            request=request_data,
            api_base=S3D_BASE_URL,
        )
        # Execute with our pre-authed client
        submit_response = await submit_op.execute(client=client)

        logging.info(f"[S3D] Job submitted: {submit_response.id} status={submit_response.status}")

        # If already succeeded (wait=true mode or instant), return
        if submit_response.status == "succeeded":
            return submit_response

        # Step 2: Poll for completion
        poll_url = submit_response.url
        if not poll_url:
            raise Exception(f"No poll URL in response: {submit_response}")

        # Extract the path from the full URL
        from urllib.parse import urlparse
        parsed = urlparse(poll_url)
        poll_path = parsed.path

        poll_endpoint = ApiEndpoint(
            path=poll_path,
            method=HttpMethod.GET,
            request_model=EmptyRequest,
            response_model=S3DJobResponse,
        )

        poll_op = PollingOperation(
            poll_endpoint=poll_endpoint,
            request=EmptyRequest(),
            completed_statuses=["succeeded"],
            failed_statuses=["failed"],
            status_extractor=lambda r: r.status,
            api_base=S3D_BASE_URL,
            poll_interval=poll_interval,
            max_poll_attempts=max_poll_attempts,
            node_id=node_id,
            estimated_duration=estimated_duration,
        )

        result = await poll_op.execute(client=client)
        return result

    finally:
        await client.close()
