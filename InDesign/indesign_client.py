"""
InDesign API Client Helper

Creates authenticated API clients for indesign.adobe.io and provides
the submit+poll async workflow for Data Merge jobs.
"""

from __future__ import annotations
import logging
import json
import asyncio
from typing import Optional
from urllib.parse import urlparse

from ..client import (
    ApiClient,
    ApiEndpoint,
    HttpMethod,
    EmptyRequest,
    SynchronousOperation,
)
from ..adobe_auth import get_adobe_auth_manager
from .indesign_api import DataMergeRequest, InDesignJobResponse

INDESIGN_BASE_URL = "https://indesign.adobe.io"

logger = logging.getLogger(__name__)


async def create_indesign_client() -> ApiClient:
    """Create an ApiClient configured for InDesign API with Adobe OAuth."""
    auth_manager = get_adobe_auth_manager()
    auth_headers = await auth_manager.get_auth_headers()

    client = ApiClient(
        base_url=INDESIGN_BASE_URL,
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


async def submit_and_poll_indesign(
    request_data: DataMergeRequest,
    node_id: Optional[str] = None,
    poll_interval: float = 5.0,
    max_poll_attempts: int = 120,
) -> InDesignJobResponse:
    """
    Submit a Data Merge job and poll until completion.

    Returns:
        InDesignJobResponse with outputs populated on success.
    """
    client = await create_indesign_client()

    try:
        # Step 1: Submit the job
        submit_endpoint = ApiEndpoint(
            path="/v3/merge-data",
            method=HttpMethod.POST,
            request_model=DataMergeRequest,
            response_model=InDesignJobResponse,
        )

        submit_op = SynchronousOperation(
            endpoint=submit_endpoint,
            request=request_data,
            api_base=INDESIGN_BASE_URL,
        )

        submit_response = await submit_op.execute(client=client)

        logger.info(f"[InDesign] Job submitted: jobId={submit_response.jobId} statusUrl={submit_response.statusUrl}")

        # If already completed, return
        if submit_response.status and submit_response.status.lower() in ("succeeded", "completed", "done"):
            return submit_response

        # Step 2: Poll for completion using statusUrl
        poll_url = submit_response.statusUrl
        if not poll_url:
            raise Exception(f"No statusUrl in InDesign response: {submit_response}")

        # Extract the path from the full URL
        parsed = urlparse(poll_url)
        poll_path = parsed.path

        poll_endpoint = ApiEndpoint(
            path=poll_path,
            method=HttpMethod.GET,
            request_model=EmptyRequest,
            response_model=InDesignJobResponse,
        )

        for attempt in range(max_poll_attempts):
            await asyncio.sleep(poll_interval)

            poll_op = SynchronousOperation(
                endpoint=poll_endpoint,
                request=EmptyRequest(),
                api_base=INDESIGN_BASE_URL,
            )
            poll_response = await poll_op.execute(client=client)

            status = (poll_response.status or "").lower()
            logger.info(f"[InDesign] Poll attempt {attempt + 1}: status={status}")

            if status in ("succeeded", "completed", "done", "partial_success"):
                return poll_response
            elif status in ("failed", "error"):
                error_msg = poll_response.error or poll_response.errorDetail or "Unknown error"
                raise Exception(f"InDesign job failed: {error_msg}")

        raise Exception(f"InDesign job timed out after {max_poll_attempts * poll_interval}s")

    finally:
        await client.close()
