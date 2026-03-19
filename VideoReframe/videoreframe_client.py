"""
Video Reframe API Client Helper

Creates authenticated API clients for audio-video-api.adobe.io and provides
common helpers for the submit+poll async workflow.
"""

from __future__ import annotations
import logging
import asyncio
import aiohttp
import json
from typing import Optional

from ..adobe_auth import get_adobe_auth_manager
from .videoreframe_api import ReframeRequest, ReframeSubmitResponse, ReframeJobResponse

REFRAME_BASE_URL = "https://audio-video-api.adobe.io"


async def _get_auth_headers() -> dict:
    """Get Adobe OAuth headers."""
    auth_manager = get_adobe_auth_manager()
    return await auth_manager.get_auth_headers()


async def submit_reframe_job(request_data: ReframeRequest) -> ReframeJobResponse:
    """
    Submit a reframe job and poll until completion.

    Args:
        request_data: ReframeRequest with video, composition, and output settings

    Returns:
        ReframeJobResponse with status and output URLs
    """
    auth_headers = await _get_auth_headers()

    headers = {
        **auth_headers,
        "Content-Type": "application/json",
    }

    payload = request_data.model_dump(exclude_none=True)

    timeout = aiohttp.ClientTimeout(total=600.0)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Step 1: Submit the job
        url = f"{REFRAME_BASE_URL}/v2/reframe"
        logging.info(f"[VideoReframe] POST {url}")
        logging.info(f"[VideoReframe] Payload: {json.dumps(payload, indent=2)}")

        async with session.post(url, json=payload, headers=headers) as resp:
            resp_text = await resp.text()
            logging.info(f"[VideoReframe] Response status: {resp.status}")
            logging.info(f"[VideoReframe] Response: {resp_text[:500]}")

            if resp.status not in (200, 201, 202):
                raise Exception(
                    f"Reframe API error (status {resp.status}): {resp_text[:500]}"
                )

            if not resp_text or not resp_text.strip():
                raise Exception(f"Reframe API returned empty response (status {resp.status})")

            try:
                resp_json = json.loads(resp_text)
            except json.JSONDecodeError as e:
                raise Exception(
                    f"Reframe API returned non-JSON response (status {resp.status}): {resp_text[:300]}"
                ) from e

        # Parse submit response to get statusUrl
        submit_resp = ReframeSubmitResponse(**resp_json)

        if not submit_resp.jobId:
            raise Exception(f"No jobId in reframe submit response: {resp_json}")

        # Use statusUrl from response, or construct fallback
        poll_url = submit_resp.statusUrl or f"{REFRAME_BASE_URL}/v2/status/{submit_resp.jobId}"
        logging.info(f"[VideoReframe] Polling: {poll_url}")

        # Step 2: Poll for completion
        for attempt in range(120):
            await asyncio.sleep(5.0)

            async with session.get(poll_url, headers=headers) as poll_resp:
                poll_text = await poll_resp.text()

                if poll_resp.status == 403:
                    logging.warning(f"[VideoReframe] Poll 403 Forbidden - may need different auth")
                    raise Exception(f"Reframe poll returned 403 Forbidden. Check API auth scopes.")

                try:
                    poll_json = json.loads(poll_text)
                except json.JSONDecodeError:
                    logging.warning(f"[VideoReframe] Poll non-JSON (status {poll_resp.status}): {poll_text[:200]}")
                    continue

            status = poll_json.get("status", "unknown")
            logging.info(f"[VideoReframe] Poll #{attempt+1}: status={status}")
            if status in ("succeeded", "completed", "partially_succeeded"):
                logging.info(f"[VideoReframe] FULL POLL RESPONSE: {json.dumps(poll_json, indent=2)}")

            if status in ("succeeded", "completed", "partially_succeeded"):
                return ReframeJobResponse(**poll_json)
            elif status in ("failed", "error"):
                error_msg = json.dumps(poll_json.get("outputs", poll_json), indent=2)
                raise Exception(f"Reframe job failed: {error_msg}")

        raise Exception(f"Reframe job timed out after 120 poll attempts (jobId={submit_resp.jobId})")
