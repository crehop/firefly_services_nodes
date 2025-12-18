"""
Firefly Storage Helper for Adobe Firefly Nodes

Provides functions to upload images to Firefly's native storage
for use with Adobe Firefly API.
"""

import torch
from ..apinode_utils import tensor_to_bytesio
from .firefly_api import (
    UploadImageRequest,
    UploadImageResponse,
    FireflyImageFormat,
)
from ..client import (
    ApiClient,
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
)
from ..adobe_auth import get_adobe_auth_manager


async def create_adobe_client() -> ApiClient:
    """Create an ApiClient configured for Adobe Firefly API with OAuth authentication."""
    auth_manager = get_adobe_auth_manager()
    auth_headers = await auth_manager.get_auth_headers()

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
) -> tuple[str, str]:
    """
    Upload an image tensor to Firefly storage and return the upload ID and debug log.

    Args:
        image: Image tensor to upload
        total_pixels: Maximum total pixels for the image

    Returns:
        Tuple of (upload_id, debug_log)
    """
    client = await create_adobe_client()

    try:
        # Convert tensor to bytes
        image_bytes = tensor_to_bytesio(image, total_pixels=total_pixels)
        image_size = image_bytes.getbuffer().nbytes

        # Build debug log
        debug_log = ""
        debug_log += "-------------------------------------------------------\n"
        debug_log += "FIREFLY STORAGE UPLOAD\n"
        debug_log += "-------------------------------------------------------\n"
        debug_log += f"Size: {image_size:,} bytes\n"
        debug_log += f"Format: {FireflyImageFormat.IMAGE_PNG.value}\n"

        # Upload directly with binary data and image/png content-type
        headers = client.get_headers()
        headers["Content-Type"] = FireflyImageFormat.IMAGE_PNG.value

        # Send raw binary data (not JSON)
        image_bytes.seek(0)  # Reset buffer position
        response_json = await client.request(
            path="/v2/storage/image",
            method=HttpMethod.POST,
            data=image_bytes.read(),  # Send raw bytes
            headers=headers,
        )

        # Extract upload ID from response {"images": [{"id": "..."}]}
        if "images" in response_json and len(response_json["images"]) > 0:
            upload_id = response_json["images"][0]["id"]
            debug_log += f"Upload ID: {upload_id}\n"
            debug_log += "Status: ✓ Upload successful\n"
            return upload_id, debug_log
        else:
            raise Exception(f"Unexpected response format: {response_json}")

    except Exception as e:
        debug_log += f"Status: ✗ Upload failed - {str(e)}\n"
        raise Exception(f"Failed to upload image to Firefly: {str(e)}")
    finally:
        await client.close()
