"""
Photoshop Storage Helper for Adobe Photoshop API

Provides functions to upload images to AWS S3 for use with Adobe Photoshop API.
"""

import torch
import uuid
import logging
import asyncio
from typing import Tuple
from ..apinode_utils import tensor_to_bytesio
from ..adobe_auth import _load_firefly_config


def _upload_to_s3_sync(image_bytes_data: bytes, aws_config: dict, filename: str) -> str:
    """
    Synchronous S3 upload operation (to be run in executor).

    Args:
        image_bytes_data: Image data as bytes
        aws_config: Dict with AWS credentials and config
        filename: S3 object key/filename

    Returns:
        Pre-signed GET URL for the uploaded image
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise ImportError(
            "boto3 is required for AWS S3 uploads. Install it with: pip install boto3"
        )

    # Create S3 client with regional endpoint
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_config['aws_access_key_id'],
        aws_secret_access_key=aws_config['aws_secret_access_key'],
        region_name=aws_config['aws_region'],
        config=Config(
            signature_version='s3v4',
            s3={'addressing_style': 'path'}
        )
    )

    # Upload to S3
    s3_client.put_object(
        Bucket=aws_config['aws_bucket'],
        Key=filename,
        Body=image_bytes_data,
        ContentType='image/png'
    )

    logging.info(f"[Photoshop S3] Upload successful: s3://{aws_config['aws_bucket']}/{filename}")

    # Generate pre-signed GET URL (24 hours)
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': aws_config['aws_bucket'], 'Key': filename},
        ExpiresIn=86400  # 24 hours
    )

    logging.info(f"[Photoshop S3] Generated pre-signed URL (valid for 24 hours)")

    return presigned_url


async def upload_image_to_s3(image: torch.Tensor, total_pixels: int = 4096 * 4096) -> str:
    """
    Upload an image tensor to AWS S3 and return a pre-signed GET URL.

    Args:
        image: Image tensor to upload
        total_pixels: Maximum total pixels for the image

    Returns:
        Pre-signed GET URL for the uploaded image (valid for 24 hours)

    Raises:
        ValueError: If AWS credentials are not configured
        Exception: If upload fails
    """
    # Load AWS credentials from config
    config = _load_firefly_config()

    aws_access_key_id = config.get("aws_access_key_id")
    aws_secret_access_key = config.get("aws_secret_access_key")
    aws_region = config.get("aws_region", "us-east-1")
    aws_bucket = config.get("aws_bucket")

    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError(
            "AWS credentials not configured. Please add aws_access_key_id and "
            "aws_secret_access_key to firefly_config.json"
        )

    if not aws_bucket:
        raise ValueError(
            "AWS bucket not configured. Please add aws_bucket to firefly_config.json"
        )

    try:
        # Convert tensor to PNG bytes
        image_bytes = tensor_to_bytesio(image, total_pixels=total_pixels)
        image_size = image_bytes.getbuffer().nbytes

        logging.info(f"[Photoshop S3] Uploading image ({image_size:,} bytes) to S3...")

        # Generate unique filename
        filename = f"photoshop-input-{uuid.uuid4()}.png"

        # Prepare AWS config
        aws_config = {
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
            'aws_region': aws_region,
            'aws_bucket': aws_bucket,
        }

        # Get image bytes data
        image_bytes.seek(0)
        image_bytes_data = image_bytes.read()

        # Run synchronous boto3 operations in executor to avoid blocking event loop
        loop = asyncio.get_running_loop()
        presigned_url = await loop.run_in_executor(
            None,
            _upload_to_s3_sync,
            image_bytes_data,
            aws_config,
            filename
        )

        return presigned_url

    except Exception as e:
        logging.error(f"[Photoshop S3] Upload failed: {str(e)}")
        raise Exception(f"Failed to upload image to S3: {str(e)}")


def _generate_presigned_put_url_sync(aws_config: dict, filename: str) -> str:
    """
    Generate a pre-signed PUT URL synchronously (to be run in executor).

    Args:
        aws_config: Dict with AWS credentials and config
        filename: S3 object key/filename

    Returns:
        Pre-signed PUT URL for uploading to S3
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise ImportError(
            "boto3 is required for AWS S3 operations. Install it with: pip install boto3"
        )

    # Create S3 client with regional endpoint
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_config['aws_access_key_id'],
        aws_secret_access_key=aws_config['aws_secret_access_key'],
        region_name=aws_config['aws_region'],
        config=Config(
            signature_version='s3v4',
            s3={'addressing_style': 'path'}
        )
    )

    # Generate pre-signed PUT URL (24 hours)
    presigned_url = s3_client.generate_presigned_url(
        'put_object',
        Params={
            'Bucket': aws_config['aws_bucket'],
            'Key': filename,
        },
        ExpiresIn=86400  # 24 hours
    )

    logging.info(f"[Photoshop S3] Generated pre-signed PUT URL (valid for 24 hours)")
    logging.info(f"[Photoshop S3] Output will be stored at: s3://{aws_config['aws_bucket']}/{filename}")

    return presigned_url


async def generate_output_presigned_url(file_extension: str = "png") -> Tuple[str, str]:
    """
    Generate a pre-signed PUT URL for Adobe Photoshop API to upload output.

    This is used for endpoints like photoshopActions and actionJSON that require
    a writable URL for the output. Adobe will upload the processed result to this URL.

    Args:
        file_extension: File extension for the output (e.g., "png", "psd", "jpg")

    Returns:
        Tuple of (Pre-signed PUT URL for Adobe to upload result, S3 object key/filename)

    Raises:
        ValueError: If AWS credentials are not configured
        Exception: If URL generation fails
    """
    # Load AWS credentials from config
    config = _load_firefly_config()

    aws_access_key_id = config.get("aws_access_key_id")
    aws_secret_access_key = config.get("aws_secret_access_key")
    aws_region = config.get("aws_region", "us-east-1")
    aws_bucket = config.get("aws_bucket")

    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError(
            "AWS credentials not configured. Please add aws_access_key_id and "
            "aws_secret_access_key to firefly_config.json"
        )

    if not aws_bucket:
        raise ValueError(
            "AWS bucket not configured. Please add aws_bucket to firefly_config.json"
        )

    try:
        # Generate unique filename for output
        filename = f"photoshop-output-{uuid.uuid4()}.{file_extension}"

        logging.info(f"[Photoshop S3] Generating pre-signed PUT URL for output...")

        # Prepare AWS config
        aws_config = {
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
            'aws_region': aws_region,
            'aws_bucket': aws_bucket,
        }

        # Run synchronous boto3 operations in executor to avoid blocking event loop
        loop = asyncio.get_running_loop()
        presigned_url = await loop.run_in_executor(
            None,
            _generate_presigned_put_url_sync,
            aws_config,
            filename
        )

        return presigned_url, filename

    except Exception as e:
        logging.error(f"[Photoshop S3] Failed to generate pre-signed PUT URL: {str(e)}")
        raise Exception(f"Failed to generate output URL for S3: {str(e)}")


def _generate_presigned_get_url_sync(aws_config: dict, filename: str) -> str:
    """
    Generate a pre-signed GET URL synchronously (to be run in executor).

    Args:
        aws_config: Dict with AWS credentials and config
        filename: S3 object key/filename

    Returns:
        Pre-signed GET URL for downloading from S3
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise ImportError(
            "boto3 is required for AWS S3 operations. Install it with: pip install boto3"
        )

    # Create S3 client with regional endpoint
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_config['aws_access_key_id'],
        aws_secret_access_key=aws_config['aws_secret_access_key'],
        region_name=aws_config['aws_region'],
        config=Config(
            signature_version='s3v4',
            s3={'addressing_style': 'path'}
        )
    )

    # Generate pre-signed GET URL (24 hours)
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': aws_config['aws_bucket'],
            'Key': filename,
        },
        ExpiresIn=86400  # 24 hours
    )

    logging.info(f"[Photoshop S3] Generated pre-signed GET URL for download (valid for 24 hours)")

    return presigned_url


async def generate_download_url(filename: str) -> str:
    """
    Generate a pre-signed GET URL to download a file from S3.

    Args:
        filename: S3 object key/filename to download

    Returns:
        Pre-signed GET URL for downloading (valid for 24 hours)

    Raises:
        ValueError: If AWS credentials are not configured
        Exception: If URL generation fails
    """
    # Load AWS credentials from config
    config = _load_firefly_config()

    aws_access_key_id = config.get("aws_access_key_id")
    aws_secret_access_key = config.get("aws_secret_access_key")
    aws_region = config.get("aws_region", "us-east-1")
    aws_bucket = config.get("aws_bucket")

    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError(
            "AWS credentials not configured. Please add aws_access_key_id and "
            "aws_secret_access_key to firefly_config.json"
        )

    if not aws_bucket:
        raise ValueError(
            "AWS bucket not configured. Please add aws_bucket to firefly_config.json"
        )

    try:
        # Prepare AWS config
        aws_config = {
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
            'aws_region': aws_region,
            'aws_bucket': aws_bucket,
        }

        # Run synchronous boto3 operations in executor to avoid blocking event loop
        loop = asyncio.get_running_loop()
        presigned_url = await loop.run_in_executor(
            None,
            _generate_presigned_get_url_sync,
            aws_config,
            filename
        )

        return presigned_url

    except Exception as e:
        logging.error(f"[Photoshop S3] Failed to generate pre-signed GET URL: {str(e)}")
        raise Exception(f"Failed to generate download URL for S3: {str(e)}")


def _upload_file_to_s3_sync(file_path: str, aws_config: dict, filename: str, content_type: str) -> str:
    """
    Synchronous file upload to S3 (to be run in executor).

    Args:
        file_path: Local file path to upload
        aws_config: Dict with AWS credentials and config
        filename: S3 object key/filename
        content_type: MIME type of the file

    Returns:
        Pre-signed GET URL for the uploaded file
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise ImportError(
            "boto3 is required for AWS S3 uploads. Install it with: pip install boto3"
        )

    # Create S3 client with regional endpoint
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_config['aws_access_key_id'],
        aws_secret_access_key=aws_config['aws_secret_access_key'],
        region_name=aws_config['aws_region'],
        config=Config(
            signature_version='s3v4',
            s3={'addressing_style': 'path'}
        )
    )

    # Upload file to S3
    with open(file_path, 'rb') as f:
        s3_client.put_object(
            Bucket=aws_config['aws_bucket'],
            Key=filename,
            Body=f,
            ContentType=content_type
        )

    logging.info(f"[Photoshop S3] Upload successful: s3://{aws_config['aws_bucket']}/{filename}")

    # Generate pre-signed GET URL (24 hours)
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': aws_config['aws_bucket'], 'Key': filename},
        ExpiresIn=86400  # 24 hours
    )

    logging.info(f"[Photoshop S3] Generated pre-signed URL (valid for 24 hours)")

    return presigned_url


async def upload_file_to_s3(file_path: str, content_type: str = "application/octet-stream") -> str:
    """
    Upload a file from disk to AWS S3 and return a pre-signed GET URL.

    Args:
        file_path: Local file path to upload
        content_type: MIME type of the file (e.g., "image/vnd.adobe.photoshop" for PSD)

    Returns:
        Pre-signed GET URL for the uploaded file (valid for 24 hours)

    Raises:
        ValueError: If AWS credentials are not configured or file doesn't exist
        Exception: If upload fails
    """
    import os

    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    # Load AWS credentials from config
    config = _load_firefly_config()

    aws_access_key_id = config.get("aws_access_key_id")
    aws_secret_access_key = config.get("aws_secret_access_key")
    aws_region = config.get("aws_region", "us-east-1")
    aws_bucket = config.get("aws_bucket")

    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError(
            "AWS credentials not configured. Please add aws_access_key_id and "
            "aws_secret_access_key to firefly_config.json"
        )

    if not aws_bucket:
        raise ValueError(
            "AWS bucket not configured. Please add aws_bucket to firefly_config.json"
        )

    try:
        # Get file size
        file_size = os.path.getsize(file_path)
        file_ext = os.path.splitext(file_path)[1]

        logging.info(f"[Photoshop S3] Uploading file ({file_size:,} bytes) to S3...")

        # Generate unique filename
        filename = f"photoshop-input-{uuid.uuid4()}{file_ext}"

        # Prepare AWS config
        aws_config = {
            'aws_access_key_id': aws_access_key_id,
            'aws_secret_access_key': aws_secret_access_key,
            'aws_region': aws_region,
            'aws_bucket': aws_bucket,
        }

        # Run synchronous boto3 operations in executor to avoid blocking event loop
        loop = asyncio.get_running_loop()
        presigned_url = await loop.run_in_executor(
            None,
            _upload_file_to_s3_sync,
            file_path,
            aws_config,
            filename,
            content_type
        )

        return presigned_url

    except Exception as e:
        logging.error(f"[Photoshop S3] Upload failed: {str(e)}")
        raise Exception(f"Failed to upload file to S3: {str(e)}")
