import logging
from .substance3d_api import ConvertRequest, MountedSource, MountedSourceURL
from .substance3d_client import submit_and_poll_s3d
from server import PromptServer

logger = logging.getLogger(__name__)


class Substance3DConvertNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_url": ("STRING", {"forceInput": True, "tooltip": "URL of the 3D model file to convert"}),
                "output_format": (["glb", "gltf", "fbx", "usdz", "usda", "usdc", "obj"], {"tooltip": "Target 3D file format for conversion"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_url", "debug_log")
    FUNCTION = "convert"
    CATEGORY = "api node/Substance 3D"
    API_NODE = True

    async def convert(self, model_url, output_format, unique_id=None):
        debug_lines = []

        # Build sources from model_url
        sources = [
            MountedSource(
                url=MountedSourceURL(url=model_url),
                mountPoint="/",
            )
        ]

        # Build the convert request
        request = ConvertRequest(
            format=output_format,
            sources=sources,
        )

        debug_lines.append(f"Submitting convert job: format={output_format}")
        logger.info(f"[S3D Convert] Submitting: format={output_format}")

        # Submit and poll
        try:
            result = await submit_and_poll_s3d(
                "/v1/scenes/convert",
                ConvertRequest,
                request,
                node_id=unique_id,
                estimated_duration=30.0,
            )
        except Exception as e:
            debug_lines.append(f"ERROR: {str(e)}")
            logger.error(f"[S3D Convert] Job failed: {e}")
            raise

        debug_lines.append(f"Job status: {result.status}")
        logger.info(f"[S3D Convert] Job completed: status={result.status}")

        # Extract output file URL from result.outputSpace.files[0].url
        output_url = ""
        if result.result:
            output_space = result.result.get("outputSpace", {})
            files = output_space.get("files", [])
            if files:
                output_url = files[0].get("url", "")
                debug_lines.append(f"Output file: {files[0].get('name', 'unknown')}")

        if not output_url:
            debug_lines.append("WARNING: No output file URL found in result")
            logger.warning(f"[S3D Convert] No output URL in result: {result.result}")

        # Display success notification on the node
        if unique_id:
            PromptServer.instance.send_progress_text(
                f"Convert complete -> {output_format}", unique_id
            )

        debug_log = "\n".join(debug_lines)
        return (output_url, debug_log)
