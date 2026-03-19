"""
ComfyUI node for Adobe Substance 3D Describe endpoint.
POST /v1/scenes/describe
"""

from __future__ import annotations
import json
import logging

from .substance3d_api import DescribeRequest, MountedSource, MountedSourceURL
from .substance3d_client import submit_and_poll_s3d
from server import PromptServer

logger = logging.getLogger(__name__)


class Substance3DDescribeNode:
    """Describe a 3D model/scene using Adobe Substance 3D API."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_url": ("STRING", {"forceInput": True, "tooltip": "URL of the 3D model or scene file to analyze"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("description_json", "num_triangles", "num_vertices", "num_meshes", "camera_names", "material_names", "debug_log")
    FUNCTION = "describe"
    CATEGORY = "api node/Substance 3D"
    API_NODE = True

    async def describe(self, model_url, unique_id=None):
        debug_lines = []
        debug_lines.append("=== Substance 3D Describe ===")
        debug_lines.append(f"Model URL: {model_url[:120]}...")

        try:
            # ── Build sources list ────────────────────────────────────────
            sources = [
                MountedSource(
                    url=MountedSourceURL(url=model_url),
                    mountPoint="/",
                )
            ]

            # ── Build the request ─────────────────────────────────────────
            request = DescribeRequest(sources=sources)

            debug_lines.append("Submitting describe job...")

            # ── Submit and poll ───────────────────────────────────────────
            job_response = await submit_and_poll_s3d(
                endpoint_path="/v1/scenes/describe",
                request_model=DescribeRequest,
                request_data=request,
                node_id=unique_id,
                estimated_duration=30.0,
            )

            debug_lines.append(f"Job ID: {job_response.id}")
            debug_lines.append(f"Job status: {job_response.status}")

            # ── Extract stats from result ─────────────────────────────────
            result = job_response.result or {}
            stats = result.get("stats", {}) or {}

            num_triangles = stats.get("numTriangles", 0) or 0
            num_vertices = stats.get("numVertices", 0) or 0
            num_meshes = stats.get("numMeshes", 0) or 0
            camera_names_list = stats.get("cameraNames", []) or []
            material_names_list = stats.get("materialNames", []) or []

            camera_names = ", ".join(camera_names_list)
            material_names = ", ".join(material_names_list)

            description_json = json.dumps(result, indent=2)

            debug_lines.append(f"Triangles: {num_triangles}")
            debug_lines.append(f"Vertices: {num_vertices}")
            debug_lines.append(f"Meshes: {num_meshes}")
            debug_lines.append(f"Cameras: {camera_names or '(none)'}")
            debug_lines.append(f"Materials: {material_names or '(none)'}")
            debug_lines.append("=== Done ===")

            debug_log = "\n".join(debug_lines)
            logger.info(debug_log)

            # Display success notification on the node
            if unique_id:
                PromptServer.instance.send_progress_text(
                    f"Describe complete - {num_meshes} meshes, {num_triangles} triangles", unique_id
                )

            return (description_json, num_triangles, num_vertices, num_meshes, camera_names, material_names, debug_log)

        except Exception as e:
            debug_lines.append(f"ERROR: {str(e)}")
            debug_log = "\n".join(debug_lines)
            logger.error(debug_log)
            # Return error info instead of re-raising so node doesn't turn red
            return (json.dumps({"error": str(e)}), 0, 0, 0, "", "", debug_log)
