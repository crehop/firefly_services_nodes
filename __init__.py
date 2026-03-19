"""
ComfyUI Firefly Services

Custom nodes for Adobe Firefly, Photoshop, Substance 3D, and utility operations.
"""

import logging
import os
import shutil
import folder_paths

# Auto-copy example input files to ComfyUI's input directory
_example_inputs_dir = os.path.join(os.path.dirname(__file__), "example_inputs")
if os.path.isdir(_example_inputs_dir):
    _input_dir = folder_paths.get_input_directory()
    for _fname in os.listdir(_example_inputs_dir):
        _src = os.path.join(_example_inputs_dir, _fname)
        _dst = os.path.join(_input_dir, _fname)
        if os.path.isfile(_src) and not os.path.exists(_dst):
            try:
                shutil.copy2(_src, _dst)
                logging.info(f"[Firefly Services] Copied example input: {_fname}")
            except Exception as _e:
                logging.warning(f"[Firefly Services] Could not copy {_fname}: {_e}")

from .Firefly import NODE_CLASS_MAPPINGS as FIREFLY_NODES
from .Firefly import NODE_DISPLAY_NAME_MAPPINGS as FIREFLY_NAMES
from .Utils import NODE_CLASS_MAPPINGS as UTILS_NODES
from .Utils import NODE_DISPLAY_NAME_MAPPINGS as UTILS_NAMES
from .Photoshop import NODE_CLASS_MAPPINGS as PS_NODES
from .Photoshop import NODE_DISPLAY_NAME_MAPPINGS as PS_NAMES

NODE_CLASS_MAPPINGS = {**FIREFLY_NODES, **UTILS_NODES, **PS_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**FIREFLY_NAMES, **UTILS_NAMES, **PS_NAMES}

try:
    from .Substance3D import NODE_CLASS_MAPPINGS as S3D_NODES
    from .Substance3D import NODE_DISPLAY_NAME_MAPPINGS as S3D_NAMES
    NODE_CLASS_MAPPINGS.update(S3D_NODES)
    NODE_DISPLAY_NAME_MAPPINGS.update(S3D_NAMES)
except Exception as e:
    logging.warning(f"[ComfyUI Firefly Services] Failed to load Substance 3D nodes: {e}")

try:
    from .InDesign import NODE_CLASS_MAPPINGS as INDESIGN_NODES
    from .InDesign import NODE_DISPLAY_NAME_MAPPINGS as INDESIGN_NAMES
    NODE_CLASS_MAPPINGS.update(INDESIGN_NODES)
    NODE_DISPLAY_NAME_MAPPINGS.update(INDESIGN_NAMES)
except Exception as e:
    logging.warning(f"[ComfyUI Firefly Services] Failed to load InDesign nodes: {e}")

try:
    from .VideoReframe import NODE_CLASS_MAPPINGS as VR_NODES
    from .VideoReframe import NODE_DISPLAY_NAME_MAPPINGS as VR_NAMES
    NODE_CLASS_MAPPINGS.update(VR_NODES)
    NODE_DISPLAY_NAME_MAPPINGS.update(VR_NAMES)
except Exception as e:
    logging.warning(f"[ComfyUI Firefly Services] Failed to load Video Reframe nodes: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
