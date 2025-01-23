from .nodes.minicpm_o_loader import MiniCPMLoader
from .nodes.minicpm_o_image import MiniCPMInference

NODE_CLASS_MAPPINGS = {
    "Load MiniCPM Model": MiniCPMLoader,
    "MiniCPM Image Chat": MiniCPMInference
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load MiniCPM Model": "Load MiniCPM-o Model",
    "MiniCPM Image Chat": "MiniCPM-o image"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 