from .nodes.minicpm_o_loader import MiniCPMLoader
from .nodes.minicpm_o_image import MiniCPMInference
from .nodes.minicpm_o_image_analyzer import MiniCPMImageAnalyzer
from .nodes.minicpm_o_video import MiniCPMVideoInference

NODE_CLASS_MAPPINGS = {
    "Load MiniCPM Model": MiniCPMLoader,
    "MiniCPM Image Chat": MiniCPMInference,
    "MiniCPMImageAnalyzer": MiniCPMImageAnalyzer,
    "MiniCPM Video Chat": MiniCPMVideoInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load MiniCPM Model": "Load MiniCPM-o Model",
    "MiniCPM Image Chat": "MiniCPM-o image",
    "MiniCPMImageAnalyzer": "MiniCPM-o Image Analyzer",
    "MiniCPM Video Chat": "MiniCPM-o Video",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']