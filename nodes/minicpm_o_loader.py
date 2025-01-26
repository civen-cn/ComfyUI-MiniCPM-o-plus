import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import folder_paths
import shutil
import sys

class MiniCPMLoader:
    """MiniCPM 模型加载节点"""
    
    RETURN_TYPES = ("MODEL", "TOKENIZER")
    RETURN_NAMES = ("model", "tokenizer")
    FUNCTION = "load_model"
    CATEGORY = "MiniCPM-o"

    @classmethod
    def INPUT_TYPES(s):
        models_path = Path(folder_paths.models_dir) / "MiniCPM"
        models_path.mkdir(parents=True, exist_ok=True)
        
        model_list = []
        if models_path.exists():
            for item in models_path.iterdir():
                if item.is_dir():
                    model_list.append(item.name)
        
        if not model_list:
            model_list = ["none"]
            print(f"警告：在 {models_path} 目录下没有找到模型")
        
        return {
            "required": {
                "model_name": (model_list, ),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "init_vision": ("BOOLEAN", {"default": True}),  # 是否启用视觉功能
                "init_audio": ("BOOLEAN", {"default": False}),  # 是否启用音频功能
                "init_tts": ("BOOLEAN", {"default": False}),    # 是否启用语音合成功能
            }
        }
    
    def load_model(self, model_name, device, init_vision=True, init_audio=False, init_tts=False):
        """加载模型和tokenizer"""
        model_path = Path(folder_paths.models_dir) / "MiniCPM" / model_name
        
        try:
            print(f"正在加载模型：{model_path}")
            
            # 按照官方文档加载模型，添加 local_files_only=True 参数
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                local_files_only=True,  # 强制使用本地文件
                attn_implementation='sdpa',  # 使用 sdpa 实现
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
                init_vision=init_vision,   # 用户可选择是否启用视觉功能
                init_audio=init_audio,     # 用户可选择是否启用音频功能
                init_tts=init_tts          # 用户可选择是否启用语音合成功能
            )
            
            print("正在加载分词器...")
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                local_files_only=True  # 分词器也强制使用本地文件
            )
            
            return (model, tokenizer)
            
        except Exception as e:
            print(f"详细错误信息: {str(e)}")
            raise RuntimeError(f"加载模型时发生错误: {str(e)}") 