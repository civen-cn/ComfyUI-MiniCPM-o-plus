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
        return {
            "required": {
                "model_name": (["MiniCPM-o-2_6"],),  # 直接使用固定的模型文件夹名称
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "init_vision": ("BOOLEAN", {"default": True}),  # 是否启用视觉功能
                "init_audio": ("BOOLEAN", {"default": False}),  # 是否启用音频功能
                "init_tts": ("BOOLEAN", {"default": False}),    # 是否启用语音合成功能
            }
        }
    
    def load_model(self, model_name, device, attn_implementation="sdpa", init_vision=True, init_audio=False, init_tts=False):
        """加载模型和tokenizer"""
        try:
            model_path = Path(folder_paths.models_dir) / "MiniCPM" / model_name
            if not model_path.exists():
                raise ValueError(f"本地模型未找到：{model_path}。请将模型文件放置在 ComfyUI/models/MiniCPM/MiniCPM-o-2_6 文件夹中。")
            
            print(f"正在加载模型：{model_path}")
            
            # 检测实际的 transformers 缓存目录
            from transformers.utils import TRANSFORMERS_CACHE
            actual_cache = Path(TRANSFORMERS_CACHE)
            modules_cache = actual_cache.parent / "modules/transformers_modules" / model_name
            
            # 特别处理 image_processing_minicpmv.py 文件
            source_file = model_path / "image_processing_minicpmv.py"
            target_file = modules_cache / "image_processing_minicpmv.py"
            
            if source_file.exists():
                try:
                    # 确保目标目录存在
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    # 复制文件
                    shutil.copy2(str(source_file), str(target_file))
                except Exception as copy_error:
                    # 如果复制失败，尝试使用备用路径
                    backup_cache = Path(folder_paths.models_dir).parent / ".cache/huggingface/modules/transformers_modules" / model_name
                    backup_target = backup_cache / "image_processing_minicpmv.py"
                    
                    backup_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(source_file), str(backup_target))
                    
                    # 更新环境变量指向新位置
                    os.environ["TRANSFORMERS_CACHE"] = str(backup_cache.parent.parent)
                    os.environ["HF_HOME"] = str(backup_cache.parent.parent)
                    os.environ["HF_MODULES_CACHE"] = str(backup_cache.parent)
            
            print("\n开始加载模型...")
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map=device,
                init_vision=init_vision,
                init_audio=init_audio,
                init_tts=init_tts
            )
            
            print("正在加载分词器...")
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            
            return (model, tokenizer)
        
        except Exception as e:
            print(f"\n详细错误信息: {str(e)}")
            raise RuntimeError(f"加载模型时发生错误: {str(e)}") 