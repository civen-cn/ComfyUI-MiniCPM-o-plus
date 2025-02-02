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
            
            # 使用 ComfyUI 的缓存目录
            comfyui_root = Path(folder_paths.models_dir).parent
            cache_dir = comfyui_root / ".cache/huggingface/modules/transformers_modules" / model_name
            
            # 如果缓存目录存在，先删除它
            if cache_dir.exists():
                print("清理现有缓存...")
                shutil.rmtree(str(cache_dir))
            
            # 只在缓存不存在时创建和复制文件
            if not cache_dir.exists():
                # print(f"创建缓存目录：{cache_dir}")
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                # 创建 __init__.py 文件
                init_file = cache_dir / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
                    # print(f"创建 __init__.py: {init_file}")
                
                # 需要复制的文件列表
                files_to_copy = [
                    "image_processing_minicpmv.py",
                    "configuration_minicpm.py",
                    "modeling_minicpmo.py",
                    "processing_minicpmo.py",
                    "modeling_navit_siglip.py",
                    "resampler.py",
                    "utils.py"
                ]
                
                # 复制文件到缓存目录
                copied_files = []
                missing_files = []
                for file in files_to_copy:
                    source = model_path / file
                    target = cache_dir / file
                    if source.exists():
                        # print(f"复制文件到缓存: {file}")
                        shutil.copy2(str(source), str(target))
                        copied_files.append(file)
                    else:
                        print(f"警告：找不到源文件: {file}")
                        missing_files.append(file)
                
                # print(f"\n成功复制的文件: {len(copied_files)}")
                for file in copied_files:
                    ''
                    # print(f"- {file}")
                
                if missing_files:
                    print(f"\n缺失的文件: {len(missing_files)}")
                    for file in missing_files:
                        print(f"- {file}")
            else:
                ''
                # print(f"使用现有缓存：{cache_dir}")
            
            # 设置环境变量，确保 transformers 使用正确的缓存目录
            os.environ["TRANSFORMERS_CACHE"] = str(comfyui_root / ".cache/huggingface")
            os.environ["HF_HOME"] = str(comfyui_root / ".cache/huggingface")
            os.environ["HF_MODULES_CACHE"] = str(comfyui_root / ".cache/huggingface/modules")
            
            print("\n开始加载模型...")
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                attn_implementation=attn_implementation,  # 添加注意力实现参数
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