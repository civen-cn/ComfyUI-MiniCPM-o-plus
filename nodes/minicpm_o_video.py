import torch
from PIL import Image
import numpy as np
import os
import tempfile
from pathlib import Path
from decord import VideoReader, cpu  # 使用 decord 替代 OpenCV
import comfy.model_management

class MiniCPMVideoInference:
    """MiniCPM 视频推理节点"""
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "MiniCPM-o"

    # 预设的提示词模板
    TEMPLATE_PROMPT = """请根据以下要求，详细描述这段视频，确保描述可以用于生成与原始视频高度相似的内容。输出为纯文本，不包含任何markdown格式、标题或项目符号：

场景概述：描述整体场景、时间和环境条件。
主要元素：列出并详细描述主要对象，包括它们的外观特征（形状、大小、颜色、材质、纹理）。对于人物，包括年龄、性别、服装、动作和表情的细节。
动作和变化：描述视频中的动作序列、场景变化和时间流动。
布局和构图：描述元素的位置关系及其相对距离。
光线和颜色：描述光源、光线方向和强度、整体色调和分布。
风格和氛围：描述艺术风格和整体氛围。
细节和辅助元素：描述次要元素，包括任何文本、标志或符号。

请确保描述基于视频中可见的信息，避免推测。描述应足够详细和准确，以便生成与原始视频高度相似的内容。以连续段落格式编写所有内容，不使用分节符或特殊格式。"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_CPM",),
                "tokenizer": ("TOKENIZER",),
                "video": ("PATH",),
                "prompt_mode": (["Use System Preset", "Use Custom Input"], {"default": "Use System Preset"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 666666666666666, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048}),
                "max_frames": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "sample_fps_divisor": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "max_slice_nums": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
            }
        }

    def generate(self, model, tokenizer, video, prompt_mode, prompt, seed, temperature=0.7, top_p=0.9, max_new_tokens=512, max_frames=16, sample_fps_divisor=1, max_slice_nums=2):
        """生成回答"""
        try:
            # 设置随机种子
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            
            # 确保模型在推理前已加载到GPU
            if hasattr(model, 'to') and torch.cuda.is_available():
                device = torch.device("cuda")
                # 计算推理所需的内存
                inference_memory = 1024 * 1024 * 1024  # 假设需要1GB内存用于推理
                # 确保有足够的显存用于推理
                comfy.model_management.free_memory(inference_memory, device)
            
            # 视频采样函数
            def uniform_sample(l, n):
                if n >= l:
                    return list(range(l))
                else:
                    return [i * l // n + l // (2 * n) for i in range(n)]
            
            # 加载视频
            print(f"加载视频: {video}")
            vr = VideoReader(video, ctx=cpu(0))
            
            # 获取视频信息
            total_frames = len(vr)
            fps = vr.get_avg_fps()
            duration = total_frames / fps
            
            print(f"视频信息: 总帧数={total_frames}, FPS={fps}, 时长={duration:.2f}秒")
            
            # 计算采样帧
            sample_interval = sample_fps_divisor
            sampled_indices = list(range(0, total_frames, sample_interval))
            
            # 如果采样后的帧数仍然太多，进一步均匀采样
            if len(sampled_indices) > max_frames * max_slice_nums:
                sampled_indices = uniform_sample(len(sampled_indices), max_frames * max_slice_nums)
                sampled_indices = [i * sample_interval for i in sampled_indices]
            
            # 确保不超过视频总帧数
            sampled_indices = [i for i in sampled_indices if i < total_frames]
            
            print(f"采样后帧数: {len(sampled_indices)}")
            
            # 将采样帧分成多个片段
            slice_size = min(max_frames, len(sampled_indices) // max_slice_nums + 1)
            frame_slices = [sampled_indices[i:i+slice_size] for i in range(0, len(sampled_indices), slice_size)]
            
            # 根据选择使用模板提示词或用户输入的提示词
            final_prompt = self.TEMPLATE_PROMPT if prompt_mode == "Use System Preset" else prompt
            
            all_responses = []
            
            for i, slice_indices in enumerate(frame_slices):
                print(f"处理视频片段 {i+1}/{len(frame_slices)}, 帧数: {len(slice_indices)}")
                
                # 获取该片段的帧
                frames = vr.get_batch(slice_indices).asnumpy()
                
                # 转换为PIL图像列表
                pil_frames = [Image.fromarray(frame) for frame in frames]
                
                # 构建消息
                messages = [
                    {
                        'role': 'user', 
                        'content': [*pil_frames, final_prompt]
                    }
                ]
                
                # 生成回答
                response = model.chat(
                    msgs=messages,
                    tokenizer=tokenizer,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens
                )
                
                all_responses.append(response)
            
            # 合并所有回答
            final_response = "\n\n".join(all_responses)
            
            # 推理完成后，可以考虑释放一些显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return (final_response,)
            
        except Exception as e:
            # 发生错误时确保清理资源
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(f"处理视频时发生错误: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed

