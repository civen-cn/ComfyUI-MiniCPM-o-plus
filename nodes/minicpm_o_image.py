import torch
from PIL import Image
import numpy as np

class MiniCPMInference:
    """MiniCPM 推理节点"""
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "MiniCPM-o"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "tokenizer": ("TOKENIZER",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048}),
            }
        }

    def generate(self, model, tokenizer, image, prompt, temperature=0.7, top_p=0.9, max_new_tokens=512):
        """生成回答"""
        try:
            # print("正在处理输入图像...")
            # ComfyUI 的图像是 NHWC 格式的 tensor
            if len(image.shape) == 4:
                image = image[0]  # 取第一张图片
            
            # 转换为 numpy，并确保值在 0-255 范围内
            image_np = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            
            # 转换为 PIL Image
            pil_image = Image.fromarray(image_np, 'RGB')
            
            print(f"图像大小: {pil_image.size}")
            
            # 按照官方示例构建消息
            messages = [
                {
                    'role': 'user', 
                    'content': [pil_image, prompt]
                }
            ]
            
            # 生成回答
            # print("正在生成回答...")
            response = model.chat(
                msgs=messages,
                tokenizer=tokenizer,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens
            )
            
            # print(f"生成的回答: {response}")
            return (response,)
            
        except Exception as e:
            # print(f"发生错误: {str(e)}")
            # print(f"图像形状: {image.shape}")
            # print(f"图像类型: {type(image)}")
            raise e

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN") 