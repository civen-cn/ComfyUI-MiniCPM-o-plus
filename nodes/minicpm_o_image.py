import torch
from PIL import Image
import numpy as np

class MiniCPMInference:
    """MiniCPM 推理节点"""
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "MiniCPM-o"

    # 预设的提示词模板
    TEMPLATE_PROMPT = """Please provide a detailed description of the image according to the following requirements, ensuring that the description can be used to generate images highly similar to the original. Output as plain text without any markdown formatting, headers, or bullet points:

Scene Overview: Describe the overall scene, time of day, and weather conditions. Main Elements: List and describe in detail the main objects, including their appearance characteristics (shape, size, color, material, texture). For people, include details about age, gender, clothing, actions, and expressions. Layout and Composition: Describe the positional relationships of elements and their relative distances. Light and Color: Describe light sources, their direction and intensity, overall color tone and distribution. Style and Atmosphere: Describe the artistic style and overall atmosphere. Details and Supporting Elements: Describe secondary elements, including any text, logos, or symbols.

Please ensure the description is based on visible information in the image, avoiding speculation. The description should be detailed and accurate enough to generate images highly similar to the original. Write everything in a continuous paragraph format without section breaks or special formatting."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "tokenizer": ("TOKENIZER",),
                "image": ("IMAGE",),
                "prompt_mode": (["Use System Preset", "Use Custom Input"], {"default": "Use System Preset"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 666666666666666, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048}),
            }
        }

    def generate(self, model, tokenizer, image, prompt_mode, prompt, seed, temperature=0.7, top_p=0.9, max_new_tokens=512):
        """生成回答"""
        try:
            # 设置随机种子
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            # ComfyUI 的图像是 NHWC 格式的 tensor
            if len(image.shape) == 4:
                image = image[0]  # 取第一张图片
            
            # 转换为 numpy，并确保值在 0-255 范围内
            image_np = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            
            # 转换为 PIL Image
            pil_image = Image.fromarray(image_np, 'RGB')
            
            print(f"图像大小: {pil_image.size}")
            
            # 根据选择使用模板提示词或用户输入的提示词
            final_prompt = self.TEMPLATE_PROMPT if prompt_mode == "Use System Preset" else prompt
            
            # 按照官方示例构建消息
            messages = [
                {
                    'role': 'user', 
                    'content': [pil_image, final_prompt]
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
            
            return (response,)
            
        except Exception as e:
            raise e

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed 