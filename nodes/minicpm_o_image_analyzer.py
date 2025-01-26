import torch
from PIL import Image
import numpy as np
import random

class MiniCPMImageAnalyzer:
    """MiniCPM Image Analyzer Node for analyzing theme, scene and style"""
    
    # 预设的分析提示词
    THEME_PROMPT = """
    Focus ONLY on the main character/subject in this image. 
    Describe ONLY: appearance, clothing, pose, and expression.
    Ignore the background, scene, and artistic style.
    Format your response as a concise character/subject description for image generation.
    """
    
    SCENE_PROMPT = """
    Focus ONLY on the environment and background elements in this image.
    Describe ONLY: setting, atmosphere, lighting, and environmental details.
    Ignore any characters/subjects and artistic style.
    Format your response as a concise scene description for image generation.
    """
    
    STYLE_PROMPT = """
    Analyze ONLY the pure artistic style elements of this image.
    Focus strictly on:
    - Art medium (digital, oil painting, watercolor, etc.)
    - Color scheme and palette
    - Lighting technique
    - Rendering style (realistic, cartoon, anime, etc.)
    - Visual effects and textures
    - Overall artistic approach

    DO NOT describe:
    - Any characters or subjects
    - Any specific objects
    - Scene content or background
    - Story elements or narrative
    - Compositional layout

    Format your response as a concise style description using only artistic terminology.
    Example format: "photorealistic digital art with vibrant neon colors, dramatic lighting, glossy textures"
    """

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("theme_analysis", "scene_analysis", "style_analysis", "combined_prompt")
    FUNCTION = "analyze"
    CATEGORY = "MiniCPM-o"

    PROMPT_TEMPLATE = """
    {theme_analysis}, 
    in a scene with {scene_analysis}, 
    rendered in {style_analysis}
    """

    # 在类中添加新的提示词模板
    COMBINE_PROMPT = """
    Combine these three separate aspects into a single image generation prompt:
    - Character/Subject: {theme_analysis}
    - Scene/Environment: {scene_analysis}
    - Artistic Style: {style_analysis}

    Create a structured prompt in this format:
    [character/subject description] in [scene/environment description], [artistic style description]
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "tokenizer": ("TOKENIZER",),
                "theme_image": ("IMAGE",),
                "scene_image": ("IMAGE",),
                "style_image": ("IMAGE",),  
                "seed": ("INT", {"default": 666666666666666, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048}),
            }
        }

    def process_image(self, image):
        """处理输入图像为PIL格式"""
        if len(image.shape) == 4:
            image = image[0]
        image_np = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(image_np, 'RGB')

    def get_analysis(self, model, tokenizer, image, prompt, temperature, top_p, max_new_tokens):
        """获取单张图片的分析结果"""
        messages = [
            {
                'role': 'user',
                'content': [image, prompt]
            }
        ]
        
        response = model.chat(
            msgs=messages,
            tokenizer=tokenizer,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens
        )
        return response

    def analyze(self, model, tokenizer, theme_image, scene_image, style_image, 
               seed, temperature=0.7, top_p=0.9, max_new_tokens=512):
        """分析图片并生成组合提示词"""
        try:
            # 设置随机种子
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            
            # 处理图片
            theme_pil = self.process_image(theme_image)
            scene_pil = self.process_image(scene_image)
            style_pil = self.process_image(style_image)

            # 获取各个方面的分析
            theme_analysis = self.get_analysis(model, tokenizer, theme_pil, self.THEME_PROMPT, 
                                            temperature, top_p, max_new_tokens)
            scene_analysis = self.get_analysis(model, tokenizer, scene_pil, self.SCENE_PROMPT, 
                                           temperature, top_p, max_new_tokens)
            style_analysis = self.get_analysis(model, tokenizer, style_pil, self.STYLE_PROMPT, 
                                           temperature, top_p, max_new_tokens)

            # 使用大模型融合提示词
            combine_prompt = self.COMBINE_PROMPT.format(
                theme_analysis=theme_analysis.strip(),
                scene_analysis=scene_analysis.strip(),
                style_analysis=style_analysis.strip()
            )
            
            messages = [
                {
                    'role': 'user',
                    'content': combine_prompt
                }
            ]
            
            combined_prompt = model.chat(
                msgs=messages,
                tokenizer=tokenizer,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens
            )

            return (theme_analysis, scene_analysis, style_analysis, combined_prompt)

        except Exception as e:
            raise e

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        # 在固定模式下，返回seed值作为唯一标识
        # 这样只有当seed值改变时，节点才会重新运行
        return seed 