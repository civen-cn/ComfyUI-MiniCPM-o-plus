import torch
from PIL import Image
import numpy as np
import random

class MiniCPMImageAnalyzer:
    """MiniCPM Image Analyzer Node for analyzing theme, scene and style"""
    
    # 预设的分析提示词
    THEME_PROMPT = """
    Describe ONLY the main subject's physical appearance and attire. Focus strictly on:
    - Physical appearance (features, colors, proportions)
    - Clothing and accessories (exact details of what they're wearing)
    - Visible details of the subject itself

    ABSOLUTELY NO:
    - Background or environment
    - Lighting or effects
    - Art style or technique
    - Composition or layout
    - Atmosphere or mood
    - Any elements not directly part of the subject

    Example:
    A humanoid figure with long black hair and pointed fox-like ears, striking red eyes and sharp facial features, wearing an elaborate white dress with gold-trimmed edges, a prominent star-shaped golden brooch at the chest, and a dark blue cape draped across the shoulders, with triangular decorative elements along the garment edges

    CRITICAL:
    - Describe ONLY what is physically part of the subject
    - Stop at the subject's boundaries
    - One flowing description
    - No artistic interpretation
    """
    
    SCENE_PROMPT = """
    Focus ONLY on the environment and background elements in this image.
    Describe ONLY: setting, atmosphere, lighting, and environmental details.
    Ignore any characters/subjects and artistic style.
    Format your response as a concise scene description for image generation.
    """
    
    STYLE_PROMPT = """
    Identify ONLY the artistic style and visual technique.

    Consider:
    - Art style (e.g., Van Gogh, Ghibli, anime, paper-cut)
    - Color scheme
    - Technique (e.g., layered, brushwork, cel-shading)
    - Visual effect (e.g., soft, sharp, gradient)

    CRITICAL:
    - Output ONLY style and technique
    - NO content descriptions (no objects, patterns, or subjects)
    - Format: [art style] with [visual technique and color]
    - Keep it minimal and precise

    Examples:
    Paper-cut art with layered monochrome technique.
    Ghibli style with soft watercolor effects.
    Van Gogh style with expressive brushstrokes.
    Anime style with bold cel-shading.
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

    # 提示词模板
    COMBINE_PROMPT = """
    Create a single image generation prompt that MUST include all elements:

    Required Elements:
    1. Style: {style_analysis}
    2. Subject: {theme_analysis}
    3. Scene: {scene_analysis}
    4. User Element: {user_prompt}

    Composition Rules:
    1. Start with style description
    2. For subject/scene integration:
       - If user element is an action/pose → blend with subject description
       - If user element is an object → integrate with subject's interaction
       - If user element is environmental → incorporate into scene
       - If user element is visual → merge with style

    Structure: [style], featuring [subject with integrated user element] in [scene]

    CRITICAL:
    - User element MUST be included
    - Place user element where it fits most naturally
    - Keep style at the beginning
    - Make one continuous prompt
    - NO explanations or markers

    Example with user element "eating ice cream":
    Minimalist paper-cut art with layered monochrome technique, featuring a tall figure with fox ears and flowing dress elegantly eating ice cream amid tranquil waters and geometric stone formations

    IMPORTANT:
    - Output ONLY the final prompt
    - Ensure user element is clearly present
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
            },
            "optional": {
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
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
               seed, temperature=0.7, top_p=0.9, max_new_tokens=512, user_prompt=""):
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

            # 直接在模板中包含用户提示词
            combine_prompt = self.COMBINE_PROMPT.format(
                theme_analysis=theme_analysis.strip(),
                scene_analysis=scene_analysis.strip(),
                style_analysis=style_analysis.strip(),
                user_prompt=user_prompt.strip() if user_prompt.strip() else "none"
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

            # # 去掉结果中的引号
            # combined_prompt = combined_prompt.strip().strip('"').strip("'")

            return (theme_analysis, scene_analysis, style_analysis, combined_prompt)

        except Exception as e:
            raise e

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        # 在固定模式下，返回seed值作为唯一标识
        # 这样只有当seed值改变时，节点才会重新运行
        return seed 