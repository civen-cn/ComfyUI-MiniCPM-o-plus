# ComfyUI-MiniCPM

这是一个 ComfyUI 的自定义节点，用于在 ComfyUI 中使用 MiniCPM 的多模态能力。

## 安装步骤

1. 克隆此仓库到你的 ComfyUI 的 custom_nodes 文件夹下：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CY-CHENYUE/ComfyUI-MiniCPM-o.git
```

2. 使用 ComfyUI 的 Python 安装依赖：

Windows 用户:
```bash
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
```


## 功能特点

- 支持图像和文本输入
- 可调节的生成参数：
  - Temperature
  - Top-p 采样
  - 最大生成长度
- 与 ComfyUI 工作流无缝集成

## 使用方法

安装完成后，你可以在 ComfyUI 的节点浏览器中找到 "MiniCPM-o" 类别。