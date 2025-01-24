# ComfyUI-MiniCPM

这是一个 ComfyUI 的自定义节点，用于在 ComfyUI 中使用 MiniCPM-o 的多模态能力。

还在增加这个节点的功能，希望能用这个模型实时音视频能力在ComfyUI里做一些有趣实用的事情

## 安装步骤

### 方法一：使用 ComfyUI Manager（推荐）

1. 在 ComfyUI 中安装 [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. 打开 ComfyUI，点击右上角的 "Manager" 标签
3. 在搜索框中输入 "MiniCPM-o"
4. 点击安装按钮完成安装

### 方法二：手动安装

1. 克隆此仓库到你的 ComfyUI 的 custom_nodes 文件夹下：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CY-CHENYUE/ComfyUI-MiniCPM-o.git
```

2. 使用 ComfyUI 的 Python 安装依赖：

```bash
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

## 功能特点

- 支持图像和文本输入
- 支持中英双语对话
- 可调节的生成参数：
  - Temperature（温度）
  - Top-p 采样
  - 最大生成长度
- 与 ComfyUI 工作流无缝集成

## 使用方法

安装完成后，你可以在 ComfyUI 的节点浏览器中找到 "MiniCPM-o" 类别。将节点添加到你的工作流中即可开始使用。

