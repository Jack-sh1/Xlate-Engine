# TransForge - 模块化多语种翻译引擎

TransForge 是一个基于 Transformer 架构和 Flask 框架构建的高性能、模块化多语种翻译系统。它支持直接翻译以及通过英语作为中转的跨语言翻译（Pivot Translation），并具备自动硬件加速检测功能。

## 🌟 核心特性

- **模块化架构**：代码高度解耦，各组件（配置、模型管理、业务逻辑、API 处理）职责分明。
- **多语种支持**：通过 MarianMT 模型支持多种语言对，并具备中转翻译能力。
- **硬件加速**：自动检测并利用 MPS (Apple Silicon)、CUDA (NVIDIA GPU) 或 CPU 进行推理。
- **现代前端**：采用 SCSS 编写样式，具备自动编译功能。
- **生产就绪**：分离了应用定义与启动入口，支持生产环境部署。



## 🚀 快速开始

### 1. 克隆与环境配置

确保你已安装 Python 3.8+。

```bash
# 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行应用

```bash
python run.py
```

应用启动后，访问 `http://127.0.0.1:80` 即可进入翻译界面。

## 🛠️ 技术栈

- **后端**: Python, Flask
- **深度学习**: PyTorch, Hugging Face Transformers
- **模型**: MarianMT
- **前端**: HTML5, SCSS, JavaScript
- **工具**: libsass (SCSS 编译)

## 📝 许可证

MIT 
