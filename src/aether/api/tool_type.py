from enum import Enum


class ToolType(str, Enum):
    """
    支持的模型类型枚举类

    模型类型按照使用来源或推理框架分类，例如：
    - OPENAI_* 表示基于 OpenAI 接口的模型
    - LOCAL_* 表示本地加载的模型（Torch 或其他）
    - HF_* 表示基于 HuggingFace Transformers/Diffusers 的模型
    - ULTRA_* 表示基于 Ultralytics YOLO 系列
    """

    # OpenAI 接口类模型
    OPENAI_CHAT = "openai_chat"  # gpt-3.5, gpt-4 等
    OPENAI_EMBEDDING = "openai_embedding"

    # 本地兼容 OpenAI 接口的大模型（如 LM Studio, OpenRouter, Local LLM）
    LOCAL_OPENAI = "local_openai"

    # HuggingFace 相关模型（支持transformers/diffusers）
    HF_CLS = "hf_cls"  # 文本分类模型
    HF_QA = "hf_qa"  # 问答
    HF_SD = "hf_stable_diffusion"  # 文生图
    HF_GROUNDING_DINO = "hf_grounding_dino"

    # YOLO 系列模型
    YOLO_ULTRA = "yolo_ultralytics"  # Ultralytics YOLO
    YOLO_PT = "yolo_torch"  # 自定义 YOLO 模型（如 yolov5.pt）

    # 自定义本地模型
    TORCH_GAN = "torch_gan"
    TORCH_CLS = "torch_classification"
    TORCH_SEG = "torch_segmentation"

    # ONNX 推理模型（可选）
    ONNX_MODEL = "onnx_model"
