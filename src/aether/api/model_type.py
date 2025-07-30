from enum import Enum


class ModelType(str, Enum):
    """
    模型类型枚举类

    定义了系统支持的模型类型常量
    """

    OPENAI = "openai"
    YOLO = "yolo"
    GAN = "gan"
    GROUNDING_DINO = "grounding_dino"  # support huggingface model now
    LOCAL_LLM = "local_llm"
    LOCAL_MLLM = "local_mllm"
