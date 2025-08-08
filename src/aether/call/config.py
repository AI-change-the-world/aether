from typing import Literal, Optional, Union

from pydantic import BaseModel

from aether.api.model_type import ModelType

global_activated_model = {}


class BaseModelConfig(BaseModel):
    model_type: ModelType
    description: Optional[str] = None


# ---------- OpenAI ----------


class OpenAIChatConfig(BaseModelConfig):
    model_type: Literal[ModelType.OPENAI_CHAT]
    api_key: str
    base_url: Optional[str] = "https://api.openai.com/v1"
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048


class OpenAIEmbeddingConfig(BaseModelConfig):
    model_type: Literal[ModelType.OPENAI_EMBEDDING]
    api_key: str
    base_url: Optional[str] = "https://api.openai.com/v1"
    model_name: str = "text-embedding-ada-002"


# ---------- Local OpenAI (兼容 OpenAI 接口) ----------


class LocalOpenAIConfig(BaseModelConfig):
    model_type: Literal[ModelType.LOCAL_OPENAI]
    base_url: str
    model_name: str = "local-llm"
    temperature: float = 0.7
    max_tokens: int = 2048


# ---------- HuggingFace ----------


class HFClassificationConfig(BaseModelConfig):
    model_type: Literal[ModelType.HF_CLS]
    model_path: str  # e.g., "distilbert-base-uncased" or local dir
    task: str = "text-classification"


class HFQAConfig(BaseModelConfig):
    model_type: Literal[ModelType.HF_QA]
    model_path: str  # e.g., "deepset/roberta-base-squad2"
    context_max_length: Optional[int] = 512


class HFStableDiffusionConfig(BaseModelConfig):
    model_type: Literal[ModelType.HF_SD]
    model_path: str
    scheduler: Optional[str] = "DDIM"
    use_safety_checker: bool = False


class HFGroundingDINOConfig(BaseModelConfig):
    model_type: Literal[ModelType.HF_GROUNDING_DINO]
    model_path: str


# ---------- YOLO ----------


class YOLOConfig(BaseModelConfig):
    model_type: Union[Literal[ModelType.YOLO_ULTRA], Literal[ModelType.YOLO_PT]]
    model_path: str  # e.g., "yolov8n.pt"
    device: Optional[str] = "cpu"
    # conf_thres: Optional[float] = 0.25


# ---------- Torch 本地模型 ----------


class TorchGANConfig(BaseModelConfig):
    model_type: Literal[ModelType.TORCH_GAN]
    model_path: str
    z_dim: int = 100


class TorchClassificationConfig(BaseModelConfig):
    model_type: Literal[ModelType.TORCH_CLS]
    model_path: str
    label_file: Optional[str] = None


class TorchSegmentationConfig(BaseModelConfig):
    model_type: Literal[ModelType.TORCH_SEG]
    model_path: str
    input_size: Optional[int] = 512


# ---------- ONNX ----------


class ONNXModelConfig(BaseModelConfig):
    model_type: Literal[ModelType.ONNX_MODEL]
    model_path: str
    input_names: Optional[list[str]] = None
    output_names: Optional[list[str]] = None
