from typing import Literal, Optional, Union

from pydantic import BaseModel

from aether.api.tool_type import ToolType

global_activated_tool = {}


class BaseToolConfig(BaseModel):
    tool_type: ToolType
    description: Optional[str] = None


# ---------- OpenAI ----------


class OpenAIChatConfig(BaseToolConfig):
    tool_type: Literal[ToolType.OPENAI_CHAT]
    api_key: str
    base_url: Optional[str] = "https://api.openai.com/v1"
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048


class OpenAIEmbeddingConfig(BaseToolConfig):
    tool_type: Literal[ToolType.OPENAI_EMBEDDING]
    api_key: str
    base_url: Optional[str] = "https://api.openai.com/v1"
    model_name: str = "text-embedding-ada-002"


# ---------- Local OpenAI (兼容 OpenAI 接口) ----------


class LocalOpenAIConfig(BaseToolConfig):
    tool_type: Literal[ToolType.LOCAL_OPENAI]
    base_url: str
    model_name: str = "local-llm"
    temperature: float = 0.7
    max_tokens: int = 2048


# ---------- HuggingFace ----------


class HFClassificationConfig(BaseToolConfig):
    tool_type: Literal[ToolType.HF_CLS]
    model_path: str  # e.g., "distilbert-base-uncased" or local dir
    task: str = "text-classification"


class HFQAConfig(BaseToolConfig):
    tool_type: Literal[ToolType.HF_QA]
    model_path: str  # e.g., "deepset/roberta-base-squad2"
    context_max_length: Optional[int] = 512


class HFStableDiffusionConfig(BaseToolConfig):
    tool_type: Literal[ToolType.HF_SD]
    model_path: str
    scheduler: Optional[str] = "DDIM"
    use_safety_checker: bool = False


class HFGroundingDINOConfig(BaseToolConfig):
    tool_type: Literal[ToolType.HF_GROUNDING_DINO]
    model_path: str


# ---------- YOLO ----------


class YOLOConfig(BaseToolConfig):
    tool_type: Union[Literal[ToolType.YOLO_ULTRA], Literal[ToolType.YOLO_PT]]
    model_path: str  # e.g., "yolov8n.pt"
    device: Optional[str] = "cpu"
    # conf_thres: Optional[float] = 0.25


# ---------- Torch 本地模型 ----------


class TorchGANConfig(BaseToolConfig):
    tool_type: Literal[ToolType.TORCH_GAN]
    model_path: str
    z_dim: int = 100


class TorchClassificationConfig(BaseToolConfig):
    tool_type: Literal[ToolType.TORCH_CLS]
    model_path: str
    label_file: Optional[str] = None


class TorchSegmentationConfig(BaseToolConfig):
    tool_type: Literal[ToolType.TORCH_SEG]
    model_path: str
    input_size: Optional[int] = 512


# ---------- ONNX ----------


class ONNXModelConfig(BaseToolConfig):
    tool_type: Literal[ToolType.ONNX_MODEL]
    model_path: str
    input_names: Optional[list[str]] = None
    output_names: Optional[list[str]] = None
