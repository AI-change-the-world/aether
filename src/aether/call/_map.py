import json
import traceback
from typing import Dict, Type

from pydantic import ValidationError

from aether.call.base import BaseClient
from aether.call.config import *
from aether.call.fetch_result_client import FetchResultClient
from aether.call.get_tools_client import GetToolsClient
from aether.call.openai_client import OpenAIClient
from aether.call.register_model_client import RegisterModelClient
from aether.call.yolo.yolo_client import YoloClient
from aether.common.logger import logger
from aether.models.tool.tool import AetherTool

TOOL_CONFIG_MAP: Dict[str, Type[BaseToolConfig]] = {
    ToolType.OPENAI_CHAT: OpenAIChatConfig,
    ToolType.OPENAI_EMBEDDING: OpenAIEmbeddingConfig,
    ToolType.LOCAL_OPENAI: LocalOpenAIConfig,
    ToolType.HF_CLS: HFClassificationConfig,
    ToolType.HF_QA: HFQAConfig,
    ToolType.HF_GROUNDING_DINO: HFGroundingDINOConfig,
    ToolType.HF_SD: HFStableDiffusionConfig,
    ToolType.TORCH_CLS: TorchClassificationConfig,
    ToolType.TORCH_GAN: TorchGANConfig,
    ToolType.TORCH_SEG: TorchSegmentationConfig,
    ToolType.YOLO_PT: YOLOConfig,
    ToolType.YOLO_ULTRA: YOLOConfig,
    ToolType.ONNX_MODEL: ONNXModelConfig,
    ToolType.REGISTER_MODEL: BaseToolConfig,
    ToolType.FETCH_RESULT: BaseToolConfig,
    ToolType.GET_TOOLS: BaseToolConfig,
}


TOOL_CLIENT_MAP: Dict[str, Type[BaseClient]] = {
    ToolType.OPENAI_CHAT: OpenAIClient,
    ToolType.LOCAL_OPENAI: OpenAIClient,
    ToolType.YOLO_ULTRA: YoloClient,
    ToolType.YOLO_PT: YoloClient,
    ToolType.REGISTER_MODEL: RegisterModelClient,
    ToolType.FETCH_RESULT: FetchResultClient,
    ToolType.GET_TOOLS: GetToolsClient,
}


def create_client(tool_model: AetherTool):
    """
    根据 JSON 字符串动态创建配置和客户端实例。
    """
    try:
        tool_config = json.loads(tool_model.tool_config)
        tool_type = tool_config.get("tool_type", None)

        if tool_type is None:
            raise ValueError("tool_type is required")

        # 获取对应的 Config 类
        config_class = TOOL_CONFIG_MAP.get(tool_type)
        if not config_class:
            raise ValueError(f"Unknown tool_type: {tool_type}")

        # 使用 Pydantic 构造配置对象，自动验证和转换
        config_obj = config_class(**tool_config)

        # 获取对应的 Client 类
        client_class = TOOL_CLIENT_MAP.get(tool_type)
        if not client_class:
            raise ValueError(f"No client class found for tool_type: {tool_type}")

        setattr(config_obj, "tool_model", tool_model)

        # 实例化客户端
        client_instance = client_class(config_obj)

        return client_instance

    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        logger.error(f"Failed to create client: {e}")
        traceback.print_exc()
        return None
