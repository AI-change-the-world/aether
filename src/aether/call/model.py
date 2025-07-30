from typing import Optional

from pydantic import BaseModel, model_validator

from aether.api.model_type import ModelType

global_activated_model = {}


class ModelConfig(BaseModel):
    """
    模型配置类

    用于存储和验证模型配置信息的Pydantic模型

    Attributes:
        base_url (Optional[str]): 模型服务的基础URL地址
        api_key (Optional[str]): 访问模型服务所需的API密钥
        model_name (Optional[str]): 模型名称
        save_path (Optional[str]): 模型文件的本地保存路径
        model_type (ModelType): 模型类型，必须指定
    """

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    save_path: Optional[str] = None
    model_type: ModelType

    @model_validator(mode="before")
    def check_url_or_path(cls, values):
        """
        验证器函数，确保base_url和save_path至少有一个被指定

        Args:
            values: 包含所有字段值的字典

        Returns:
            values: 验证通过的字段值字典

        Raises:
            ValueError: 当base_url和save_path都为空时抛出异常
        """
        # 获取base_url和save_path的值
        a, b = values.get("base_url"), values.get("save_path")
        # 验证base_url和save_path不能同时为空
        if not a and not b:
            raise ValueError("参数base_url和参数save_path不能同时为空")
        return values
