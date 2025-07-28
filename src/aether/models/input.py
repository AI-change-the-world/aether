from enum import Enum

from pydantic import BaseModel, Field, field_validator


class DataSource(str, Enum):
    """
    数据源类型枚举类

    定义了三种数据源类型：
    - S3: Amazon S3存储桶
    - LOCAL: 本地文件系统
    - BASE64: Base64编码的字符串数据
    """

    S3 = "s3"
    LOCAL = "local"
    BASE64 = "base64"


class DataType(str, Enum):
    """
    数据类型枚举类

    定义了系统支持的各种数据类型，继承自str和Enum，
    使得枚举值既可以作为字符串使用，又具有枚举的特性

    枚举值:
        TEXT: 文本类型
        IMAGE: 图像类型
        VIDEO: 视频类型
        AUDIO: 音频类型
        AUTO: 自动识别类型
    """

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    AUTO = "auto"


class Input(BaseModel):
    """
    输入数据模型类，用于定义和验证输入数据的结构

    参数:
        data (str): 输入的数据内容
        data_type (DataType): 数据类型，使用Field定义别名为"data_type"
        data_source (DataSource): 数据来源，使用Field定义别名为"data_source"
    """

    data: str
    data_type: DataType = Field(..., alias="data_type")
    data_source: DataSource = Field(..., alias="data_source")

    @field_validator("data_type", pre=True)
    def case_insensitive(cls, v):
        if isinstance(v, str):
            v = v.lower()
        return v
