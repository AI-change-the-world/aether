import json
import traceback
from typing import Union

from pydantic import BaseModel, ValidationError

from aether.utils.pydantic_utils import create_nested_dynamic_model
from aether.utils.schema_utils import json_to_schema


def validate(input: str | dict, target: str) -> bool:
    try:
        schema = json_to_schema(target)
        model = create_nested_dynamic_model(model_name="Input", fields=schema)
        if isinstance(input, str):
            j = json.loads(input)
        else:
            j = input
        return is_object_match(j, model)
    except:
        traceback.print_exc()
        return False


def is_object_match(data: Union[dict, str], model_cls: type[BaseModel],) -> bool:
    """
    判断一个 JSON dict 是否能完整满足指定 BaseModel 的字段（支持别名）

    :param model_cls: 要验证的 Pydantic BaseModel 子类
    :param data: 输入的 JSON 数据（dict）
    :return: 如果可以创建该模型实例，则返回 True，否则 False
    """
    try:
        if isinstance(data, str):  # 如果输入是字符串，则尝试将字符串解析为 JSON
            data = json.loads(data)
        model_cls.model_validate(data)
        return True
    except ValidationError:
        return False


def is_full_object_match(data: Union[dict, str], model_cls: type[BaseModel]) -> bool:
    if isinstance(data, str):  # 如果输入是字符串，则尝试将字符串解析为 JSON
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return False

    if not isinstance(data, dict):
        return False

    model_fields = set(model_cls.model_fields.keys())
    json_fields = set(data.keys())

    return model_fields == json_fields


def is_full_object_match_with_alias(
    data: Union[dict, str], model_cls: type[BaseModel]
) -> bool:
    if isinstance(data, str):  # 如果输入是字符串，则尝试将字符串解析为 JSON
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return False

    if not isinstance(data, dict):
        return False

    # 使用 alias（别名）而非字段名
    model_aliases = set(
        field.alias or name for name, field in model_cls.model_fields.items()
    )
    json_keys = set(data.keys())

    return model_aliases == json_keys
