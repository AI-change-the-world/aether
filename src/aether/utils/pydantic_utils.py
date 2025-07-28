from typing import Any, Dict, Tuple, Type

from pydantic import BaseModel, Field, create_model


def create_dynamic_model(
    model_name: str, fields: Dict[str, Tuple[Type, Any]], aliases: Dict[str, str] = None
) -> Type[BaseModel]:
    """
    动态创建一个 Pydantic 模型类。

    参数:
        model_name: 模型名称
        fields: 字段定义，形式为 {字段名: (类型, 默认值)}
        aliases: 字段别名映射 {字段名: 别名}（可选）

    返回:
        新创建的 Pydantic 模型类
    """
    aliases = aliases or {}

    model_fields = {}
    for field_name, (field_type, default_value) in fields.items():
        if field_name in aliases:
            model_fields[field_name] = (
                field_type,
                Field(default=default_value, alias=aliases[field_name]),
            )
        else:
            model_fields[field_name] = (field_type, default_value)

    return create_model(model_name, **model_fields)
