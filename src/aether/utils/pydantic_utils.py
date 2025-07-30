from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_origin

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


def create_nested_dynamic_model(
    model_name: str,
    fields: Dict[str, Tuple[Union[Type, Dict, List], Any]],
    aliases: Optional[Dict[str, str]] = None,
) -> Type[BaseModel]:
    aliases = aliases or {}
    model_fields = {}

    for field_name, (field_type, default_value) in fields.items():
        actual_type = None

        # 子模型：Dict 结构 -> 转换成模型
        if isinstance(field_type, dict):
            nested_model = create_nested_dynamic_model(
                model_name=f"{model_name}_{field_name.capitalize()}",
                fields=field_type,
            )
            actual_type = nested_model

        # 子模型：List 结构
        elif isinstance(field_type, list) and len(field_type) == 1:
            inner = field_type[0]
            if isinstance(inner, dict):
                inner_model = create_nested_dynamic_model(
                    model_name=f"{model_name}_{field_name.capitalize()}Item",
                    fields=inner,
                )
                actual_type = List[inner_model]
            else:
                actual_type = List[inner]

        # 类型提示 List[X] / Optional[X] / Dict[str, X] 的结构
        elif isinstance(field_type, type) or get_origin(field_type) in [
            list,
            dict,
            Union,
        ]:
            actual_type = field_type

        else:
            raise ValueError(f"Unsupported field type: {field_type}")

        # 构造字段
        if field_name in aliases:
            model_fields[field_name] = (
                actual_type,
                Field(default=default_value, alias=aliases[field_name]),
            )
        else:
            model_fields[field_name] = (actual_type, default_value)

    return create_model(model_name, **model_fields)
