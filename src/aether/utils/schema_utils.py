import json
from typing import Any, Dict, List, Tuple

TYPE_MAP = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "object": dict,
    "string": str,
    "boolean": bool,
}

REVERSE_TYPE_MAP = {v: k for k, v in TYPE_MAP.items()}


# ===== 将 schema 转为 JSON 可序列化结构 =====
def schema_to_json(schema: Dict[str, Tuple[Any, Any]]) -> str:
    def serialize(field_type: Any, default: Any) -> Dict[str, Any]:
        if isinstance(field_type, dict):
            return {
                "type": "object",
                "default": "..." if default is ... else default,
                "fields": {k: serialize(v[0], v[1]) for k, v in field_type.items()},
            }

        elif isinstance(field_type, list) and len(field_type) == 1:
            inner = field_type[0]
            if isinstance(inner, dict):
                return {
                    "type": "list",
                    "default": "..." if default is ... else default,
                    "items": {
                        "type": "object",
                        "fields": {k: serialize(v[0], v[1]) for k, v in inner.items()},
                    },
                }
            else:
                return {
                    "type": "list",
                    "default": "..." if default is ... else default,
                    "items": {
                        "type": REVERSE_TYPE_MAP.get(inner, str(inner)),
                        "default": "...",
                    },
                }

        else:
            return {
                "type": REVERSE_TYPE_MAP.get(field_type, str(field_type)),
                "default": "..." if default is ... else default,
            }

    result = {k: serialize(v[0], v[1]) for k, v in schema.items()}
    return json.dumps(result, indent=2)


# ===== 将 JSON 字符串还原为 schema 结构 =====
def json_to_schema(json_str: str) -> Dict[str, Tuple[Any, Any]]:
    def deserialize(obj: Dict[str, Any]) -> Tuple[Any, Any]:
        typ = obj["type"]
        default = obj.get("default", ...)

        if default == "...":
            default = ...

        if typ == "object":
            sub_fields = {k: deserialize(v) for k, v in obj["fields"].items()}
            return (sub_fields, default)

        elif typ == "list":
            item_type = obj["items"]
            if item_type["type"] == "object":
                sub_fields = {k: deserialize(v) for k, v in item_type["fields"].items()}
                return ([sub_fields], default)
            else:
                inner_type = TYPE_MAP.get(item_type["type"], str)
                return ([inner_type], default)

        else:
            return (TYPE_MAP.get(typ, str), default)

    data = json.loads(json_str)
    return {k: deserialize(v) for k, v in data.items()}
