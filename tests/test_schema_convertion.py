from aether.utils.schema_utils import *


def test_schema_convertion():
    """Test schema convertion"""
    schema = {
        "image": (str, "*.png"),
        "bbox": (
            [
                {
                    "x": (int, ...),
                    "y": (int, ...),
                    "w": (int, ...),
                    "h": (int, ...),
                    "label": (str, ...),
                }
            ],
            ...,
        ),
    }
    json_str = schema_to_json(schema)

    json_obj = json.loads(json_str)

    assert json_obj["image"]["default"] == "*.png"
    assert schema == json_to_schema(json_str)
