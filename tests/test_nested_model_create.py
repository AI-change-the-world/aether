from aether.utils.object_match import is_object_match
from aether.utils.pydantic_utils import create_nested_dynamic_model


def test_create_nested_model():
    json_str = """
    {
        "image": "aaa.jpeg",
        "bbox": [
            {"x": 0, "y": 0, "w": 100, "h": 100, "label": "car"}
        ]
    }
    """
    model = create_nested_dynamic_model(
        "DynamicModel",
        {
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
        },
    )
    assert is_object_match(json_str, model)
