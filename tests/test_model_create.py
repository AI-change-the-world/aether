from aether.utils.object_match import is_object_match
from aether.utils.pydantic_utils import create_dynamic_model


def test_create_dynamic_model():
    json_str = """
    {
        "name": "John",
        "age": 30,
        "city": "New York"
    }
    """
    model = create_dynamic_model(
        "DynamicModel", {"name": (str, "John"), "age": (int, 30)}
    )
    assert is_object_match(json_str, model)
