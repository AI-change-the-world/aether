from aether.api.request import AetherRequest, RegisterModelRequest
from aether.call.client import Client


def test_register_model():
    client = Client(auto_dispose=True)
    r: RegisterModelRequest = RegisterModelRequest(
        model_name="test_model",
        model_type="local_llm",
        model_path="",
        base_url="localhost:9997",
        api_key="sk-",
        request_definition={},
        response_definition={},
    )
    ar = AetherRequest(task="register_model", extra=r)
    client.call(ar)
