from aether import init_db
from aether.api.request import AetherRequest, RegisterModelRequest
from aether.api.response import AetherResponse
from aether.call import RegisterModelClient


def test_register_model():
    init_db()
    client = RegisterModelClient()
    r: RegisterModelRequest = RegisterModelRequest(
        model_name="test_model",
        model_type="local_openai",
        model_path="",
        base_url="localhost:9997",
        api_key="sk-",
        request_definition={},
        response_definition={},
    )
    ar = AetherRequest(task="register_model", extra=r)
    res = client.call(ar)
    assert isinstance(res, AetherResponse) and res.success
