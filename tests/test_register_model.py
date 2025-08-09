from aether import init_db
from aether.api.input import Input
from aether.api.request import (
    AetherRequest,
    Execution,
    RegisterModelRequest,
    RequestMeta,
)
from aether.api.response import AetherResponse
from aether.call import FetchResultClient, RegisterModelClient


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


def test_register_model_async():
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
    ar = AetherRequest(
        task="register_model", extra=r, meta=RequestMeta(execution=Execution.ASYNC)
    )
    res = client.call(ar)
    assert isinstance(res, AetherResponse)

    import time

    time.sleep(3)
    task_id = str(res.meta.task_id)
    f_client = FetchResultClient()
    fr = AetherRequest(task="fetch_result", input=Input(data=task_id))
    res = f_client.call(fr)
    assert (
        isinstance(res, AetherResponse)
        and res.success
        and res.output["output"]["status"] == 1
    )
