from aether import init_db
from aether.api.input import Input
from aether.api.request import AetherRequest
from aether.api.response import AetherResponse
from aether.call import ActivatedToolRegistry, OpenAIClient


def test_chat_openai():
    init_db()
    client = OpenAIClient(auto_dispose=False)
    req = AetherRequest(
        task="chat",
        tool_id=1,
        input=Input(data="repeat what i said\nhello world"),
        extra={
            "temperature": 0.5,
            "history": [{"role": "system", "content": "you are a helpful assistant"}],
        },
    )
    res = client.call(req)
    assert (
        isinstance(res, AetherResponse)
        and res.success
        and "hello world" in str(res.output["output"])
    )
    print(res.output)
    assert ActivatedToolRegistry.instance().get(1) is not None

    client2 = OpenAIClient(auto_dispose=False)
    req = AetherRequest(
        task="chat",
        tool_id=1,
        input=Input(data="repeat what i said\ngood day"),
        extra={
            "temperature": 0.5,
            "history": [{"role": "system", "content": "you are a helpful assistant"}],
        },
    )
    res2 = client.call(req)
    assert (
        isinstance(res2, AetherResponse)
        and res2.success
        and "good day" in str(res.output["output"])
    )
