from aether import init_db
from aether.api.input import Input
from aether.api.request import AetherRequest
from aether.api.response import AetherResponse
from aether.call import OpenAIClient


def test_chat_openai():
    init_db()
    client = OpenAIClient(auto_dispose=True)
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
