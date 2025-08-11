from aether import init_db
from aether.api.input import Input
from aether.api.request import AetherRequest, RequestMeta
from aether.api.response import AetherResponse
from aether.call import GLOBAL_CLIENT_MANAGER
from aether.call.client import ActivatedToolRegistry


def test_chat_openai():
    init_db()
    req = AetherRequest(
        task="chat",
        tool_id=1,
        input=Input(data="repeat what i said\nhello world"),
        meta=RequestMeta(auto_dispose=False),
        extra={
            "temperature": 0.5,
            "history": [{"role": "system", "content": "you are a helpful assistant"}],
        },
    )
    client = GLOBAL_CLIENT_MANAGER.get_client(req)
    res = client.call(req)
    assert (
        isinstance(res, AetherResponse)
        and res.success
        and "hello world" in str(res.output["output"])
    )
    print(res.output)
    assert ActivatedToolRegistry.instance().get(1) is not None
