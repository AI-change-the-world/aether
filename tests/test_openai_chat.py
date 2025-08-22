from aether import init_db
from aether.api.input import Input
from aether.api.request import AetherRequest, RequestMeta
from aether.api.response import AetherResponse
from aether.call import GLOBAL_CLIENT_MANAGER
from aether.call.client import ActivatedToolRegistry


def test_chat_openai():
    init_db()
    get_tools_req = AetherRequest(
        task="chat",
        tool_id=3,
        extra={
            "like": "chat",
        },
    )
    tools = GLOBAL_CLIENT_MANAGER.call(get_tools_req)
    assert tools.output is not None and len(tools.output) > 0, "获取工具列表失败"

    print(type(tools.output))

    tool_id = tools.output["output"][0]["aether_tool_id"]

    req = AetherRequest(
        task="chat",
        tool_id=tool_id,
        input=Input(data="repeat what i said\nhello world"),
        meta=RequestMeta(auto_dispose=False),
        extra={
            "temperature": 0.5,
            "history": [{"role": "system", "content": "you are a helpful assistant"}],
        },
    )
    res = GLOBAL_CLIENT_MANAGER.call(req)
    assert (
        isinstance(res, AetherResponse)
        and res.success
        and "hello world" in str(res.output["output"]).lower()
    )
    print(res.output)
    assert ActivatedToolRegistry.instance().get(tool_id) is not None
