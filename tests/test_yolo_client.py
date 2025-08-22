from aether import init_db
from aether.api.input import Input
from aether.api.request import AetherRequest
from aether.api.response import AetherResponse
from aether.call import GLOBAL_CLIENT_MANAGER, ActivatedToolRegistry


def test_yolo_detection():
    init_db()

    get_tools_req = AetherRequest(
        task="chat",
        tool_id=3,
        extra={
            "like": "yolo_detection",
        },
    )
    tools = GLOBAL_CLIENT_MANAGER.call(get_tools_req)
    assert tools.output is not None and len(tools.output) > 0, "获取工具列表失败"

    tool_id = tools.output["output"][0]["aether_tool_id"]

    req = AetherRequest(
        task="yolo_detection",
        tool_id=tool_id,
        input=Input(data=r"tests\1.webp"),
        extra={"conf_thres": 0.5},
    )
    res = GLOBAL_CLIENT_MANAGER.call(req)
    assert (
        isinstance(res, AetherResponse) and res.success and len(res.output["bbox"]) == 8
    )
    print(res.output)
    assert ActivatedToolRegistry.instance().get(tool_id) is None
