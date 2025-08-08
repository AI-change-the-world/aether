from aether import init_db
from aether.api.input import Input
from aether.api.request import AetherRequest
from aether.api.response import AetherResponse
from aether.call import YoloClient


def test_yolo_detection():
    init_db()
    client = YoloClient(auto_dispose=True)
    req = AetherRequest(
        task="yolo_detection",
        model_id=2,
        input=Input(data=r"tests\1.webp"),
        extra={"conf_thres": 0.5},
    )
    res = client.call(req)
    assert (
        isinstance(res, AetherResponse) and res.success and len(res.output["bbox"]) == 8
    )
    print(res.output)
