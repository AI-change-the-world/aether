from aether import init_db
from aether.api.input import Input
from aether.api.request import AetherRequest
from aether.api.response import AetherResponse
from aether.call.client import Client


def test_chat_openai():
    init_db()
    client = Client(auto_dispose=True)
    req = AetherRequest(task="chat", model_id=1, input=Input(data="hello"))
    res = client.call(req)
    assert isinstance(res, AetherResponse) and res.success
    print(res.output)
