from aether.call.base import BaseClient


class ImageCvAugment(BaseClient):
    def __init__(self, config=None, auto_dispose=True):
        super().__init__(config, auto_dispose)

    def call(self, req, **kwargs):
        return self.__call(req)
