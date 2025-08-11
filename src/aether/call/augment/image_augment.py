from aether.call.base import BaseClient


class ImageCvAugment(BaseClient):
    def __init__(self, config=None):
        super().__init__(config)

    def call(self, req, **kwargs):
        return self.__call(req)
