from abc import ABC, abstractmethod

from aether.api.request import AetherRequest

class ModelHandler(ABC):
    def __init__(self, tool_model, session):
        self.tool_model = tool_model
        self.session = session

    @abstractmethod
    def handle(self, req: AetherRequest) -> dict:
        pass