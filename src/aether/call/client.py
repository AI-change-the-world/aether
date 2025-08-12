import json
from threading import RLock
from typing import Dict, Optional

from aether.api.request import AetherRequest
from aether.api.response import AetherResponse
from aether.call._map import create_client
from aether.call.base import BaseClient
from aether.common.logger import logger
from aether.db.basic import get_session
from aether.models.tool.tool_crud import AetherToolCRUD


class ActivatedToolRegistry:
    """全局激活工具注册表（懒加载单例）"""

    _instance: Optional["ActivatedToolRegistry"] = None
    _lock = RLock()

    def __init__(self):
        self._tools: Dict[int, "BaseClient"] = {}
        self._data_lock = RLock()

    @classmethod
    def instance(cls) -> "ActivatedToolRegistry":
        """获取单例实例（懒加载）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # 双重检查
                    cls._instance = cls()
        return cls._instance

    def register(self, tool_id: int, client: "BaseClient") -> None:
        """注册激活工具"""
        with self._data_lock:
            self._tools[tool_id] = client

    def get(self, tool_id: int) -> Optional["BaseClient"]:
        """根据 ID 获取激活工具"""
        with self._data_lock:
            return self._tools.get(tool_id)

    def unregister(self, tool_id: int) -> None:
        """移除激活工具"""
        with self._data_lock:
            self._tools.pop(tool_id, None)

    def all(self) -> Dict[int, "BaseClient"]:
        """获取全部激活工具副本"""
        with self._data_lock:
            return dict(self._tools)


class ClientManager:
    """统一管理客户端的生命周期和缓存"""

    def __init__(self, registry: ActivatedToolRegistry, session):
        self.registry = registry
        self.session = session

    def __get_client(self, req: AetherRequest) -> BaseClient:
        """根据 tool_id 获取或创建并缓存一个客户端实例"""
        client = self.registry.get(req.tool_id)
        if client:
            logger.info(f"Using cached client for tool_id: {req.tool_id}")
            return client

        # 缓存中没有，则创建新的客户端
        logger.info(f"Client for tool_id: {req.tool_id} not found, creating new one.")
        tool_model = AetherToolCRUD.get_by_id(self.session, req.tool_id)
        if not tool_model:
            raise ValueError(f"Tool model with ID {req.tool_id} not found.")

        try:
            new_client = create_client(tool_model)

            # 注册到缓存中（如果客户端不需要自动释放）
            # 这段逻辑应在ClientManager中，而不是在客户端内部
            if not req.meta.auto_dispose:
                self.registry.register(req.tool_id, new_client)

            return new_client

        except json.JSONDecodeError:
            logger.error(
                f"Tool model config for tool_id {req.tool_id} is not valid json"
            )
            raise ValueError("tool model config is not valid json")
        except Exception as e:
            logger.error(f"Failed to create client for tool_id {req.tool_id}: {e}")
            raise e

    def call(self, req: AetherRequest) -> "AetherResponse":
        """调用客户端的 call 方法"""
        client = self.__get_client(req)
        return client.call(req)


GLOBAL_CLIENT_MANAGER = ClientManager(ActivatedToolRegistry.instance(), get_session())
