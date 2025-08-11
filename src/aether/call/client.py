import functools
import json
from threading import RLock
from typing import Callable, Dict, Optional

from aether.call._map import create_client_from_json
from aether.call.base import BaseClient
from aether.common.logger import logger
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


def register_on_success(get_tool_id: Callable):
    """
    装饰器：方法执行成功后，将当前 client 注册到全局 ActivatedToolRegistry
    - get_tool_id: (self, args, kwargs) -> tool_id
    - 如果 auto_dispose=False，则跳过注册
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            try:
                # 跳过不需要注册的客户端
                if getattr(self, "auto_dispose", True) is True:
                    return result

                tool_id = get_tool_id(self, args, kwargs)
                if tool_id:
                    ActivatedToolRegistry.instance().register(tool_id, self)
            except Exception as e:
                logger.warning(f"Registry auto-register failed: {e}")
            return result

        return wrapper

    return decorator


class ClientManager:
    """统一管理客户端的生命周期和缓存"""

    def __init__(self, registry: ActivatedToolRegistry, session):
        self.registry = registry
        self.session = session

    def get_client(self, tool_id: int) -> BaseClient:
        """根据 tool_id 获取或创建并缓存一个客户端实例"""
        client = self.registry.get(tool_id)
        if client:
            logger.info(f"Using cached client for tool_id: {tool_id}")
            return client

        # 缓存中没有，则创建新的客户端
        logger.info(f"Client for tool_id: {tool_id} not found, creating new one.")
        tool_model = AetherToolCRUD.get_by_id(self.session, tool_id)
        if not tool_model:
            raise ValueError(f"Tool model with ID {tool_id} not found.")

        try:
            tool_config = json.loads(tool_model.tool_config)
            new_client = create_client_from_json(tool_config)

            # 注册到缓存中（如果客户端不需要自动释放）
            # 这段逻辑应在ClientManager中，而不是在客户端内部
            if not new_client.auto_dispose:
                self.registry.register(tool_id, new_client)

            return new_client

        except json.JSONDecodeError:
            logger.error(f"Tool model config for tool_id {tool_id} is not valid json")
            raise ValueError("tool model config is not valid json")
        except Exception as e:
            logger.error(f"Failed to create client for tool_id {tool_id}: {e}")
            raise e
