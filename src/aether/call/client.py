import functools
from threading import RLock
from typing import Callable, Dict, Optional

from aether.call.base import BaseClient
from aether.common.logger import logger


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
