"""Engine 注册表 — 管理所有可用的 Engine 类型。

提供 Engine 的注册、发现和实例化功能。
"""

from __future__ import annotations

from pathlib import Path
from typing import Type

import yaml

from docmill.engines.base import BaseEngine
from docmill.utils.logging import get_logger

logger = get_logger("engines.registry")


class EngineRegistry:
    """Engine 注册表。

    用于管理所有可用的 Engine 类型：
    - 注册新 Engine 类型
    - 查找 Engine 类型
    - 列出所有已注册的 Engine
    """

    _engines: dict[str, Type[BaseEngine]] = {}

    @classmethod
    def register(cls, engine_class: Type[BaseEngine]) -> Type[BaseEngine]:
        """注册 Engine 类型。

        可以作为装饰器使用：
            @EngineRegistry.register
            class MyEngine(BaseEngine):
                ...

        Args:
            engine_class: Engine 类。

        Returns:
            注册的 Engine 类（便于装饰器使用）。
        """
        name = engine_class.engine_name()
        if name in cls._engines:
            logger.warning("Engine '%s' 已存在，将被覆盖", name)

        cls._engines[name] = engine_class
        logger.info("注册 Engine: %s -> %s", name, engine_class.__name__)
        return engine_class

    @classmethod
    def get(cls, name: str) -> Type[BaseEngine] | None:
        """获取 Engine 类型。

        Args:
            name: Engine 名称。

        Returns:
            Engine 类，如果不存在返回 None。
        """
        return cls._engines.get(name)

    @classmethod
    def get_or_raise(cls, name: str) -> Type[BaseEngine]:
        """获取 Engine 类型，不存在则抛出异常。

        Args:
            name: Engine 名称。

        Returns:
            Engine 类。

        Raises:
            KeyError: Engine 不存在。
        """
        engine_class = cls._engines.get(name)
        if engine_class is None:
            available = list(cls._engines.keys())
            raise KeyError(f"Engine '{name}' 未注册，可用: {available}")
        return engine_class

    @classmethod
    def list_engines(cls) -> list[str]:
        """列出所有已注册的 Engine 名称。

        Returns:
            Engine 名称列表。
        """
        return list(cls._engines.keys())

    @classmethod
    def list_vllm_engines(cls) -> list[str]:
        """列出所有需要 vLLM sidecar 的 Engine。

        Returns:
            需要 vLLM 的 Engine 名称列表。
        """
        return [
            name
            for name, engine_class in cls._engines.items()
            if engine_class.requires_vllm_sidecar()
        ]

    @classmethod
    def clear(cls) -> None:
        """清空注册表（主要用于测试）。"""
        cls._engines.clear()
        logger.debug("Engine 注册表已清空")

    @classmethod
    def get_defaults(cls, name: str) -> dict:
        """读取 Engine 的 defaults.yaml 作为默认配置。

        Args:
            name: Engine 名称。

        Returns:
            默认配置字典。
        """
        engine_dir = Path(__file__).parent / name
        defaults_file = engine_dir / "defaults.yaml"
        
        if defaults_file.exists():
            return yaml.safe_load(defaults_file.read_text()) or {}
        return {}


# 自动注册内置 Engine
def _register_builtin_engines() -> None:
    """注册内置的 Engine 类型。"""
    # 延迟导入避免循环依赖
    try:
        from docmill.engines.paddle_ocr_vl import PaddleOCRVLEngine
        from docmill.engines.deepseek_ocr import DeepSeekOCREngine

        EngineRegistry.register(PaddleOCRVLEngine)
        EngineRegistry.register(DeepSeekOCREngine)
    except ImportError as e:
        logger.warning("部分内置 Engine 注册失败: %s", e)


# 模块加载时自动注册
_register_builtin_engines()
