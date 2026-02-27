"""Worker 抽象基类 — 本地推理执行单元。

Worker 是"Model.forward"角色：
- 负责实际的模型加载、推理和卸载
- 被 Pipeline 通过 ExecutionStrategy 调用
- 不决定流程拓扑
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseWorker(ABC):
    """Worker 抽象基类。"""

    @abstractmethod
    def load(self, **kwargs: Any) -> None:
        """加载模型权重到设备。"""
        ...

    @abstractmethod
    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        """执行推理。

        Args:
            payload: 输入数据字典。

        Returns:
            输出数据字典。
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """卸载模型，释放 GPU 显存。"""
        ...

    @abstractmethod
    def estimate_vram_mb(self) -> float:
        """估算模型所需显存 (MB)。"""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """检查模型是否已加载。"""
        ...
