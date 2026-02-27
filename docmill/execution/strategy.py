"""执行策略抽象基类。

ExecutionStrategy 决定"模型怎么跑"，而不是"模型是什么"。
三种实现对应三种运行时形态：
- LocalExecution: 纯本地 Python（torch/paddle）
- HybridExecution: 本地 Vision + 远程 LLM（vLLM sidecar）
- RemoteExecution: 纯远程 LLM（vLLM sidecar）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeHandle:
    """模型运行时句柄。"""

    model_name: str
    endpoint: str = ""  # LLM API endpoint (如果有)
    worker: Any = None  # 本地 Worker 实例 (如果有)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionStrategy(ABC):
    """执行策略抽象基类。"""

    @abstractmethod
    def ensure_ready(self, **kwargs: Any) -> RuntimeHandle:
        """确保运行时就绪（可能启动 sidecar / 加载模型）。

        Returns:
            运行时句柄。
        """
        ...

    @abstractmethod
    def infer(self, handle: RuntimeHandle, payload: dict[str, Any]) -> dict[str, Any]:
        """执行推理。

        Args:
            handle: 运行时句柄。
            payload: 输入数据。

        Returns:
            输出数据。
        """
        ...

    @abstractmethod
    def shutdown(self, handle: RuntimeHandle) -> None:
        """关闭运行时（可能停止 sidecar / 卸载模型）。"""
        ...

    @abstractmethod
    def health(self, handle: RuntimeHandle) -> bool:
        """检查运行时健康状态。"""
        ...

    @abstractmethod
    def estimate_vram_mb(self) -> float:
        """估算总显存需求 (MB)。"""
        ...
