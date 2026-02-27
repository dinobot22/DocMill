"""LocalExecution — 纯本地 Python 执行策略。

适用于纯 PyTorch / PaddlePaddle 模型。
无 sidecar、无远程调用。
"""

from __future__ import annotations

from typing import Any

from docmill.execution.strategy import ExecutionStrategy, RuntimeHandle
from docmill.utils.logging import get_logger
from docmill.workers.base import BaseWorker

logger = get_logger("execution.local")


class LocalExecution(ExecutionStrategy):
    """本地执行策略。"""

    def __init__(self, worker: BaseWorker):
        self.worker = worker

    def ensure_ready(self, **kwargs: Any) -> RuntimeHandle:
        """加载本地模型。"""
        if not self.worker.is_loaded():
            logger.info("加载本地模型...")
            self.worker.load(**kwargs)
        return RuntimeHandle(
            model_name=kwargs.get("model_name", "local"),
            worker=self.worker,
        )

    def infer(self, handle: RuntimeHandle, payload: dict[str, Any]) -> dict[str, Any]:
        """本地推理。"""
        return self.worker.infer(payload)

    def shutdown(self, handle: RuntimeHandle) -> None:
        """卸载模型。"""
        self.worker.unload()
        logger.info("本地模型已关闭")

    def health(self, handle: RuntimeHandle) -> bool:
        """检查模型是否已加载。"""
        return self.worker.is_loaded()

    def estimate_vram_mb(self) -> float:
        return self.worker.estimate_vram_mb()
