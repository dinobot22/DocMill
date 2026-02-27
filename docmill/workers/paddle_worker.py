"""Paddle Worker — 基于 PaddlePaddle 的本地 Worker 实现。

v1: Stub 占位实现。
后续集成 PaddleOCR Vision Encoder 等组件。
"""

from __future__ import annotations

from typing import Any

from docmill.utils.logging import get_logger
from docmill.workers.base import BaseWorker

logger = get_logger("workers.paddle")


class PaddleWorker(BaseWorker):
    """基于 PaddlePaddle 的 Worker。"""

    def __init__(self, model_path: str = "", device: str = "cuda:0", **kwargs: Any):
        self.model_path = model_path
        self.device = device
        self.extra_args = kwargs
        self._loaded = False
        self._model: Any = None

    def load(self, **kwargs: Any) -> None:
        """加载 Paddle 模型。"""
        logger.info("加载 Paddle 模型: %s -> %s", self.model_path, self.device)
        # TODO: 集成 PaddleOCR Vision Encoder
        # import paddle
        # from paddleocr import ...
        self._loaded = True
        logger.info("Paddle 模型已就绪 (stub)")

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        """执行 Paddle 推理。"""
        if not self._loaded:
            raise RuntimeError("模型未加载，请先调用 load()")

        logger.debug("Paddle Worker 推理 (stub): keys=%s", list(payload.keys()))
        # TODO: 替换为真实推理逻辑
        return {
            "text": f"[Paddle Stub Output] 输入: {list(payload.keys())}",
            "vision_features": None,
            "layout": None,
        }

    def unload(self) -> None:
        """卸载模型。"""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        logger.info("Paddle 模型已卸载")

    def estimate_vram_mb(self) -> float:
        """估算显存需求。"""
        return self.extra_args.get("vram_estimate_mb", 2048.0)

    def is_loaded(self) -> bool:
        return self._loaded
