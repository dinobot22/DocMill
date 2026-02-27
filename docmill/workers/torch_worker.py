"""Torch Worker — 基于 PyTorch 的本地 Worker 实现。

v1: Stub 占位实现，用于验证 Pipeline 端到端流程。
后续替换为真实模型加载逻辑。
"""

from __future__ import annotations

from typing import Any

from docmill.utils.logging import get_logger
from docmill.workers.base import BaseWorker

logger = get_logger("workers.torch")


class TorchWorker(BaseWorker):
    """基于 PyTorch 的 Worker。"""

    def __init__(self, model_path: str = "", device: str = "cuda:0", **kwargs: Any):
        self.model_path = model_path
        self.device = device
        self.extra_args = kwargs
        self._loaded = False
        self._model: Any = None

    def load(self, **kwargs: Any) -> None:
        """加载 PyTorch 模型。"""
        logger.info("加载 Torch 模型: %s -> %s", self.model_path, self.device)
        # TODO: 集成真实模型加载逻辑
        # import torch
        # self._model = torch.load(self.model_path)
        # self._model.to(self.device)
        self._loaded = True
        logger.info("Torch 模型已就绪 (stub)")

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        """执行 PyTorch 推理。"""
        if not self._loaded:
            raise RuntimeError("模型未加载，请先调用 load()")

        logger.debug("Torch Worker 推理 (stub): keys=%s", list(payload.keys()))
        # TODO: 替换为真实推理逻辑
        return {
            "text": f"[Torch Stub Output] 输入: {list(payload.keys())}",
            "vision_features": None,
        }

    def unload(self) -> None:
        """卸载模型。"""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        logger.info("Torch 模型已卸载")

    def estimate_vram_mb(self) -> float:
        """估算显存需求。"""
        # TODO: 根据实际模型自动估算
        return self.extra_args.get("vram_estimate_mb", 2048.0)

    def is_loaded(self) -> bool:
        return self._loaded
