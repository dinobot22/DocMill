"""VisionOnlyPipeline — 纯视觉推理管线。

流程: input → prehandle → vision_worker → posthandle → output

适用于纯 OCR / Layout 分析模型（如 PaddleOCR 基础版）。
"""

from __future__ import annotations

from typing import Any

from docmill.execution.strategy import ExecutionStrategy, RuntimeHandle
from docmill.pipelines.base import BasePipeline, HooksProtocol, PipelineInput, PipelineOutput
from docmill.utils.logging import get_logger

logger = get_logger("pipelines.vision_only")


class VisionOnlyPipeline(BasePipeline):
    """纯视觉推理管线。"""

    def __init__(
        self,
        strategy: ExecutionStrategy,
        hooks: HooksProtocol | None = None,
    ):
        super().__init__(hooks=hooks)
        self.strategy = strategy
        self._handle: RuntimeHandle | None = None

    def run(self, pipeline_input: PipelineInput) -> PipelineOutput:
        """执行纯视觉推理。"""
        if self._handle is None:
            raise RuntimeError("Pipeline 未就绪，请先调用 ensure_ready()")

        # 1. prehandle
        processed_input = self.hooks.prehandle(self._build_payload(pipeline_input))

        # 2. vision worker 推理
        logger.debug("执行 Vision 推理...")
        raw_result = self.strategy.infer(self._handle, processed_input)

        # 3. posthandle
        output = self.hooks.posthandle(raw_result)
        return output

    def ensure_ready(self, **kwargs: Any) -> None:
        """准备 Pipeline。"""
        self._handle = self.strategy.ensure_ready(**kwargs)

    def is_ready(self) -> bool:
        return self._handle is not None and self.strategy.health(self._handle)

    @staticmethod
    def _build_payload(pipeline_input: PipelineInput) -> dict[str, Any]:
        """将 PipelineInput 转为 Worker payload。"""
        payload: dict[str, Any] = {}
        if pipeline_input.file_path:
            payload["file_path"] = str(pipeline_input.file_path)
        if pipeline_input.image_bytes:
            payload["image_bytes"] = pipeline_input.image_bytes
        if pipeline_input.raw_text:
            payload["text"] = pipeline_input.raw_text
        payload.update(pipeline_input.options)
        return payload
