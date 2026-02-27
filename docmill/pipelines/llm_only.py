"""LLMOnlyPipeline — 纯 LLM 推理管线。

流程: input → prehandle → build_prompt → llm_client → posthandle → output

适用于 DeepSeek OCR 等纯 vLLM 模型。
"""

from __future__ import annotations

from typing import Any

from docmill.execution.strategy import ExecutionStrategy, RuntimeHandle
from docmill.pipelines.base import BasePipeline, HooksProtocol, PipelineInput, PipelineOutput
from docmill.utils.logging import get_logger

logger = get_logger("pipelines.llm_only")


class LLMOnlyPipeline(BasePipeline):
    """纯 LLM 推理管线。"""

    def __init__(
        self,
        strategy: ExecutionStrategy,
        hooks: HooksProtocol | None = None,
    ):
        super().__init__(hooks=hooks)
        self.strategy = strategy
        self._handle: RuntimeHandle | None = None

    def run(self, pipeline_input: PipelineInput) -> PipelineOutput:
        """执行纯 LLM 推理。"""
        if self._handle is None:
            raise RuntimeError("Pipeline 未就绪，请先调用 ensure_ready()")

        # 1. prehandle
        processed_input = self.hooks.prehandle(self._build_payload(pipeline_input))

        # 2. build_prompt
        context = processed_input if isinstance(processed_input, dict) else {"data": processed_input}
        messages = self.hooks.build_prompt(context)

        # 3. LLM 推理
        payload = {"messages": messages}
        if isinstance(processed_input, dict):
            payload.update(
                {k: v for k, v in processed_input.items() if k in ("max_tokens", "temperature")}
            )

        logger.debug("执行 LLM 推理...")
        raw_result = self.strategy.infer(self._handle, payload)

        # 4. posthandle
        output = self.hooks.posthandle(raw_result)
        return output

    def ensure_ready(self, **kwargs: Any) -> None:
        """准备 Pipeline。"""
        self._handle = self.strategy.ensure_ready(**kwargs)

    def is_ready(self) -> bool:
        return self._handle is not None and self.strategy.health(self._handle)

    @staticmethod
    def _build_payload(pipeline_input: PipelineInput) -> dict[str, Any]:
        """将 PipelineInput 转为 payload。"""
        payload: dict[str, Any] = {}
        if pipeline_input.file_path:
            payload["file_path"] = str(pipeline_input.file_path)
        if pipeline_input.image_bytes:
            payload["image_bytes"] = pipeline_input.image_bytes
        if pipeline_input.raw_text:
            payload["text"] = pipeline_input.raw_text
        payload.update(pipeline_input.options)
        return payload
