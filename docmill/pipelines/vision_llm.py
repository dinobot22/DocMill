"""VisionLLMPipeline — Vision + LLM 混合推理管线。

流程: input → prehandle → vision_worker → build_prompt → llm_client → posthandle → output

适用于 PaddleOCR-VL、MinerU 等混合模型。
"""

from __future__ import annotations

from typing import Any

from docmill.execution.strategy import ExecutionStrategy, RuntimeHandle
from docmill.pipelines.base import BasePipeline, HooksProtocol, PipelineInput, PipelineOutput
from docmill.utils.logging import get_logger

logger = get_logger("pipelines.vision_llm")


class VisionLLMPipeline(BasePipeline):
    """Vision + LLM 混合推理管线。"""

    def __init__(
        self,
        strategy: ExecutionStrategy,
        hooks: HooksProtocol | None = None,
    ):
        super().__init__(hooks=hooks)
        self.strategy = strategy
        self._handle: RuntimeHandle | None = None

    def run(self, pipeline_input: PipelineInput) -> PipelineOutput:
        """执行 Vision → LLM 混合推理。"""
        if self._handle is None:
            raise RuntimeError("Pipeline 未就绪，请先调用 ensure_ready()")

        # 1. prehandle — 输入预处理
        processed_input = self.hooks.prehandle(self._build_payload(pipeline_input))

        # 2. 构建 LLM 请求（使用 hooks.build_prompt）
        context = processed_input if isinstance(processed_input, dict) else {"data": processed_input}
        messages = self.hooks.build_prompt(context)

        # 3. 将 messages 注入 payload，由 strategy 协调 vision + llm
        if isinstance(processed_input, dict):
            processed_input["messages"] = messages
        else:
            processed_input = {"data": processed_input, "messages": messages}

        logger.debug("执行 Vision+LLM 混合推理...")
        raw_result = self.strategy.infer(self._handle, processed_input)

        # 4. posthandle — 输出后处理
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
