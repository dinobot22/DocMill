"""RemoteExecution — 纯远程 LLM 执行策略。

适用于 DeepSeek OCR 等纯 vLLM 模型：
- 无本地 Vision Worker
- 所有推理通过 vLLM sidecar HTTP API 完成
"""

from __future__ import annotations

from typing import Any

from docmill.execution.strategy import ExecutionStrategy, RuntimeHandle
from docmill.llm_clients.base import BaseLLMClient
from docmill.utils.logging import get_logger

logger = get_logger("execution.remote")


class RemoteExecution(ExecutionStrategy):
    """远程执行策略：纯 LLM sidecar。"""

    def __init__(self, llm_client: BaseLLMClient, llm_model_name: str = ""):
        self.llm_client = llm_client
        self.llm_model_name = llm_model_name

    def ensure_ready(self, **kwargs: Any) -> RuntimeHandle:
        """验证 LLM 后端可用性。"""
        if not self.llm_client.health():
            raise RuntimeError("LLM 后端不可用，请检查 vLLM sidecar 状态")

        logger.info("Remote 模式就绪: LLM (remote)")
        return RuntimeHandle(
            model_name=kwargs.get("model_name", "remote"),
            metadata={"llm_client": self.llm_client},
        )

    def infer(self, handle: RuntimeHandle, payload: dict[str, Any]) -> dict[str, Any]:
        """远程推理：直接调用 LLM。"""
        messages = payload.get("messages", [])
        if not messages:
            messages = [
                {
                    "role": "user",
                    "content": payload.get("text", "请识别文档内容。"),
                }
            ]

        llm_response = self.llm_client.chat(
            messages=messages,
            model=self.llm_model_name,
            max_tokens=payload.get("max_tokens", 4096),
            temperature=payload.get("temperature", 0.0),
        )

        return {"text": llm_response}

    def shutdown(self, handle: RuntimeHandle) -> None:
        """无需关闭本地资源（sidecar 由 Orchestrator 管理）。"""
        logger.info("Remote 执行策略已关闭")

    def health(self, handle: RuntimeHandle) -> bool:
        """检查 LLM 健康状态。"""
        return self.llm_client.health()

    def estimate_vram_mb(self) -> float:
        """远程模式不消耗本地 Worker 显存。"""
        return 0.0
