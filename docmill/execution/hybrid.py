"""HybridExecution — 本地 Vision + 远程 LLM 执行策略。

适用于 PaddleOCR-VL、MinerU 等混合模型：
- Vision 部分通过本地 Worker 处理
- LLM 部分通过 vLLM sidecar（HTTP API）处理
"""

from __future__ import annotations

from typing import Any

from docmill.execution.strategy import ExecutionStrategy, RuntimeHandle
from docmill.llm_clients.base import BaseLLMClient
from docmill.utils.logging import get_logger
from docmill.workers.base import BaseWorker

logger = get_logger("execution.hybrid")


class HybridExecution(ExecutionStrategy):
    """混合执行策略：本地 Vision + 远程 LLM。"""

    def __init__(
        self,
        vision_worker: BaseWorker,
        llm_client: BaseLLMClient,
        llm_model_name: str = "",
    ):
        self.vision_worker = vision_worker
        self.llm_client = llm_client
        self.llm_model_name = llm_model_name

    def ensure_ready(self, **kwargs: Any) -> RuntimeHandle:
        """加载 Vision 模型并验证 LLM 可用性。"""
        # 1. 加载本地 Vision Worker
        if not self.vision_worker.is_loaded():
            logger.info("加载 Vision Worker...")
            self.vision_worker.load(**kwargs)

        # 2. 验证 LLM Client 可用
        if not self.llm_client.health():
            raise RuntimeError("LLM 后端不可用，请检查 vLLM sidecar 状态")

        logger.info("Hybrid 模式就绪: Vision (local) + LLM (remote)")
        return RuntimeHandle(
            model_name=kwargs.get("model_name", "hybrid"),
            worker=self.vision_worker,
            metadata={"llm_client": self.llm_client},
        )

    def infer(self, handle: RuntimeHandle, payload: dict[str, Any]) -> dict[str, Any]:
        """混合推理：先 Vision 后 LLM。"""
        # 1. Vision 处理
        logger.debug("执行 Vision 推理...")
        vision_result = self.vision_worker.infer(payload)

        # 2. 构建 LLM 请求
        messages = payload.get("messages", [])
        if not messages:
            # 使用 Vision 输出构建默认消息
            messages = [
                {
                    "role": "user",
                    "content": vision_result.get("text", ""),
                }
            ]

        # 3. LLM 推理
        logger.debug("执行 LLM 推理...")
        llm_response = self.llm_client.chat(
            messages=messages,
            model=self.llm_model_name,
            max_tokens=payload.get("max_tokens", 4096),
            temperature=payload.get("temperature", 0.0),
        )

        return {
            "text": llm_response,
            "vision_result": vision_result,
        }

    def shutdown(self, handle: RuntimeHandle) -> None:
        """卸载 Vision 模型（LLM sidecar 由 Orchestrator 管理）。"""
        self.vision_worker.unload()
        logger.info("Hybrid Vision Worker 已关闭")

    def health(self, handle: RuntimeHandle) -> bool:
        """检查 Vision 和 LLM 健康状态。"""
        vision_ok = self.vision_worker.is_loaded()
        llm_ok = self.llm_client.health()
        return vision_ok and llm_ok

    def estimate_vram_mb(self) -> float:
        """Vision Worker 显存 + LLM sidecar 显存（由 Orchestrator 计算）。"""
        return self.vision_worker.estimate_vram_mb()
