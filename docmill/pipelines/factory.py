"""Pipeline 工厂 — 根据 ModelSpec 创建对应的 Pipeline 实例。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from docmill.config.schema import ExecutionMode, PipelineType
from docmill.utils.logging import get_logger

if TYPE_CHECKING:
    from docmill.config.schema import ModelSpec
    from docmill.execution.strategy import ExecutionStrategy
    from docmill.pipelines.base import BasePipeline, HooksProtocol

logger = get_logger("pipelines.factory")


def create_pipeline(
    spec: "ModelSpec",
    strategy: "ExecutionStrategy",
    hooks: "HooksProtocol | None" = None,
) -> "BasePipeline":
    """根据 ModelSpec 创建 Pipeline。

    Args:
        spec: 模型配置规范。
        strategy: 执行策略实例。
        hooks: 可选的自定义 Hooks。

    Returns:
        对应的 Pipeline 实例。
    """
    from docmill.pipelines.llm_only import LLMOnlyPipeline
    from docmill.pipelines.vision_llm import VisionLLMPipeline
    from docmill.pipelines.vision_only import VisionOnlyPipeline

    if spec.pipeline == PipelineType.VISION_ONLY:
        pipeline = VisionOnlyPipeline(strategy=strategy, hooks=hooks)
    elif spec.pipeline == PipelineType.VISION_LLM:
        pipeline = VisionLLMPipeline(strategy=strategy, hooks=hooks)
    elif spec.pipeline == PipelineType.LLM_ONLY:
        pipeline = LLMOnlyPipeline(strategy=strategy, hooks=hooks)
    else:
        raise ValueError(f"不支持的 Pipeline 类型: {spec.pipeline}")

    logger.info("已创建 Pipeline: %s (type=%s, execution=%s)", spec.name, spec.pipeline.value, spec.execution.value)
    return pipeline


def create_strategy(spec: "ModelSpec") -> "ExecutionStrategy":
    """根据 ModelSpec 创建 ExecutionStrategy。

    Args:
        spec: 模型配置规范。

    Returns:
        对应的 ExecutionStrategy 实例。
    """
    from docmill.execution.hybrid import HybridExecution
    from docmill.execution.local import LocalExecution
    from docmill.execution.remote import RemoteExecution
    from docmill.llm_clients.openai_compat import OpenAICompatClient

    if spec.execution == ExecutionMode.LOCAL:
        worker = _create_worker(spec)
        return LocalExecution(worker=worker)

    elif spec.execution == ExecutionMode.HYBRID:
        worker = _create_worker(spec)
        llm_client = _create_llm_client(spec)
        model_name = spec.llm.model_path if spec.llm else ""
        return HybridExecution(
            vision_worker=worker,
            llm_client=llm_client,
            llm_model_name=model_name,
        )

    elif spec.execution == ExecutionMode.REMOTE:
        llm_client = _create_llm_client(spec)
        model_name = spec.llm.model_path if spec.llm else ""
        return RemoteExecution(llm_client=llm_client, llm_model_name=model_name)

    else:
        raise ValueError(f"不支持的执行模式: {spec.execution}")


def _create_worker(spec: "ModelSpec") -> "BaseWorker":
    """根据 Vision 配置创建 Worker。"""
    from docmill.workers.base import BaseWorker
    from docmill.workers.paddle_worker import PaddleWorker
    from docmill.workers.torch_worker import TorchWorker

    if spec.vision is None:
        raise ValueError(f"模型 '{spec.name}' 需要 vision 配置")

    framework = spec.vision.framework.lower()
    if framework == "paddle":
        return PaddleWorker(
            model_path=spec.vision.model_path,
            device=spec.vision.device,
            **spec.vision.extra_args,
        )
    elif framework in ("torch", "pytorch"):
        return TorchWorker(
            model_path=spec.vision.model_path,
            device=spec.vision.device,
            **spec.vision.extra_args,
        )
    else:
        raise ValueError(f"不支持的 Vision 框架: {framework}")


def _create_llm_client(spec: "ModelSpec") -> "OpenAICompatClient":
    """根据 LLM 配置创建 Client。"""
    from docmill.llm_clients.openai_compat import OpenAICompatClient

    if spec.llm is None:
        raise ValueError(f"模型 '{spec.name}' 需要 llm 配置")

    base_url = spec.llm.api_base or "http://127.0.0.1:8000/v1"
    return OpenAICompatClient(
        base_url=base_url,
        api_key=spec.llm.api_key,
    )
