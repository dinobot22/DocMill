"""模型运行时池 — 管理模型实例生命周期和资源调度。

Pool 是 DocMill 的"大管家"：
- 模型生命周期管理（COLD → LOADING → READY → IDLE → EVICTED）
- LRU + Watermark 调度策略
- 自动拉起/释放 vLLM sidecar
- 防止重复加载（per-model lock）
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any

from docmill.config.schema import ExecutionMode, ModelSpec
from docmill.llm_clients.openai_compat import OpenAICompatClient
from docmill.orchestrator.health import HealthChecker
from docmill.orchestrator.launcher import SidecarLauncher, SidecarProcess
from docmill.orchestrator.planner import ResourcePlanner
from docmill.orchestrator.registry import ModelRuntime, RuntimeRegistry, RuntimeState
from docmill.pipelines.base import BasePipeline
from docmill.pipelines.factory import create_pipeline, create_strategy
from docmill.utils.errors import InsufficientVRAMError, ModelLoadTimeoutError
from docmill.utils.hardware import get_total_vram_mb
from docmill.utils.logging import get_logger

logger = get_logger("orchestrator.pool")


class ModelRuntimePool:
    """模型运行时池。

    核心职责：
    1. get_or_load(spec) → Pipeline（可能阻塞等待加载）
    2. release(model_name) → 标记最后访问时间
    3. evict_if_needed(required_vram) → LRU 驱逐 IDLE 模型
    4. gc_idle() → 空闲超时自动驱逐
    """

    def __init__(
        self,
        default_watermark: float = 0.9,
        health_check_timeout: float = 180.0,
    ):
        self.registry = RuntimeRegistry()
        self.launcher = SidecarLauncher()
        self.planner = ResourcePlanner()
        self.health_checker = HealthChecker()
        self.default_watermark = default_watermark
        self.health_check_timeout = health_check_timeout

        self._pipelines: dict[str, BasePipeline] = {}
        self._sidecars: dict[str, SidecarProcess] = {}
        self._model_locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def get_or_load(self, spec: ModelSpec) -> BasePipeline:
        """获取或加载模型，返回就绪的 Pipeline。

        如果模型已在 READY/IDLE 状态，直接复用。
        否则启动加载流程（可能启动 vLLM sidecar）。

        Args:
            spec: 模型配置规范。

        Returns:
            就绪的 Pipeline 实例。
        """
        model_name = spec.name

        # 获取 per-model lock（防止并发重复加载）
        lock = self._get_model_lock(model_name)
        with lock:
            # 检查是否已有可用实例
            runtime = self.registry.get(model_name)
            if runtime and runtime.state in (RuntimeState.READY, RuntimeState.IDLE):
                runtime.touch()
                runtime.state = RuntimeState.READY
                runtime.active_requests += 1
                logger.info("复用已有模型: %s (state=%s)", model_name, runtime.state.value)
                return self._pipelines[model_name]

            # 需要新加载
            return self._load_model(spec)

    def release(self, model_name: str) -> None:
        """释放模型使用（减少活跃请求计数）。"""
        runtime = self.registry.get(model_name)
        if runtime:
            runtime.active_requests = max(0, runtime.active_requests - 1)
            runtime.touch()
            if runtime.active_requests == 0:
                runtime.state = RuntimeState.IDLE
            logger.debug("释放模型: %s (active=%d)", model_name, runtime.active_requests)

    def evict(self, model_name: str) -> None:
        """显式驱逐指定模型。"""
        lock = self._get_model_lock(model_name)
        with lock:
            self._evict_runtime(model_name)

    def gc_idle(self) -> int:
        """垃圾回收：驱逐超时的 IDLE 模型。

        Returns:
            驱逐的模型数量。
        """
        evicted_count = 0
        for runtime in self.registry.list_all():
            if runtime.state == RuntimeState.IDLE and not runtime.is_busy:
                idle_timeout = runtime.metadata.get("idle_timeout_s", 1800)
                if runtime.idle_seconds > idle_timeout:
                    logger.info("GC: 驱逐空闲模型 %s (空闲 %.0fs > %ds)", runtime.model_name, runtime.idle_seconds, idle_timeout)
                    self._evict_runtime(runtime.model_name)
                    evicted_count += 1
        return evicted_count

    def shutdown(self) -> None:
        """关闭所有运行时并清理资源。"""
        logger.info("正在关闭 ModelRuntimePool...")
        for runtime in self.registry.list_all():
            self._evict_runtime(runtime.model_name)
        self.launcher.stop_all()
        logger.info("ModelRuntimePool 已关闭")

    def list_runtimes(self) -> list[dict[str, Any]]:
        """列出所有运行时信息。"""
        result = []
        for runtime in self.registry.list_all():
            result.append({
                "name": runtime.model_name,
                "state": runtime.state.value,
                "endpoint": runtime.endpoint,
                "vram_mb": runtime.estimated_vram_mb,
                "active_requests": runtime.active_requests,
                "idle_seconds": runtime.idle_seconds,
            })
        return result

    # --- Private methods ---

    def _load_model(self, spec: ModelSpec) -> BasePipeline:
        """加载模型的完整流程。"""
        model_name = spec.name
        logger.info("开始加载模型: %s (pipeline=%s, execution=%s)", model_name, spec.pipeline.value, spec.execution.value)

        # 1. 注册 LOADING 状态
        spec_hash = self._hash_spec(spec)
        runtime = ModelRuntime(
            model_name=model_name,
            spec_hash=spec_hash,
            state=RuntimeState.LOADING,
            metadata={"idle_timeout_s": spec.resources.idle_timeout_s},
        )
        self.registry.register(runtime)

        try:
            # 2. 如果需要 vLLM sidecar，先启动
            sidecar: SidecarProcess | None = None
            if spec.execution in (ExecutionMode.HYBRID, ExecutionMode.REMOTE):
                sidecar = self._start_sidecar(spec)
                runtime.endpoint = sidecar.endpoint
                runtime.port = sidecar.port
                runtime.pid = sidecar.pid

            # 3. 创建 Strategy 和 Pipeline
            # 如果有 sidecar，需要更新 spec 中的 api_base
            if sidecar and spec.llm:
                spec.llm.api_base = sidecar.endpoint

            strategy = create_strategy(spec)
            pipeline = create_pipeline(spec, strategy)

            # 4. 准备 Pipeline
            pipeline.ensure_ready(model_name=model_name)

            # 5. 估算显存
            runtime.estimated_vram_mb = strategy.estimate_vram_mb()
            if sidecar:
                # 加上 vLLM 显存
                plan = self.planner.plan(spec)
                runtime.estimated_vram_mb += plan.estimated_vram_mb

            # 6. 更新状态为 READY
            runtime.state = RuntimeState.READY
            runtime.active_requests = 1

            self._pipelines[model_name] = pipeline
            if sidecar:
                self._sidecars[model_name] = sidecar

            logger.info("模型加载完成: %s (vram_estimate=%.0fMB)", model_name, runtime.estimated_vram_mb)
            return pipeline

        except Exception as e:
            runtime.state = RuntimeState.FAILED
            logger.error("模型加载失败: %s - %s", model_name, e)
            self._evict_runtime(model_name)
            raise

    def _start_sidecar(self, spec: ModelSpec) -> SidecarProcess:
        """启动 vLLM sidecar 并等待就绪。"""
        plan = self.planner.plan(spec)

        # 检查显存是否足够
        self._ensure_vram_available(plan.estimated_vram_mb, spec)

        # 启动 sidecar
        sidecar = self.launcher.launch(
            model_path=plan.model_path,
            port=plan.port,
            gpu_memory_utilization=plan.gpu_memory_utilization,
            max_model_len=plan.max_model_len,
            tensor_parallel_size=plan.tensor_parallel_size,
            trust_remote_code=plan.trust_remote_code,
            extra_args=plan.extra_args,
            gpu_id=plan.gpu_id,
        )

        # 等待就绪
        try:
            self.health_checker.wait_until_ready(
                endpoint=sidecar.endpoint,
                timeout=self.health_check_timeout,
                model_name=spec.name,
            )
        except ModelLoadTimeoutError:
            self.launcher.stop(sidecar)
            raise

        return sidecar

    def _ensure_vram_available(self, required_mb: float, spec: ModelSpec) -> None:
        """确保有足够显存，必要时驱逐 IDLE 模型。"""
        gpu_id = spec.resources.gpu_id
        total_vram = get_total_vram_mb(gpu_id)
        if total_vram <= 0:
            logger.warning("无法检测 GPU 显存，跳过显存检查")
            return

        watermark = spec.resources.watermark or self.default_watermark
        max_allowed = total_vram * watermark
        current_usage = self.registry.total_estimated_vram_mb()

        if current_usage + required_mb <= max_allowed:
            return  # 显存充足

        # 需要驱逐
        need_free = current_usage + required_mb - max_allowed
        logger.info(
            "显存压力: 需要释放 %.0fMB (当前=%.0fMB, 需要=%.0fMB, 上限=%.0fMB)",
            need_free, current_usage, required_mb, max_allowed,
        )

        freed = 0.0
        for idle_runtime in self.registry.get_idle_sorted_by_lru():
            if freed >= need_free:
                break
            freed += idle_runtime.estimated_vram_mb
            logger.info("驱逐 IDLE 模型: %s (释放 %.0fMB)", idle_runtime.model_name, idle_runtime.estimated_vram_mb)
            self._evict_runtime(idle_runtime.model_name)

        if freed < need_free:
            raise InsufficientVRAMError(
                required_mb=required_mb,
                available_mb=max_allowed - (current_usage - freed),
                model_name=spec.name,
            )

    def _evict_runtime(self, model_name: str) -> None:
        """驱逐单个运行时实例。"""
        # 关闭 Pipeline/Strategy
        pipeline = self._pipelines.pop(model_name, None)

        # 停止 sidecar
        sidecar = self._sidecars.pop(model_name, None)
        if sidecar:
            self.launcher.stop(sidecar)

        # 更新注册表
        self.registry.update_state(model_name, RuntimeState.EVICTED)
        self.registry.unregister(model_name)

        logger.info("模型已驱逐: %s", model_name)

    def _get_model_lock(self, model_name: str) -> threading.Lock:
        """获取 per-model 锁。"""
        with self._global_lock:
            if model_name not in self._model_locks:
                self._model_locks[model_name] = threading.Lock()
            return self._model_locks[model_name]

    @staticmethod
    def _hash_spec(spec: ModelSpec) -> str:
        """生成配置唯一 hash。"""
        key = f"{spec.name}:{spec.pipeline.value}:{spec.execution.value}"
        if spec.llm:
            key += f":{spec.llm.model_path}"
        if spec.vision:
            key += f":{spec.vision.model_path}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
