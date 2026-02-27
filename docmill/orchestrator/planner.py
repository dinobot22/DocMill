"""资源规划器 — 自动计算 vLLM 启动参数。

Planner 的职责：
- 检测 GPU 显存
- 计算 Vision 模型预留后的可用显存
- 自动生成 gpu_memory_utilization 等参数
"""

from __future__ import annotations

from dataclasses import dataclass

from docmill.config.schema import ModelSpec
from docmill.utils.hardware import get_total_vram_mb
from docmill.utils.logging import get_logger

logger = get_logger("orchestrator.planner")


@dataclass
class LaunchPlan:
    """vLLM 启动计划。"""

    model_path: str
    port: int | None  # None = 自动分配
    gpu_memory_utilization: float
    max_model_len: int
    tensor_parallel_size: int
    trust_remote_code: bool
    extra_args: list[str]
    gpu_id: int
    estimated_vram_mb: float  # 预计 vLLM 占用的显存


class ResourcePlanner:
    """资源规划器。"""

    def plan(self, spec: ModelSpec) -> LaunchPlan:
        """根据 ModelSpec 生成 vLLM 启动计划。

        自动计算 gpu_memory_utilization：
        - 获取 GPU 总显存
        - 减去 Vision 模块预留
        - 减去系统预留
        - 计算比例并 clamp 到 [0.1, 0.95]

        Args:
            spec: 模型配置。

        Returns:
            LaunchPlan。
        """
        if spec.llm is None:
            raise ValueError(f"模型 '{spec.name}' 没有 LLM 配置")

        resources = spec.resources
        gpu_id = resources.gpu_id
        total_vram = get_total_vram_mb(gpu_id)

        # 计算 gpu_memory_utilization
        if resources.gpu_memory_utilization > 0:
            # 用户显式指定
            utilization = resources.gpu_memory_utilization
            logger.info("使用用户指定的 gpu_memory_utilization: %.2f", utilization)
        elif total_vram > 0:
            # 自动计算
            available = total_vram - resources.vision_vram_reserve_mb - resources.system_vram_reserve_mb
            utilization = available / total_vram
            utilization = max(0.1, min(0.95, utilization))
            logger.info(
                "自动计算 gpu_memory_utilization: %.2f "
                "(总显存=%.0fMB, Vision预留=%.0fMB, 系统预留=%.0fMB, 可用=%.0fMB)",
                utilization,
                total_vram,
                resources.vision_vram_reserve_mb,
                resources.system_vram_reserve_mb,
                available,
            )
        else:
            # 无法检测 GPU，使用安全默认值
            utilization = 0.8
            logger.warning("无法检测 GPU 显存，使用默认 gpu_memory_utilization=0.8")

        estimated_vram = total_vram * utilization if total_vram > 0 else 0

        plan = LaunchPlan(
            model_path=spec.llm.model_path,
            port=None,  # 由 Launcher 自动分配
            gpu_memory_utilization=utilization,
            max_model_len=spec.llm.max_model_len,
            tensor_parallel_size=spec.llm.tensor_parallel_size,
            trust_remote_code=spec.llm.trust_remote_code,
            extra_args=list(spec.llm.extra_launch_args),
            gpu_id=gpu_id,
            estimated_vram_mb=estimated_vram,
        )

        logger.info(
            "启动计划: model=%s, utilization=%.2f, max_len=%d, tp=%d",
            plan.model_path,
            plan.gpu_memory_utilization,
            plan.max_model_len,
            plan.tensor_parallel_size,
        )

        return plan
