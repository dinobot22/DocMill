"""DocMill 模型配置 Schema 定义。

使用 Pydantic v2 进行配置校验。
所有模型通过 YAML 声明式配置，而非硬编码。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class PipelineType(str, Enum):
    """推理管线类型。"""

    VISION_ONLY = "vision_only"
    VISION_LLM = "vision_llm"
    LLM_ONLY = "llm_only"


class ExecutionMode(str, Enum):
    """执行策略类型。"""

    LOCAL = "local"
    HYBRID = "hybrid"
    REMOTE = "remote"


class VisionConfig(BaseModel):
    """Vision 模块配置。"""

    framework: str = Field(description="视觉框架: paddle / torch / custom")
    model_path: str = Field(default="", description="模型权重路径")
    device: str = Field(default="cuda:0", description="设备标识")
    extra_args: dict[str, Any] = Field(default_factory=dict, description="额外参数")


class LLMConfig(BaseModel):
    """LLM 模块配置（用于 vLLM sidecar 或外部 API）。"""

    backend: str = Field(default="vllm", description="LLM 后端: vllm / openai / custom")
    model_path: str = Field(default="", description="模型路径（HuggingFace 格式或本地路径）")
    api_base: str = Field(default="", description="外部 API 地址（如已有 vLLM 实例）")
    api_key: str = Field(default="dummy", description="API Key（vLLM 可用 dummy）")
    max_model_len: int = Field(default=4096, description="最大序列长度")
    trust_remote_code: bool = Field(default=True, description="是否信任远程代码")
    tensor_parallel_size: int = Field(default=1, description="张量并行大小")
    extra_launch_args: list[str] = Field(default_factory=list, description="额外 vLLM 启动参数")
    extra_args: dict[str, Any] = Field(default_factory=dict, description="额外参数")


class ResourceConfig(BaseModel):
    """资源配置（GPU 显存调度相关）。"""

    gpu_id: int = Field(default=0, description="GPU 设备索引")
    vision_vram_reserve_mb: float = Field(default=0, description="Vision 模块预留显存 (MB)")
    system_vram_reserve_mb: float = Field(default=2048, description="系统预留显存 (MB)")
    gpu_memory_utilization: float = Field(
        default=0.0,
        description="vLLM gpu_memory_utilization，0 表示自动计算",
        ge=0.0,
        le=1.0,
    )
    idle_timeout_s: int = Field(default=1800, description="模型空闲超时自动驱逐 (秒)")
    watermark: float = Field(default=0.9, description="显存高水位线 (0.0~1.0)")


class HooksConfig(BaseModel):
    """Hooks 配置（pre/post 处理适配器）。"""

    module: str = Field(default="", description="自定义 hooks 模块路径 (如 docmill.models.paddle_ocr_vl.hooks)")
    prompt_template: str = Field(default="", description="Prompt 模板路径或名称")


class ModelSpec(BaseModel):
    """完整的模型声明规范 — DocMill 的核心配置单元。

    每个模型对应一个 config.yaml，描述它使用什么 Pipeline、
    什么执行策略以及需要的资源。
    """

    name: str = Field(description="模型唯一标识符")
    version: str = Field(default="1.0.0", description="版本号")
    description: str = Field(default="", description="模型描述")

    pipeline: PipelineType = Field(description="推理管线类型")
    execution: ExecutionMode = Field(description="执行策略类型")

    vision: VisionConfig | None = Field(default=None, description="Vision 模块配置")
    llm: LLMConfig | None = Field(default=None, description="LLM 模块配置")
    resources: ResourceConfig = Field(default_factory=ResourceConfig, description="资源配置")
    hooks: HooksConfig = Field(default_factory=HooksConfig, description="Hooks 配置")

    @model_validator(mode="after")
    def validate_pipeline_requirements(self) -> "ModelSpec":
        """校验 Pipeline 类型与配置的一致性。"""
        if self.pipeline == PipelineType.VISION_ONLY:
            if self.vision is None:
                raise ValueError("vision_only pipeline 必须提供 vision 配置")

        elif self.pipeline == PipelineType.VISION_LLM:
            if self.vision is None:
                raise ValueError("vision_llm pipeline 必须提供 vision 配置")
            if self.llm is None:
                raise ValueError("vision_llm pipeline 必须提供 llm 配置")

        elif self.pipeline == PipelineType.LLM_ONLY:
            if self.llm is None:
                raise ValueError("llm_only pipeline 必须提供 llm 配置")

        return self

    @model_validator(mode="after")
    def validate_execution_requirements(self) -> "ModelSpec":
        """校验执行策略与配置的一致性。"""
        if self.execution == ExecutionMode.HYBRID:
            if self.vision is None or self.llm is None:
                raise ValueError("hybrid 执行策略需要同时配置 vision 和 llm")

        elif self.execution == ExecutionMode.REMOTE:
            if self.llm is None:
                raise ValueError("remote 执行策略需要配置 llm")

        return self
