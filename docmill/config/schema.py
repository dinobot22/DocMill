"""DocMill 配置模型 — 基于 Pydantic 的配置定义。

支持 YAML 文件 + 环境变量 + CLI 参数三层覆盖。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class VLLMConfig(BaseModel):
    """vLLM Sidecar 配置。"""
    
    mode: str = "managed"  # "managed"=DocMill自动管理 | "external"=连接外部已有的vLLM
    model_path: str = ""  # managed 模式必填：HuggingFace 模型路径或本地路径
    endpoints: list[str] = []  # external 模式必填：外部 vLLM API 列表
    gpu_id: int = 0
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    served_model_name: str | None = None
    extra_args: list[str] = []
    trust_remote_code: bool = True


class EngineConfig(BaseModel):
    """单个 Engine 配置。"""
    
    engine: str  # Engine 注册名（如 "paddle_ocr_vl"）
    auto_start: bool = True  # 是否随系统启动
    vllm: VLLMConfig | None = None  # vLLM 配置（Engine.requires_vllm_sidecar()=True 时必填）
    engine_kwargs: dict[str, Any] = {}  # 传给 Engine 构造函数的额外参数


class ServerConfig(BaseModel):
    """服务器配置。"""
    
    api_port: int = 8080
    data_dir: str = "/tmp/docmill"
    workers_per_gpu: int = 1
    gpu_devices: list[int] | str = "auto"  # "auto" 或 [0, 1, 2]


class DocMillConfig(BaseModel):
    """DocMill 全局配置。"""
    
    server: ServerConfig = ServerConfig()
    engines: dict[str, EngineConfig] = Field(default_factory=dict)
