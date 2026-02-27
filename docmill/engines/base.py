"""Engine 抽象基类 — OCR 模型的统一执行单元。

Engine 是 DocMill 的核心抽象：
- 每个 OCR 模型对应一个 Engine 实现
- Engine 负责模型加载、推理、卸载
- Engine 可以声明是否需要 vLLM sidecar
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EngineInput:
    """Engine 输入数据。

    支持多种输入格式：
    - file_path: 本地文件路径
    - image_bytes: 图片字节数据
    - url: 远程 URL
    - options: 额外选项
    """

    file_path: str | Path | None = None
    image_bytes: bytes | None = None
    url: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.file_path is None and self.image_bytes is None and self.url is None:
            raise ValueError("至少需要提供 file_path、image_bytes 或 url 之一")

    @property
    def has_file(self) -> bool:
        """是否有文件路径。"""
        return self.file_path is not None

    @property
    def has_bytes(self) -> bool:
        """是否有字节数据。"""
        return self.image_bytes is not None

    @property
    def has_url(self) -> bool:
        """是否有 URL。"""
        return self.url is not None


@dataclass
class EngineOutput:
    """Engine 输出结果。

    包含多种格式的输出：
    - text: 纯文本结果
    - markdown: Markdown 格式结果
    - structured: 结构化数据（JSON、字典等）
    - metadata: 元数据（模型信息、耗时等）
    """

    text: str = ""
    markdown: str = ""
    structured: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return {
            "text": self.text,
            "markdown": self.markdown,
            "structured": self.structured,
            "metadata": self.metadata,
        }


class BaseEngine(ABC):
    """OCR Engine 抽象基类。

    每个 Engine 实现代表一个具体的 OCR 模型：
    - 封装模型 SDK 的加载、推理、卸载逻辑
    - 声明是否需要 vLLM sidecar
    - 估算显存需求

    生命周期：
        Engine 实例化 -> load() -> infer() -> unload()
    """

    # ========== 类属性 ==========

    @classmethod
    @abstractmethod
    def engine_name(cls) -> str:
        """Engine 名称标识。

        Returns:
            如 "paddle_ocr_vl", "deepseek_ocr", "mineru"
        """
        ...

    @classmethod
    def requires_vllm_sidecar(cls) -> bool:
        """是否需要 vLLM sidecar。

        Returns:
            True 表示需要 Orchestrator 自动启动 vLLM sidecar。
        """
        return False

    # ========== 实例方法 ==========

    @abstractmethod
    def load(self, vllm_endpoint: str = "") -> None:
        """加载模型。

        Args:
            vllm_endpoint: vLLM 服务地址。
                如果 requires_vllm_sidecar() 返回 True，
                Orchestrator 会启动 sidecar 并注入此参数。

        Raises:
            RuntimeError: 加载失败。
        """
        ...

    @abstractmethod
    def infer(self, input_data: EngineInput) -> EngineOutput:
        """执行推理。

        Args:
            input_data: 输入数据。

        Returns:
            推理结果。

        Raises:
            RuntimeError: 推理失败。
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """卸载模型，释放资源。"""
        ...

    @abstractmethod
    def estimate_vram_mb(self) -> float:
        """估算本地显存需求 (MB)。

        注意：
            - vLLM sidecar 的显存由 Orchestrator 单独计算
            - 此方法只返回本地 Worker 部分的显存

        Returns:
            显存需求 (MB)。
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """检查模型是否已加载。

        Returns:
            True 表示已加载且可用。
        """
        ...

    # ========== 可选方法 ==========

    def health_check(self) -> bool:
        """健康检查。

        Returns:
            True 表示模型健康可用。
        """
        return self.is_loaded()

    def warmup(self) -> None:
        """预热模型（可选实现）。

        某些模型首次推理较慢，可在此方法中执行预热推理。
        """
        pass