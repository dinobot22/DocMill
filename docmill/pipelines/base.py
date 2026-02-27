"""Pipeline 基类与核心抽象。

Pipeline 是 DocMill 的"制度层"：
- Pipeline 定义推理拓扑（先 Vision 后 LLM，还是纯 LLM）
- 模型只是 Pipeline 的配置和组件组合
- Pipeline 不允许出现在 models/ 中
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class PipelineInput:
    """Pipeline 输入数据。"""

    file_path: str | Path | None = None
    image_bytes: bytes | None = None
    raw_text: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.file_path is None and self.image_bytes is None and self.raw_text is None:
            raise ValueError("至少需要提供 file_path、image_bytes 或 raw_text 之一")


@dataclass
class PipelineOutput:
    """Pipeline 输出结果。"""

    text: str = ""
    markdown: str = ""
    structured: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class HooksProtocol(Protocol):
    """模型适配钩子协议。

    Hooks 只负责输入/输出的适配，不能决定流程拓扑。
    类似 transformers 的 Tokenizer / Head 角色。
    """

    def prehandle(self, raw_input: Any) -> Any:
        """预处理输入数据（如 resize、normalize、格式转换）。"""
        ...

    def build_prompt(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        """构建 LLM 请求的 messages（仅 vision_llm / llm_only 使用）。"""
        ...

    def posthandle(self, raw_output: Any) -> PipelineOutput:
        """后处理模型输出（如 JSON 解析、Markdown 格式化）。"""
        ...


class DefaultHooks:
    """默认空实现的 Hooks — 透传数据。"""

    def prehandle(self, raw_input: Any) -> Any:
        return raw_input

    def build_prompt(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        """默认 prompt 构建：将 context 中的 text 包装成 user message。"""
        text = context.get("text", "")
        image_url = context.get("image_url", "")

        content: list[dict[str, Any]] = []
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        if text:
            content.append({"type": "text", "text": text})

        if not content:
            content.append({"type": "text", "text": "请识别这张图片中的文字内容。"})

        return [{"role": "user", "content": content}]

    def posthandle(self, raw_output: Any) -> PipelineOutput:
        if isinstance(raw_output, PipelineOutput):
            return raw_output
        if isinstance(raw_output, str):
            return PipelineOutput(text=raw_output, markdown=raw_output)
        if isinstance(raw_output, dict):
            return PipelineOutput(
                text=raw_output.get("text", ""),
                markdown=raw_output.get("markdown", ""),
                structured=raw_output,
            )
        return PipelineOutput(text=str(raw_output))


class BasePipeline(ABC):
    """Pipeline 抽象基类。

    Pipeline 是"Trainer"角色：
    - 固定推理拓扑
    - 调用 Worker / LLM Client 进行实际计算
    - 通过 Hooks 适配不同模型的输入输出
    """

    def __init__(self, hooks: HooksProtocol | None = None):
        self.hooks: HooksProtocol = hooks or DefaultHooks()

    @abstractmethod
    def run(self, pipeline_input: PipelineInput) -> PipelineOutput:
        """执行完整推理流程。

        Args:
            pipeline_input: 输入数据。

        Returns:
            推理结果。
        """
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """检查 Pipeline 所有组件是否就绪。"""
        ...
