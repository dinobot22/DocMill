"""LLM Client 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """LLM Client 抽象基类。

    所有 LLM 交互通过 HTTP API 进行，
    vLLM 只是实现之一。
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """发送聊天请求。

        Args:
            messages: OpenAI 格式的消息列表。
            model: 模型名称。
            max_tokens: 最大生成 token 数。
            temperature: 采样温度。

        Returns:
            生成的文本内容。
        """
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """列出可用模型。"""
        ...

    @abstractmethod
    def health(self) -> bool:
        """健康检查。

        Returns:
            True 表示服务可用。
        """
        ...
