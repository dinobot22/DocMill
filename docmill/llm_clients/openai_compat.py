"""OpenAI 兼容 LLM Client — 用于 vLLM sidecar 交互。"""

from __future__ import annotations

from typing import Any

import httpx

from docmill.llm_clients.base import BaseLLMClient
from docmill.utils.logging import get_logger

logger = get_logger("llm_clients.openai_compat")


class OpenAICompatClient(BaseLLMClient):
    """OpenAI 兼容的 LLM Client。

    用于与 vLLM、OpenAI API 或任何兼容 API 交互。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "dummy",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(timeout),
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """发送 Chat Completions 请求。"""
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if model:
            payload["model"] = model
        payload.update(kwargs)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                logger.debug("LLM 响应 (tokens: %s)", data.get("usage", {}))
                return content
            except (httpx.HTTPError, KeyError, IndexError) as e:
                last_error = e
                logger.warning("LLM 请求失败 (attempt %d/%d): %s", attempt + 1, self.max_retries, e)

        raise RuntimeError(f"LLM 请求失败（已重试 {self.max_retries} 次）: {last_error}")

    def list_models(self) -> list[str]:
        """列出 vLLM / OpenAI 可用模型。"""
        try:
            response = self._client.get("/models")
            response.raise_for_status()
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            return models
        except Exception as e:
            logger.warning("获取模型列表失败: %s", e)
            return []

    def health(self) -> bool:
        """通过 /models 端点探测健康状态。"""
        try:
            response = self._client.get("/models")
            return response.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """关闭 HTTP Client。"""
        self._client.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
