"""DeepSeek OCR Engine — DeepSeek 视觉语言 OCR 模型。

DeepSeek OCR 是 DeepSeek 的视觉语言 OCR 模型：
- 纯 vLLM 部署
- DeepEncoder (Vision Compression) + MoE Decoder
- 高吞吐量 (2500 tok/s on A100)

使用方式:
    通过 vLLM 启动服务后，使用 OpenAI 兼容 API 调用。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from docmill.engines.base import BaseEngine, EngineInput, EngineOutput
from docmill.utils.logging import get_logger

logger = get_logger("engines.deepseek_ocr")


class DeepSeekOCREngine(BaseEngine):
    """DeepSeek OCR Engine。

    特点：
    - 纯 vLLM 部署，无本地 SDK
    - 通过 OpenAI 兼容 API 调用
    - 支持长上下文压缩
    """

    def __init__(
        self,
        vllm_model_path: str = "deepseek-ai/DeepSeek-OCR",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: float = 120.0,
        **kwargs: Any,
    ):
        """初始化 DeepSeek OCR Engine。

        Args:
            vllm_model_path: vLLM 模型路径或 HuggingFace 模型名。
            max_tokens: 最大生成 token 数。
            temperature: 采样温度。
            timeout: HTTP 请求超时时间。
            **kwargs: 其他参数。
        """
        self._vllm_model_path = vllm_model_path
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        self._extra_kwargs = kwargs

        self._client: Any = None
        self._vllm_endpoint: str = ""
        self._model_name: str = ""

    # ========== 类方法 ==========

    @classmethod
    def engine_name(cls) -> str:
        return "deepseek_ocr"

    @classmethod
    def requires_vllm_sidecar(cls) -> bool:
        return True

    # ========== 核心方法 ==========

    def load(self, vllm_endpoint: str = "") -> None:
        """加载 DeepSeek OCR 模型（初始化 HTTP 客户端）。

        Args:
            vllm_endpoint: vLLM 服务地址，如 "http://127.0.0.1:8080/v1"。
        """
        if self.is_loaded():
            logger.info("DeepSeek OCR 客户端已初始化，跳过")
            return

        self._vllm_endpoint = vllm_endpoint
        if not vllm_endpoint:
            raise ValueError("DeepSeek OCR 需要 vLLM endpoint，请检查 Orchestrator 配置")

        logger.info("初始化 DeepSeek OCR 客户端: endpoint=%s", vllm_endpoint)

        try:
            from docmill.clients.openai_compat import OpenAICompatClient

            self._client = OpenAICompatClient(
                base_url=vllm_endpoint,
                api_key="dummy",
                timeout=self._timeout,
            )

            # 获取模型名称
            models = self._client.list_models()
            if models:
                self._model_name = models[0]
                logger.info("DeepSeek OCR 模型: %s", self._model_name)
            else:
                # 使用默认模型名
                self._model_name = self._vllm_model_path.split("/")[-1]
                logger.warning("无法获取模型列表，使用默认名称: %s", self._model_name)

            logger.info("DeepSeek OCR 加载成功")

        except Exception as e:
            raise RuntimeError(f"DeepSeek OCR 初始化失败: {e}") from e

    def infer(self, input_data: EngineInput) -> EngineOutput:
        """执行 OCR 推理。

        Args:
            input_data: 输入数据，支持 file_path、image_bytes 或 url。

        Returns:
            OCR 结果。
        """
        if not self.is_loaded():
            raise RuntimeError("DeepSeek OCR 未加载，请先调用 load()")

        # 构建 messages
        messages = self._build_messages(input_data)
        logger.debug("DeepSeek OCR 推理: messages=%d", len(messages))

        try:
            # 调用 LLM
            response_text = self._client.chat(
                messages=messages,
                model=self._model_name,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )

            return EngineOutput(
                text=response_text,
                markdown=response_text,
                structured={"raw_response": response_text},
                metadata={
                    "engine": self.engine_name(),
                    "model": self._model_name,
                    "vllm_endpoint": self._vllm_endpoint,
                },
            )

        except Exception as e:
            logger.error("DeepSeek OCR 推理失败: %s", e)
            raise RuntimeError(f"推理失败: {e}") from e

    def unload(self) -> None:
        """卸载客户端。"""
        if self._client is not None:
            if hasattr(self._client, "close"):
                self._client.close()
            self._client = None
            logger.info("DeepSeek OCR 客户端已关闭")

    def estimate_vram_mb(self) -> float:
        """DeepSeek OCR 纯 vLLM，不占用本地显存。"""
        return 0.0

    def is_loaded(self) -> bool:
        """检查客户端是否已初始化。"""
        return self._client is not None

    # ========== 辅助方法 ==========

    def _build_messages(self, input_data: EngineInput) -> list[dict[str, Any]]:
        """构建 OpenAI 格式的消息。

        DeepSeek OCR 支持：
        - 图片 URL（远程或 base64）
        - 文本指令
        """
        content: list[dict[str, Any]] = []

        # 添加图片
        image_url = self._get_image_url(input_data)
        if image_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url},
            })

        # 添加文本指令
        prompt = input_data.options.get("prompt", "请识别图片中的所有文字内容，保持原始排版格式。")
        content.append({
            "type": "text",
            "text": prompt,
        })

        return [{"role": "user", "content": content}]

    def _get_image_url(self, input_data: EngineInput) -> str:
        """获取图片 URL。

        支持：
        - 远程 URL（直接使用）
        - 本地文件（转为 base64）
        - 字节数据（转为 base64）
        """
        # 优先使用远程 URL
        if input_data.url:
            return input_data.url

        # 本地文件
        if input_data.file_path:
            file_path = Path(input_data.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")

            image_bytes = file_path.read_bytes()
            return self._bytes_to_data_url(image_bytes)

        # 字节数据
        if input_data.image_bytes:
            return self._bytes_to_data_url(input_data.image_bytes)

        raise ValueError("EngineInput 需要提供 file_path、url 或 image_bytes")

    def _bytes_to_data_url(self, image_bytes: bytes) -> str:
        """将图片字节转为 data URL。"""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        # 简单判断图片类型
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            mime = "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            mime = "image/jpeg"
        else:
            mime = "image/png"  # 默认 PNG

        return f"data:{mime};base64,{b64}"

    @property
    def vllm_model_path(self) -> str:
        """获取 vLLM 模型路径。"""
        return self._vllm_model_path