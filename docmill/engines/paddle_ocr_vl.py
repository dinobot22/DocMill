"""PaddleOCR-VL Engine — PaddleOCR Vision-Language OCR 模型。

PaddleOCR-VL 是 PaddlePaddle 的视觉语言 OCR 模型：
- 本地 Vision Encoder (PaddleOCR)
- 远程 LLM Decoder (vLLM server)

使用方式:
    from paddleocr import PaddleOCRVL
    pipeline = PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url="http://127.0.0.1:8080/v1"
    )
    result = pipeline.predict(image_path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docmill.engines.base import BaseEngine, EngineInput, EngineOutput
from docmill.utils.logging import get_logger

logger = get_logger("engines.paddle_ocr_vl")


class PaddleOCRVLEngine(BaseEngine):
    """PaddleOCR-VL Engine。

    特点：
    - 使用 paddleocr SDK
    - 需要 vLLM sidecar 作为 VL 后端
    - 支持多语言 OCR (109 种语言)
    """

    def __init__(
        self,
        vllm_model_path: str = "",
        vram_estimate_mb: float = 2048.0,
        use_gpu: bool = True,
        lang: str = "ch",
        **kwargs: Any,
    ):
        """初始化 PaddleOCR-VL Engine。

        Args:
            vllm_model_path: vLLM 模型路径（由 Orchestrator 使用）。
            vram_estimate_mb: 本地显存估算 (MB)。
            use_gpu: 是否使用 GPU。
            lang: OCR 语言，默认中文。
            **kwargs: 其他参数传递给 PaddleOCRVL。
        """
        self._vllm_model_path = vllm_model_path
        self._vram_estimate_mb = vram_estimate_mb
        self._use_gpu = use_gpu
        self._lang = lang
        self._extra_kwargs = kwargs

        self._pipeline: Any = None
        self._vllm_endpoint: str = ""

    # ========== 类方法 ==========

    @classmethod
    def engine_name(cls) -> str:
        return "paddle_ocr_vl"

    @classmethod
    def requires_vllm_sidecar(cls) -> bool:
        return True

    # ========== 核心方法 ==========

    def load(self, vllm_endpoint: str = "") -> None:
        """加载 PaddleOCR-VL 模型。

        Args:
            vllm_endpoint: vLLM 服务地址，如 "http://127.0.0.1:8080/v1"。
        """
        if self.is_loaded():
            logger.info("PaddleOCR-VL 已加载，跳过")
            return

        self._vllm_endpoint = vllm_endpoint
        if not vllm_endpoint:
            raise ValueError("PaddleOCR-VL 需要 vLLM endpoint，请检查 Orchestrator 配置")

        logger.info("加载 PaddleOCR-VL: vllm_endpoint=%s, lang=%s", vllm_endpoint, self._lang)

        try:
            from paddleocr import PaddleOCRVL

            self._pipeline = PaddleOCRVL(
                vl_rec_backend="vllm-server",
                vl_rec_server_url=vllm_endpoint,
                use_gpu=self._use_gpu,
                lang=self._lang,
                **self._extra_kwargs,
            )
            logger.info("PaddleOCR-VL 加载成功")

        except ImportError as e:
            raise RuntimeError(
                "paddleocr 未安装，请执行: pip install paddleocr"
            ) from e
        except Exception as e:
            raise RuntimeError(f"PaddleOCR-VL 加载失败: {e}") from e

    def infer(self, input_data: EngineInput) -> EngineOutput:
        """执行 OCR 推理。

        Args:
            input_data: 输入数据，支持 file_path 或 image_bytes。

        Returns:
            OCR 结果。
        """
        if not self.is_loaded():
            raise RuntimeError("PaddleOCR-VL 未加载，请先调用 load()")

        # 准备输入
        image_input = self._prepare_input(input_data)
        logger.debug("PaddleOCR-VL 推理: input=%s", image_input[:50] if isinstance(image_input, str) else type(image_input))

        try:
            # 执行推理
            results = self._pipeline.predict(image_input)

            # 解析结果
            return self._parse_results(results)

        except Exception as e:
            logger.error("PaddleOCR-VL 推理失败: %s", e)
            raise RuntimeError(f"推理失败: {e}") from e

    def unload(self) -> None:
        """卸载模型。"""
        if self._pipeline is not None:
            # PaddleOCRVL 没有显式的卸载方法
            # 通过删除引用让 GC 回收
            del self._pipeline
            self._pipeline = None
            logger.info("PaddleOCR-VL 已卸载")

    def estimate_vram_mb(self) -> float:
        """估算本地显存需求。"""
        return self._vram_estimate_mb

    def is_loaded(self) -> bool:
        """检查是否已加载。"""
        return self._pipeline is not None

    # ========== 辅助方法 ==========

    def _prepare_input(self, input_data: EngineInput) -> str:
        """准备输入数据。

        PaddleOCRVL.predict() 支持：
        - 本地文件路径
        - URL
        - numpy array

        目前优先使用 file_path，其次 URL。
        """
        if input_data.file_path:
            return str(input_data.file_path)
        if input_data.url:
            return input_data.url
        if input_data.image_bytes:
            # 字节数据需要先保存为临时文件
            import tempfile
            import uuid

            suffix = input_data.options.get("image_suffix", ".png")
            temp_path = tempfile.gettempdir() / Path(f"docmill_{uuid.uuid4().hex}{suffix}")
            temp_path.write_bytes(input_data.image_bytes)
            logger.debug("保存临时文件: %s", temp_path)
            return str(temp_path)

        raise ValueError("EngineInput 需要提供 file_path、url 或 image_bytes")

    def _parse_results(self, results: Any) -> EngineOutput:
        """解析 PaddleOCRVL 结果。

        PaddleOCRVL.predict() 返回一个生成器或列表，
        每个元素包含 OCR 结果对象。
        """
        text_parts: list[str] = []
        markdown_parts: list[str] = []
        structured_data: dict[str, Any] = {"pages": []}

        try:
            # results 可能是生成器或列表
            for result in results:
                # 尝试获取文本
                if hasattr(result, "text"):
                    text_parts.append(result.text)
                elif hasattr(result, "rec_text"):
                    text_parts.append(result.rec_text)

                # 尝试获取 markdown
                if hasattr(result, "markdown"):
                    markdown_parts.append(result.markdown)
                elif hasattr(result, "to_markdown"):
                    markdown_parts.append(result.to_markdown())

                # 尝试获取结构化数据
                if hasattr(result, "to_dict"):
                    structured_data["pages"].append(result.to_dict())
                elif hasattr(result, "boxes"):
                    structured_data["pages"].append({
                        "boxes": result.boxes,
                        "texts": getattr(result, "texts", []),
                    })

        except TypeError:
            # results 不是可迭代的，直接处理
            if hasattr(results, "text"):
                text_parts.append(results.text)
            if hasattr(results, "markdown"):
                markdown_parts.append(results.markdown)
            if hasattr(results, "to_dict"):
                structured_data["pages"].append(results.to_dict())

        # 合并结果
        combined_text = "\n".join(text_parts)
        combined_markdown = "\n".join(markdown_parts) or combined_text

        return EngineOutput(
            text=combined_text,
            markdown=combined_markdown,
            structured=structured_data,
            metadata={
                "engine": self.engine_name(),
                "vllm_endpoint": self._vllm_endpoint,
            },
        )

    @property
    def vllm_model_path(self) -> str:
        """获取 vLLM 模型路径。"""
        return self._vllm_model_path