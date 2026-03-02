"""PaddleOCR-VL Engine — PaddleOCR Vision-Language OCR 模型。

PaddleOCR-VL 是 PaddlePaddle 的视觉语言 OCR 模型：
- 本地 Vision Encoder (PaddleOCR)
- 远程 LLM Decoder (vLLM server)

使用方式:
    from paddleocr import PaddleOCRVL
    pipeline = PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url="http://localhost:30023/v1",
        vl_rec_api_model_name="PaddleOCR-VL-0.9B"
    )
    result = pipeline.predict(image_path)
"""

from __future__ import annotations

import tempfile
import uuid
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
    - 支持 PDF、图片等格式
    """

    def __init__(
        self,
        vllm_model_path: str = "",
        vllm_endpoint: str = "",
        vllm_model_name: str = "PaddleOCR-VL-0.9B",
        **kwargs: Any,
    ):
        """初始化 PaddleOCR-VL Engine。

        Args:
            vllm_model_path: vLLM 模型路径（由 Orchestrator 使用）。
            vllm_endpoint: vLLM 服务地址，如 "http://localhost:30023/v1"。
            vllm_model_name: vLLM 模型名称。
            **kwargs: 其他参数传递给 PaddleOCRVL。
        """
        self._vllm_model_path = vllm_model_path
        self._vllm_endpoint = vllm_endpoint
        self._vllm_model_name = vllm_model_name
        self._extra_kwargs = kwargs

        self._pipeline: Any = None

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
            vllm_endpoint: vLLM 服务地址，如 "http://localhost:30023/v1"。
        """
        if self.is_loaded():
            logger.info("PaddleOCR-VL 已加载，跳过")
            return

        # 优先使用传入的 endpoint，其次使用初始化时的配置
        endpoint = vllm_endpoint or self._vllm_endpoint
        if not endpoint:
            raise ValueError("PaddleOCR-VL 需要 vLLM endpoint，请配置 vllm_endpoint")

        self._vllm_endpoint = endpoint
        logger.info("加载 PaddleOCR-VL: endpoint=%s, model=%s", endpoint, self._vllm_model_name)

        try:
            from paddleocr import PaddleOCRVL

            self._pipeline = PaddleOCRVL(
                vl_rec_backend="vllm-server",
                vl_rec_server_url=endpoint,
                vl_rec_api_model_name=self._vllm_model_name,
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
        logger.debug("PaddleOCR-VL 推理: input=%s", image_input)

        try:
            # 执行推理 - predict 返回生成器
            results = self._pipeline.predict(image_input)

            # 转换为列表并解析
            pages_results = list(results)

            # 可选：重组页面结构
            try:
                restructured = self._pipeline.restructure_pages(pages_results)
                return self._parse_restructured_results(restructured, pages_results)
            except Exception as e:
                logger.warning("重组页面失败，使用原始结果: %s", e)
                return self._parse_results(pages_results)

        except Exception as e:
            logger.error("PaddleOCR-VL 推理失败: %s", e)
            raise RuntimeError(f"推理失败: {e}") from e

    def unload(self) -> None:
        """卸载模型。"""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            logger.info("PaddleOCR-VL 已卸载")

    def estimate_vram_mb(self) -> float:
        """估算本地显存需求。"""
        return 2048.0  # PaddleOCR 部分大约需要 2GB

    def is_loaded(self) -> bool:
        """检查是否已加载。"""
        return self._pipeline is not None

    # ========== 辅助方法 ==========

    def _prepare_input(self, input_data: EngineInput) -> str:
        """准备输入数据。"""
        if input_data.file_path:
            return str(input_data.file_path)
        if input_data.url:
            return input_data.url
        if input_data.image_bytes:
            # 字节数据需要先保存为临时文件
            suffix = input_data.options.get("image_suffix", ".png")
            temp_path = Path(tempfile.gettempdir()) / f"docmill_{uuid.uuid4().hex}{suffix}"
            temp_path.write_bytes(input_data.image_bytes)
            logger.debug("保存临时文件: %s", temp_path)
            return str(temp_path)

        raise ValueError("EngineInput 需要提供 file_path、url 或 image_bytes")

    def _parse_results(self, pages_results: list) -> EngineOutput:
        """解析原始页面结果。"""
        text_parts: list[str] = []
        markdown_parts: list[str] = []
        structured_data: dict[str, Any] = {"pages": []}

        for i, result in enumerate(pages_results):
            page_data = {"page_num": i + 1}

            # 获取文本
            if hasattr(result, "text"):
                txt = result.text
                if isinstance(txt, str):
                    text_parts.append(txt)
                    page_data["text"] = txt
                elif isinstance(txt, dict):
                    # 可能是结构化文本
                    page_data["text"] = str(txt)
            elif hasattr(result, "rec_text"):
                txt = result.rec_text
                if isinstance(txt, str):
                    text_parts.append(txt)
                    page_data["text"] = txt

            # 获取 markdown
            if hasattr(result, "markdown"):
                md = result.markdown
                if isinstance(md, str):
                    markdown_parts.append(md)
                    page_data["markdown"] = md
                elif isinstance(md, dict):
                    # 可能是结构化 markdown
                    page_data["markdown_dict"] = md

            # 获取结构化数据
            if hasattr(result, "to_dict"):
                page_data["structured"] = result.to_dict()
            elif hasattr(result, "__dict__"):
                page_data["structured"] = result.__dict__

            structured_data["pages"].append(page_data)

        combined_text = "\n\n".join(text_parts) if text_parts else ""
        combined_markdown = "\n\n".join(markdown_parts) if markdown_parts else combined_text

        return EngineOutput(
            text=combined_text,
            markdown=combined_markdown,
            structured=structured_data,
            metadata={
                "engine": self.engine_name(),
                "vllm_endpoint": self._vllm_endpoint,
                "model_name": self._vllm_model_name,
                "pages": len(pages_results),
            },
        )

    def _parse_restructured_results(self, restructured: list, pages_results: list) -> EngineOutput:
        """解析重组后的结果。"""
        text_parts: list[str] = []
        markdown_parts: list[str] = []
        structured_data: dict[str, Any] = {"pages": [], "restructured": []}

        # 处理原始页面结果
        for i, result in enumerate(pages_results):
            page_data = {"page_num": i + 1}
            if hasattr(result, "text"):
                txt = result.text
                if isinstance(txt, str):
                    page_data["text"] = txt
            if hasattr(result, "to_dict"):
                page_data["structured"] = result.to_dict()
            structured_data["pages"].append(page_data)

        # 处理重组后的结果
        for i, res in enumerate(restructured):
            # 获取文本
            if hasattr(res, "text"):
                txt = res.text
                if isinstance(txt, str):
                    text_parts.append(txt)

            # 获取 markdown
            if hasattr(res, "markdown"):
                md = res.markdown
                if isinstance(md, str):
                    markdown_parts.append(md)

            # 获取结构化数据
            if hasattr(res, "to_dict"):
                structured_data["restructured"].append(res.to_dict())
            elif hasattr(res, "__dict__"):
                structured_data["restructured"].append(res.__dict__)

        combined_text = "\n\n".join(text_parts) if text_parts else str(pages_results)
        combined_markdown = "\n\n".join(markdown_parts) if markdown_parts else combined_text

        return EngineOutput(
            text=combined_text,
            markdown=combined_markdown,
            structured=structured_data,
            metadata={
                "engine": self.engine_name(),
                "vllm_endpoint": self._vllm_endpoint,
                "model_name": self._vllm_model_name,
                "pages": len(pages_results),
                "restructured": len(restructured),
            },
        )

    @property
    def vllm_endpoint(self) -> str:
        """获取 vLLM 服务地址。"""
        return self._vllm_endpoint

    @property
    def vllm_model_name(self) -> str:
        """获取 vLLM 模型名称。"""
        return self._vllm_model_name
