"""DeepSeek OCR Hooks — 纯 vLLM OCR 适配器。

DeepSeek OCR 架构:
1. DeepEncoder: 将页面图片压缩为紧凑的 vision tokens (10x 压缩率)
2. 3B MoE Decoder: 从 vision tokens 重建文本/表格/公式

全部由 vLLM 处理, 无需本地 Vision Worker。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from docmill.pipelines.base import PipelineOutput


class DeepSeekOCRHooks:
    """DeepSeek OCR 模型适配钩子。"""

    def prehandle(self, raw_input: Any) -> Any:
        """预处理: 读取图片并转 base64。"""
        if isinstance(raw_input, dict):
            file_path = raw_input.get("file_path")
            image_bytes = raw_input.get("image_bytes")

            if file_path and Path(file_path).exists():
                with open(file_path, "rb") as f:
                    image_data = f.read()
                raw_input["image_base64"] = base64.b64encode(image_data).decode("utf-8")
            elif image_bytes:
                raw_input["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")

        return raw_input

    def build_prompt(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        """构建 DeepSeek OCR 的请求。

        DeepSeek OCR 支持 document-to-markdown 转换。
        """
        image_base64 = context.get("image_base64", "")

        content: list[dict[str, Any]] = []
        if image_base64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                },
            })

        # DeepSeek OCR 标准 prompt
        content.append({
            "type": "text",
            "text": "Perform OCR on this image. Output the text content in Markdown format, preserving the original layout structure including tables, formulas, and headings.",
        })

        return [{"role": "user", "content": content}]

    def posthandle(self, raw_output: Any) -> PipelineOutput:
        """后处理: 格式化 OCR 输出 (Markdown)。"""
        if isinstance(raw_output, dict):
            text = raw_output.get("text", "")
            return PipelineOutput(
                text=text,
                markdown=text,
                structured=raw_output,
                metadata={"model": "deepseek-ocr", "pipeline": "llm_only"},
            )
        if isinstance(raw_output, str):
            return PipelineOutput(text=raw_output, markdown=raw_output)
        return PipelineOutput(text=str(raw_output))
