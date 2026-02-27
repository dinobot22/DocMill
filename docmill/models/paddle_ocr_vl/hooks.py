"""PaddleOCR-VL Hooks — 混合 VLM OCR 适配器。

PaddleOCR-VL 推理流程:
1. Vision 阶段: NaViT Vision Encoder + PP-DocLayoutV2 (Paddle 本地)
2. LLM 阶段: ERNIE-4.5 VLM Decoder (vLLM sidecar)
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from docmill.pipelines.base import PipelineOutput


class PaddleOCRVLHooks:
    """PaddleOCR-VL 模型适配钩子。"""

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
        """构建 PaddleOCR-VL 的 VLM 请求。"""
        image_base64 = context.get("image_base64", "")

        content: list[dict[str, Any]] = []
        if image_base64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                },
            })

        # PaddleOCR-VL 标准 OCR prompt
        content.append({
            "type": "text",
            "text": "Read all the text in the image.",
        })

        return [{"role": "user", "content": content}]

    def posthandle(self, raw_output: Any) -> PipelineOutput:
        """后处理: 格式化 OCR 输出。"""
        if isinstance(raw_output, dict):
            text = raw_output.get("text", "")
            return PipelineOutput(
                text=text,
                markdown=text,
                structured=raw_output,
                metadata={"model": "paddle-ocr-vl", "pipeline": "vision_llm"},
            )
        if isinstance(raw_output, str):
            return PipelineOutput(text=raw_output, markdown=raw_output)
        return PipelineOutput(text=str(raw_output))
