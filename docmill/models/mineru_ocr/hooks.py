"""MinerU OCR Hooks — 两阶段推理适配器。

MinerU 的推理流程:
1. Vision 阶段: Layout 分析 + 区域裁剪 (本地 Worker)
2. LLM 阶段: VLM 精细识别 (vLLM sidecar)

Hooks 只负责数据适配，不决定流程拓扑。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from docmill.pipelines.base import PipelineOutput


class MinerUOCRHooks:
    """MinerU OCR 模型适配钩子。"""

    def prehandle(self, raw_input: Any) -> Any:
        """预处理: 准备图片数据。

        支持 file_path 和 image_bytes 两种输入。
        """
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
        """构建 VLM OCR prompt。

        MinerU 两阶段: 先获取 layout/区域信息, 再交给 VLM 识别。
        """
        image_base64 = context.get("image_base64", "")
        text_hint = context.get("text", "")

        content: list[dict[str, Any]] = []

        # 图片输入
        if image_base64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                },
            })

        # OCR 指令
        prompt = "请识别这张图片中的所有文字内容，保持原始排版格式，输出为 Markdown 格式。"
        if text_hint:
            prompt += f"\n补充信息: {text_hint}"

        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def posthandle(self, raw_output: Any) -> PipelineOutput:
        """后处理: 提取并格式化 OCR 结果。"""
        if isinstance(raw_output, dict):
            text = raw_output.get("text", "")
            return PipelineOutput(
                text=text,
                markdown=text,
                structured=raw_output,
                metadata={"model": "mineru-ocr", "pipeline": "vision_llm"},
            )
        if isinstance(raw_output, str):
            return PipelineOutput(text=raw_output, markdown=raw_output)
        return PipelineOutput(text=str(raw_output))
