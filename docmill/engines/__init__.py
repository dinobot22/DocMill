"""Engines — OCR 模型执行引擎。

每个 Engine 封装一个具体的 OCR 模型：
- 模型加载、推理、卸载
- 声明是否需要 vLLM sidecar
- 估算显存需求
"""

from docmill.engines.base import BaseEngine, EngineInput, EngineOutput
from docmill.engines.registry import EngineRegistry
from docmill.engines.paddle_ocr_vl import PaddleOCRVLEngine
from docmill.engines.deepseek_ocr import DeepSeekOCREngine

__all__ = [
    "BaseEngine",
    "EngineInput",
    "EngineOutput",
    "EngineRegistry",
    "PaddleOCRVLEngine",
    "DeepSeekOCREngine",
]