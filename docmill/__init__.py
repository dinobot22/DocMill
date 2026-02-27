"""DocMill - A unified inference runtime for OCR & VLM document understanding."""

from docmill.core import DocMill
from docmill.engines.base import BaseEngine, EngineInput, EngineOutput
from docmill.engines.registry import EngineRegistry

__version__ = "0.2.0"

__all__ = [
    "DocMill",
    "BaseEngine",
    "EngineInput",
    "EngineOutput",
    "EngineRegistry",
]