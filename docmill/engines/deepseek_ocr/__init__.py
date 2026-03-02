"""DeepSeek OCR Engine - 视觉语言 OCR 模型。

使用方式:
    from docmill.engines.deepseek_ocr import DeepSeekOCREngine
    
    engine = DeepSeekOCREngine(vllm_model_path="deepseek-ai/DeepSeek-OCR")
    engine.load(vllm_endpoint="http://localhost:30023/v1")
    result = engine.infer(input_data)
"""

from docmill.engines.deepseek_ocr.engine import DeepSeekOCREngine

__all__ = ["DeepSeekOCREngine"]
