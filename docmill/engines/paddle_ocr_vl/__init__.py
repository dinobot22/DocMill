"""PaddleOCR-VL Engine - 视觉语言 OCR 模型。

使用方式:
    from docmill.engines.paddle_ocr_vl import PaddleOCRVLEngine
    
    engine = PaddleOCRVLEngine(vllm_model_name="PaddleOCR-VL-0.9B")
    engine.load(vllm_endpoint="http://localhost:30023/v1")
    result = engine.infer(input_data)
"""

from docmill.engines.paddle_ocr_vl.engine import PaddleOCRVLEngine

__all__ = ["PaddleOCRVLEngine"]
