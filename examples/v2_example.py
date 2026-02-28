#!/usr/bin/env python
"""DocMill V2 使用示例。

演示如何使用新的 Engine 架构：
1. 注册模型
2. 执行推理
3. 管理生命周期

使用方式:
    pip install -e .
    python examples/v2_example.py
"""

import sys

from docmill import DocMill
from docmill.engines.base import EngineInput


def main():
    """完整使用示例。"""
    # 创建 DocMill 实例
    app = DocMill()

    try:
        # ========== 注册模型 ==========

        # 注册 PaddleOCR-VL（需要 vLLM sidecar）
        # 注意：需要先准备好 vLLM 模型
        print("注册 PaddleOCR-VL...")
        app.register_model(
            name="paddle-ocr-vl",
            engine_name="paddle_ocr_vl",  # 使用已注册的 Engine 名称
            vllm_config={
                "model_path": "/path/to/paddleocr-vl-llm",  # 替换为实际路径
                "gpu_memory_utilization": 0.8,
                "max_model_len": 4096,
            },
            vllm_model_path="/path/to/paddleocr-vl-llm",
        )

        # 注册 DeepSeek OCR（纯 vLLM）
        print("注册 DeepSeek OCR...")
        app.register_model(
            name="deepseek-ocr",
            engine_name="deepseek_ocr",  # 使用已注册的 Engine 名称
            vllm_config={
                "model_path": "deepseek-ai/DeepSeek-OCR",  # HuggingFace 模型名
                "gpu_memory_utilization": 0.9,
            },
        )

        # ========== 列出模型 ==========
        print("\n已注册的模型:")
        for model_name in app.list_models():
            info = app.get_model_info(model_name)
            print(f"  - {info['name']}: engine={info['engine']}, vllm={info['requires_vllm']}")

        # ========== 执行推理 ==========
        # 注意：需要实际的模型和图片才能执行
        # result = app.infer("deepseek-ocr", EngineInput(file_path="test.png"))
        # print(f"\nOCR 结果: {result.text}")

        print("\n✓ 示例运行成功（未执行实际推理）")

    finally:
        # 关闭 DocMill，释放所有资源
        app.shutdown()


def quick_start():
    """快速开始示例。

    最简单的使用方式：
    1. 创建 DocMill
    2. 注册模型
    3. 推理
    4. 关闭
    """
    with DocMill() as app:
        # 注册模型
        app.register_model(
            name="my-ocr",
            engine_name="deepseek_ocr",  # 使用已注册的 Engine 名称
            vllm_config={
                "model_path": "deepseek-ai/DeepSeek-OCR",
            },
        )

        # 执行推理
        # result = app.infer("my-ocr", EngineInput(file_path="test.png"))
        # print(result.text)

        print("✓ 快速开始示例")


def api_usage():
    """API 使用示例。

    通过 HTTP API 使用 DocMill。
    """
    import requests

    base_url = "http://localhost:8080"

    # 健康检查
    resp = requests.get(f"{base_url}/health")
    print(f"健康检查: {resp.json()}")

    # 列出 Engine
    resp = requests.get(f"{base_url}/models/engines/list")
    print(f"可用 Engine: {resp.json()}")

    # 注册模型
    resp = requests.post(
        f"{base_url}/models/register",
        json={
            "name": "my-ocr",
            "engine_name": "deepseek_ocr",
            "vllm_config": {
                "model_path": "deepseek-ai/DeepSeek-OCR",
            },
        },
    )
    print(f"注册模型: {resp.json()}")

    # 列出模型
    resp = requests.get(f"{base_url}/models")
    print(f"已注册模型: {resp.json()}")

    print("✓ API 使用示例")


if __name__ == "__main__":
    print("=== DocMill V2 示例 ===\n")

    # 选择要运行的示例
    example = sys.argv[1] if len(sys.argv) > 1 else "main"

    if example == "quick":
        quick_start()
    elif example == "api":
        api_usage()
    else:
        main()