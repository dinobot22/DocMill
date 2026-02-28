#!/usr/bin/env python
"""PaddleOCR-VL 实际测试示例。

这个示例演示如何使用 DocMill 运行 PaddleOCR-VL 模型。

前置条件：
1. 安装依赖：pip install paddleocr vllm
2. 下载模型：模型会自动下载到 ~/.paddlex/official_models/PaddleOCR-VL

使用方式：
    # 使用 DocMill 框架
    python examples/test_paddle_ocr_vl_example.py --image test.png

    # 直接使用 paddleocr SDK
    python examples/test_paddle_ocr_vl_example.py --image test.png --direct
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 模型路径
PADDLEOCR_VL_MODEL_PATH = str(Path.home() / ".paddlex" / "official_models" / "PaddleOCR-VL")


def test_with_docmill(image_path: str, gpu_id: int = 5):
    """使用 DocMill 框架测试 PaddleOCR-VL。

    这个方法会自动管理 vLLM sidecar 的生命周期。
    """
    from docmill import DocMill
    from docmill.engines.base import EngineInput

    print("\n" + "=" * 60)
    print("使用 DocMill 框架测试 PaddleOCR-VL")
    print("=" * 60)

    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 创建 DocMill 实例
    with DocMill() as app:
        # 注册 PaddleOCR-VL 模型
        print("\n[1/3] 注册 PaddleOCR-VL 模型...")
        app.register_model(
            name="paddle-ocr-vl",
            engine_name="paddle_ocr_vl",
            vllm_config={
                "model_path": PADDLEOCR_VL_MODEL_PATH,
                "gpu_id": gpu_id,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 16384,
                "extra_args": [
                    "--no-enable-prefix-caching",
                    "--mm-processor-cache-gb", "0",
                    "--enforce-eager",
                ],
            },
            vllm_model_name="PaddleOCR-VL-0.9B",
        )
        print("  ✓ 注册成功")

        # 列出模型
        print("\n[2/3] 已注册的模型:")
        for model_name in app.list_models():
            info = app.get_model_info(model_name)
            print(f"  - {info['name']}: engine={info['engine']}, requires_vllm={info['requires_vllm']}")

        # 执行推理
        print("\n[3/3] 执行 OCR 推理...")
        print(f"  图片: {image_path}")

        result = app.infer("paddle-ocr-vl", EngineInput(file_path=image_path))

        # 输出结果
        print("\n" + "-" * 40)
        print("OCR 结果:")
        print("-" * 40)
        if result.markdown:
            print(result.markdown[:1000])
        else:
            print(result.text[:1000] if result.text else "无文本结果")
        print("-" * 40)
        print(f"\n元数据: {result.metadata}")

    print("\n✓ DocMill 测试完成")


def test_direct_sdk(image_path: str):
    """直接使用 paddleocr SDK 测试（不需要 vLLM sidecar）。

    这个方法用于验证 paddleocr 本身是否工作正常。
    """
    print("\n" + "=" * 60)
    print("直接使用 paddleocr SDK")
    print("=" * 60)

    try:
        from paddleocr import PaddleOCRVL

        print("\n[1/2] 初始化 PaddleOCRVL...")
        pipeline = PaddleOCRVL()
        print("  ✓ 初始化成功")

        print(f"\n[2/2] 处理图片: {image_path}")
        output = pipeline.predict(input=image_path)

        pages_res = list(output)
        print(f"  获取到 {len(pages_res)} 页结果")

        # 尝试重组
        print("\n重组页面结构...")
        restructured = pipeline.restructure_pages(pages_res)

        print("\n" + "-" * 40)
        print("OCR 结果:")
        print("-" * 40)

        for i, res in enumerate(restructured):
            print(f"\n=== 页面 {i+1} ===")
            if hasattr(res, 'text'):
                print(res.text[:500] if len(res.text) > 500 else res.text)
            if hasattr(res, 'save_to_markdown'):
                res.save_to_markdown(save_path="/tmp/docmill_output")
                print(f"\n  已保存 Markdown 到 /tmp/docmill_output")

        print("\n✓ paddleocr SDK 测试完成")

    except ImportError as e:
        print(f"错误: paddleocr 未安装")
        print(f"请执行: pip install paddleocr")
        return False
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PaddleOCR-VL 测试示例")
    parser.add_argument("--image", type=str, required=True, help="测试图片路径")
    parser.add_argument("--direct", action="store_true", help="直接使用 paddleocr SDK（不通过 DocMill）")
    parser.add_argument("--gpu", type=int, default=5, help="GPU ID")

    args = parser.parse_args()

    # 检查图片
    if not Path(args.image).exists():
        print(f"错误: 图片不存在: {args.image}")
        sys.exit(1)

    # 检查模型
    if not Path(PADDLEOCR_VL_MODEL_PATH).exists():
        print(f"错误: 模型不存在: {PADDLEOCR_VL_MODEL_PATH}")
        print("请先下载 PaddleOCR-VL 模型")
        sys.exit(1)

    print("=" * 60)
    print("PaddleOCR-VL 测试示例")
    print("=" * 60)
    print(f"模型路径: {PADDLEOCR_VL_MODEL_PATH}")
    print(f"测试图片: {args.image}")

    if args.direct:
        test_direct_sdk(args.image)
    else:
        test_with_docmill(args.image, args.gpu)


if __name__ == "__main__":
    main()