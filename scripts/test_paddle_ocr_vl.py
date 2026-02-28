#!/usr/bin/env python
"""PaddleOCR-VL Engine 独立测试脚本。

测试步骤：
1. 启动 vLLM sidecar 服务
2. 加载 PaddleOCR-VL Engine
3. 执行 OCR 推理
4. 验证结果

使用方式:
    python scripts/test_paddle_ocr_vl.py --image test.png

环境要求:
    - paddleocr 已安装
    - vllm 已安装
    - PaddleOCR-VL 模型已下载到 ~/.paddlex/official_models/PaddleOCR-VL
"""

import argparse
import sys
import time
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from docmill.engines.paddle_ocr_vl import PaddleOCRVLEngine
from docmill.engines.base import EngineInput
from docmill.orchestrator.launcher import SidecarLauncher
from docmill.utils.logging import get_logger

logger = get_logger("test_paddle_ocr_vl")

# 默认配置
DEFAULT_VLLM_MODEL_PATH = Path.home() / ".paddlex" / "official_models" / "PaddleOCR-VL"
DEFAULT_VLLM_PORT = 30023
DEFAULT_GPU_ID = 5


def test_vllm_sidecar_standalone():
    """测试独立启动 vLLM sidecar（不通过 DocMill 框架）。

    这个测试验证：
    1. vLLM 能否正常启动
    2. PaddleOCR-VL 模型能否被加载
    """
    print("\n" + "=" * 60)
    print("测试 1: 独立启动 vLLM sidecar")
    print("=" * 60)

    model_path = str(DEFAULT_VLLM_MODEL_PATH)
    if not Path(model_path).exists():
        print(f"错误: 模型路径不存在: {model_path}")
        print("请先下载 PaddleOCR-VL 模型")
        return False

    launcher = SidecarLauncher()

    try:
        # 启动 vLLM
        print(f"\n启动 vLLM sidecar...")
        print(f"  模型路径: {model_path}")
        print(f"  端口: {DEFAULT_VLLM_PORT}")
        print(f"  GPU: {DEFAULT_GPU_ID}")

        sidecar = launcher.launch(
            model_path=model_path,
            port=DEFAULT_VLLM_PORT,
            gpu_id=DEFAULT_GPU_ID,
            gpu_memory_utilization=0.9,
            max_model_len=16384,
            trust_remote_code=True,
            extra_args=[
                "--no-enable-prefix-caching",
                "--mm-processor-cache-gb", "0",
                "--enforce-eager",
            ],
        )

        print(f"\nvLLM sidecar 已启动:")
        print(f"  PID: {sidecar.pid}")
        print(f"  Endpoint: {sidecar.endpoint}")
        print(f"  Log: {sidecar.log_file}")

        # 等待就绪
        print("\n等待 vLLM 就绪...")
        wait_for_vllm_ready(sidecar.endpoint, timeout=300)

        print("\n✓ vLLM sidecar 启动成功!")

        # 保持运行
        print("\n按 Ctrl+C 停止服务...")
        while sidecar.is_alive:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n收到中断信号，停止服务...")
    except Exception as e:
        print(f"\n错误: {e}")
        return False
    finally:
        launcher.stop_all()

    return True


def test_paddle_ocr_vl_engine(image_path: str, vllm_endpoint: str = None):
    """测试 PaddleOCR-VL Engine。

    Args:
        image_path: 测试图片路径
        vllm_endpoint: vLLM 服务地址（如果 None，需要手动启动）
    """
    print("\n" + "=" * 60)
    print("测试 2: PaddleOCR-VL Engine")
    print("=" * 60)

    # 检查图片
    if not Path(image_path).exists():
        print(f"错误: 图片不存在: {image_path}")
        return False

    # 确定 endpoint
    if vllm_endpoint is None:
        vllm_endpoint = f"http://127.0.0.1:{DEFAULT_VLLM_PORT}/v1"

    print(f"\n配置:")
    print(f"  vLLM endpoint: {vllm_endpoint}")
    print(f"  图片: {image_path}")

    # 创建 Engine
    engine = PaddleOCRVLEngine(
        vllm_endpoint=vllm_endpoint,
        vllm_model_name="PaddleOCR-VL-0.9B",
    )

    try:
        # 加载
        print("\n加载 PaddleOCR-VL Engine...")
        engine.load()
        print("✓ Engine 加载成功")

        # 推理
        print("\n执行 OCR 推理...")
        input_data = EngineInput(file_path=image_path)
        result = engine.infer(input_data)

        # 输出结果
        print("\n" + "-" * 40)
        print("OCR 结果:")
        print("-" * 40)
        print(result.text[:500] if len(result.text) > 500 else result.text)
        print("-" * 40)
        print(f"\n元数据: {result.metadata}")

        print("\n✓ PaddleOCR-VL Engine 测试成功!")
        return True

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        engine.unload()


def test_full_workflow(image_path: str):
    """测试完整工作流：启动 vLLM + 加载 Engine + 推理。"""
    print("\n" + "=" * 60)
    print("测试 3: 完整工作流")
    print("=" * 60)

    model_path = str(DEFAULT_VLLM_MODEL_PATH)
    if not Path(model_path).exists():
        print(f"错误: 模型路径不存在: {model_path}")
        return False

    if not Path(image_path).exists():
        print(f"错误: 图片不存在: {image_path}")
        return False

    launcher = SidecarLauncher()

    try:
        # 1. 启动 vLLM
        print("\n[1/4] 启动 vLLM sidecar...")
        sidecar = launcher.launch(
            model_path=model_path,
            port=DEFAULT_VLLM_PORT,
            gpu_id=DEFAULT_GPU_ID,
            gpu_memory_utilization=0.9,
            max_model_len=16384,
            trust_remote_code=True,
            extra_args=[
                "--no-enable-prefix-caching",
                "--mm-processor-cache-gb", "0",
                "--enforce-eager",
            ],
        )
        print(f"  PID: {sidecar.pid}")
        print(f"  Endpoint: {sidecar.endpoint}")

        # 2. 等待就绪
        print("\n[2/4] 等待 vLLM 就绪...")
        wait_for_vllm_ready(sidecar.endpoint, timeout=300)

        # 3. 加载 Engine 并推理
        print("\n[3/4] 加载 Engine 并推理...")
        engine = PaddleOCRVLEngine(
            vllm_endpoint=sidecar.endpoint,
            vllm_model_name="PaddleOCR-VL-0.9B",
        )
        engine.load()

        input_data = EngineInput(file_path=image_path)
        result = engine.infer(input_data)

        # 4. 输出结果
        print("\n[4/4] OCR 结果:")
        print("-" * 40)
        print(result.text[:500] if len(result.text) > 500 else result.text)
        print("-" * 40)

        engine.unload()

        print("\n✓ 完整工作流测试成功!")
        return True

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("\n清理资源...")
        launcher.stop_all()


def test_paddle_ocr_vl_direct(image_path: str):
    """直接测试 paddleocr SDK（不通过 Engine 封装）。

    这用于验证 paddleocr 是否能正常工作。
    """
    print("\n" + "=" * 60)
    print("测试 4: 直接使用 paddleocr SDK")
    print("=" * 60)

    try:
        from paddleocr import PaddleOCRVL

        print("\n初始化 PaddleOCRVL...")
        pipeline = PaddleOCRVL()

        print(f"\n处理图片: {image_path}")
        output = pipeline.predict(input=image_path)

        pages_res = list(output)
        print(f"\n获取到 {len(pages_res)} 页结果")

        # 尝试重组
        print("\n重组页面...")
        restructured = pipeline.restructure_pages(pages_res)

        for i, res in enumerate(restructured):
            print(f"\n--- 页面 {i+1} ---")
            if hasattr(res, 'text'):
                print(res.text[:300] if len(res.text) > 300 else res.text)

        print("\n✓ paddleocr SDK 测试成功!")
        return True

    except ImportError as e:
        print(f"错误: paddleocr 未安装: {e}")
        return False
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def wait_for_vllm_ready(endpoint: str, timeout: float = 300.0):
    """等待 vLLM 服务就绪。"""
    import httpx

    health_url = endpoint.replace("/v1", "/health")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            resp = httpx.get(health_url, timeout=5.0)
            if resp.status_code == 200:
                print(f"  ✓ vLLM 就绪 (耗时 {time.time() - start_time:.1f}s)")
                return
        except Exception:
            pass

        print(f"  等待中... ({time.time() - start_time:.0f}s)")
        time.sleep(5.0)

    raise RuntimeError(f"vLLM 启动超时 ({timeout}s)")


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR-VL Engine 测试")
    parser.add_argument("--image", type=str, help="测试图片路径")
    parser.add_argument(
        "--test",
        choices=["vllm", "engine", "full", "sdk"],
        default="full",
        help="测试类型: vllm(仅启动vLLM), engine(使用已有vLLM), full(完整流程), sdk(直接测试SDK)"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_VLLM_PORT, help="vLLM 端口")
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU_ID, help="GPU ID")

    args = parser.parse_args()

    # 设置 GPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("=" * 60)
    print("PaddleOCR-VL Engine 测试")
    print("=" * 60)
    print(f"GPU: {args.gpu}")
    print(f"端口: {args.port}")
    print(f"模型: {DEFAULT_VLLM_MODEL_PATH}")

    if args.test == "vllm":
        test_vllm_sidecar_standalone()

    elif args.test == "engine":
        if not args.image:
            print("错误: --image 参数必需")
            sys.exit(1)
        endpoint = f"http://127.0.0.1:{args.port}/v1"
        test_paddle_ocr_vl_engine(args.image, endpoint)

    elif args.test == "full":
        if not args.image:
            print("错误: --image 参数必需")
            sys.exit(1)
        test_full_workflow(args.image)

    elif args.test == "sdk":
        if not args.image:
            print("错误: --image 参数必需")
            sys.exit(1)
        test_paddle_ocr_vl_direct(args.image)


if __name__ == "__main__":
    main()