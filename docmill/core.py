"""DocMill 核心 — 统一 OCR 推理运行时。

DocMill 是 OCR 模型的统一管理入口：
- 注册和管理多个 OCR 模型
- 自动管理 vLLM sidecar 生命周期
- GPU 显存感知调度
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Type

from docmill.engines.base import BaseEngine, EngineInput, EngineOutput
from docmill.engines.registry import EngineRegistry
from docmill.orchestrator.launcher import SidecarLauncher
from docmill.orchestrator.sidecar_pool import SidecarPool
from docmill.utils.logging import get_logger

logger = get_logger("core")


@dataclass
class ModelEntry:
    """已注册模型的信息。"""

    name: str
    engine: BaseEngine
    vllm_config: dict[str, Any] | None = None
    vllm_endpoint: str = ""
    status: str = "cold"  # cold, loading, ready, idle


class DocMill:
    """DocMill 核心。

    统一管理多个 OCR 模型的生命周期和推理。

    使用示例:
        from docmill import DocMill
        from docmill.engines import PaddleOCRVLEngine, DeepSeekOCREngine

        app = DocMill()

        # 注册模型
        app.register_model(
            name="paddle-ocr-vl",
            engine_class=PaddleOCRVLEngine,
            vllm_config={
                "model_path": "/models/paddleocr-vl-llm",
                "gpu_memory_utilization": 0.8,
            },
            vllm_model_path="/models/paddleocr-vl-llm",
        )

        # 推理
        result = app.infer("paddle-ocr-vl", EngineInput(file_path="test.png"))
        print(result.text)

        # 关闭
        app.shutdown()
    """

    def __init__(
        self,
        sidecar_log_dir: str | Path = "/tmp/docmill/sidecar_logs",
        default_gpu_id: int = 0,
    ):
        """初始化 DocMill。

        Args:
            sidecar_log_dir: vLLM sidecar 日志目录。
            default_gpu_id: 默认 GPU 设备 ID。
        """
        self._models: dict[str, ModelEntry] = {}
        self._sidecar_pool = SidecarPool(log_dir=sidecar_log_dir)
        self._default_gpu_id = default_gpu_id
        logger.info("DocMill 初始化完成")

    # ========== 模型注册 ==========

    def register_model(
        self,
        name: str,
        engine_class: Type[BaseEngine] | None = None,
        engine_name: str | None = None,
        vllm_config: dict[str, Any] | None = None,
        **engine_kwargs,
    ) -> None:
        """注册 OCR 模型。

        Args:
            name: 模型名称（唯一标识符）。
            engine_class: Engine 类（与 engine_name 二选一）。
            engine_name: 已注册的 Engine 名称（与 engine_class 二选一）。
            vllm_config: vLLM sidecar 配置（如果 Engine 需要）。
                - model_path: vLLM 模型路径（必需）
                - gpu_id: GPU 设备 ID（默认使用 default_gpu_id）
                - gpu_memory_utilization: 显存使用率（默认 0.8）
                - max_model_len: 最大序列长度（默认 4096）
                - tensor_parallel_size: 张量并行大小（默认 1）
                - extra_args: 额外启动参数
                - served_model_name: vLLM 服务注册的模型名称
            **engine_kwargs: Engine 初始化参数。

        Raises:
            ValueError: 参数错误。
        """
        if name in self._models:
            logger.warning("模型 '%s' 已存在，将被覆盖", name)

        # 获取 Engine 类
        if engine_class is not None:
            pass
        elif engine_name is not None:
            engine_class = EngineRegistry.get_or_raise(engine_name)
        else:
            raise ValueError("必须提供 engine_class 或 engine_name")

        # 实例化 Engine
        engine = engine_class(**engine_kwargs)

        # 验证 vLLM 配置
        if engine.requires_vllm_sidecar():
            if vllm_config is None:
                raise ValueError(f"Engine '{name}' 需要 vLLM sidecar，请提供 vllm_config")
            if "model_path" not in vllm_config:
                raise ValueError("vllm_config 必须包含 'model_path'")

        # 注册
        self._models[name] = ModelEntry(
            name=name,
            engine=engine,
            vllm_config=vllm_config,
        )
        logger.info("注册模型: %s (engine=%s, vllm=%s)", name, engine_class.engine_name(), vllm_config is not None)

    def unregister_model(self, name: str) -> None:
        """注销模型。

        Args:
            name: 模型名称。
        """
        if name not in self._models:
            logger.warning("模型 '%s' 不存在", name)
            return

        entry = self._models[name]

        # 卸载 Engine
        if entry.engine.is_loaded():
            entry.engine.unload()

        # 释放 vLLM sidecar
        if entry.vllm_endpoint:
            self._sidecar_pool.release(entry.vllm_endpoint)

        del self._models[name]
        logger.info("注销模型: %s", name)

    def list_models(self) -> list[str]:
        """列出所有已注册的模型。"""
        return list(self._models.keys())

    def get_model_info(self, name: str) -> dict[str, Any]:
        """获取模型信息。

        Args:
            name: 模型名称。

        Returns:
            模型信息字典。
        """
        if name not in self._models:
            raise KeyError(f"模型 '{name}' 未注册")

        entry = self._models[name]
        return {
            "name": entry.name,
            "engine": entry.engine.engine_name(),
            "requires_vllm": entry.engine.requires_vllm_sidecar(),
            "vllm_endpoint": entry.vllm_endpoint,
            "status": entry.status,
            "is_loaded": entry.engine.is_loaded(),
            "vram_estimate_mb": entry.engine.estimate_vram_mb(),
        }

    # ========== 模型生命周期 ==========

    def ensure_model_ready(self, name: str) -> str:
        """确保模型就绪。

        如果模型未加载，会自动：
        1. 启动 vLLM sidecar（如果需要）
        2. 加载 Engine

        Args:
            name: 模型名称。

        Returns:
            vLLM endpoint（如果有），否则返回空字符串。

        Raises:
            RuntimeError: 加载失败。
        """
        if name not in self._models:
            raise KeyError(f"模型 '{name}' 未注册")

        entry = self._models[name]

        # 已经就绪
        if entry.engine.is_loaded() and entry.status == "ready":
            return entry.vllm_endpoint

        entry.status = "loading"

        try:
            # 启动 vLLM sidecar
            if entry.engine.requires_vllm_sidecar():
                vllm_endpoint = self._start_vllm_sidecar(name, entry.vllm_config)
                entry.vllm_endpoint = vllm_endpoint

                # 加载 Engine
                entry.engine.load(vllm_endpoint=vllm_endpoint)
            else:
                # 无需 sidecar，直接加载
                entry.engine.load()

            entry.status = "ready"
            logger.info("模型 '%s' 已就绪", name)
            return entry.vllm_endpoint

        except Exception as e:
            entry.status = "cold"
            logger.error("模型 '%s' 加载失败: %s", name, e)
            raise RuntimeError(f"模型 '{name}' 加载失败: {e}") from e

    def _start_vllm_sidecar(self, model_name: str, vllm_config: dict[str, Any]) -> str:
        """启动 vLLM sidecar。

        Args:
            model_name: 模型名称（用于日志）。
            vllm_config: vLLM 配置。

        Returns:
            vLLM endpoint URL。
        """
        model_path = vllm_config["model_path"]
        gpu_id = vllm_config.get("gpu_id", self._default_gpu_id)
        gpu_memory_utilization = vllm_config.get("gpu_memory_utilization", 0.8)
        max_model_len = vllm_config.get("max_model_len", 4096)
        tensor_parallel_size = vllm_config.get("tensor_parallel_size", 1)
        trust_remote_code = vllm_config.get("trust_remote_code", True)
        extra_args = vllm_config.get("extra_args", [])
        served_model_name = vllm_config.get("served_model_name")

        logger.info("启动 vLLM sidecar: model=%s, gpu=%d, served_name=%s",
                   model_path, gpu_id, served_model_name)

        sidecar = self._sidecar_pool.acquire(
            model_path=model_path,
            gpu_id=gpu_id,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            extra_args=extra_args,
            served_model_name=served_model_name,
        )

        return sidecar.endpoint

    def unload_model(self, name: str) -> None:
        """卸载模型。

        Args:
            name: 模型名称。
        """
        if name not in self._models:
            return

        entry = self._models[name]

        # 卸载 Engine
        if entry.engine.is_loaded():
            entry.engine.unload()

        # 释放 vLLM sidecar
        if entry.vllm_endpoint:
            self._sidecar_pool.release(entry.vllm_endpoint)
            entry.vllm_endpoint = ""

        entry.status = "cold"
        logger.info("模型 '%s' 已卸载", name)

    # ========== 推理 ==========

    def infer(
        self,
        model: str,
        input_data: EngineInput | str | Path,
        **kwargs,
    ) -> EngineOutput:
        """执行 OCR 推理。

        Args:
            model: 模型名称。
            input_data: 输入数据，可以是 EngineInput、文件路径字符串或 Path 对象。
            **kwargs: 额外选项，会合并到 input_data.options 中。

        Returns:
            OCR 结果。

        Raises:
            KeyError: 模型未注册。
            RuntimeError: 推理失败。
        """
        # 确保模型就绪
        self.ensure_model_ready(model)

        # 转换输入
        if isinstance(input_data, (str, Path)):
            input_data = EngineInput(file_path=str(input_data), options=kwargs)
        elif isinstance(input_data, EngineInput):
            if kwargs:
                input_data.options.update(kwargs)
        else:
            raise TypeError(f"input_data 类型错误: {type(input_data)}")

        # 执行推理
        entry = self._models[model]
        logger.debug("执行推理: model=%s", model)

        try:
            return entry.engine.infer(input_data)
        except Exception as e:
            logger.error("推理失败: model=%s, error=%s", model, e)
            raise RuntimeError(f"推理失败: {e}") from e

    def infer_batch(
        self,
        model: str,
        inputs: list[EngineInput | str | Path],
        **kwargs,
    ) -> list[EngineOutput]:
        """批量推理。

        Args:
            model: 模型名称。
            inputs: 输入数据列表。
            **kwargs: 额外选项。

        Returns:
            结果列表。
        """
        results = []
        for input_data in inputs:
            result = self.infer(model, input_data, **kwargs)
            results.append(result)
        return results

    # ========== 健康检查 ==========

    def health_check(self, model: str) -> bool:
        """检查模型健康状态。

        Args:
            model: 模型名称。

        Returns:
            True 表示健康。
        """
        if model not in self._models:
            return False

        entry = self._models[model]
        return entry.engine.is_loaded() and entry.status == "ready"

    def health_check_all(self) -> dict[str, bool]:
        """检查所有模型健康状态。

        Returns:
            模型名称 -> 健康状态字典。
        """
        return {name: self.health_check(name) for name in self._models}

    # ========== 生命周期管理 ==========

    def shutdown(self) -> None:
        """关闭 DocMill，释放所有资源。"""
        logger.info("关闭 DocMill...")

        # 卸载所有模型
        for name in list(self._models.keys()):
            try:
                self.unload_model(name)
            except Exception as e:
                logger.warning("卸载模型 '%s' 失败: %s", name, e)

        # 关闭所有 sidecar
        self._sidecar_pool.shutdown()

        logger.info("DocMill 已关闭")

    def __enter__(self) -> "DocMill":
        """上下文管理器入口。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出。"""
        self.shutdown()