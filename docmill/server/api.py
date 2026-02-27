"""DocMill FastAPI Server — 统一推理 API 入口。"""

from __future__ import annotations

import atexit
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from docmill.config.loader import discover_models, load_model_spec
from docmill.config.schema import ModelSpec
from docmill.orchestrator.pool import ModelRuntimePool
from docmill.utils.errors import (
    DocMillError,
    InsufficientVRAMError,
    ModelLoadTimeoutError,
    ModelNotFoundError,
)
from docmill.utils.logging import get_logger, setup_logging

logger = get_logger("server")

# --- 全局状态 ---
_pool: ModelRuntimePool | None = None
_model_specs: dict[str, ModelSpec] = {}


# --- Request / Response 模型 ---
class InferRequest(BaseModel):
    """推理请求。"""
    model: str = Field(description="模型名称")
    file_path: str = Field(default="", description="输入文件路径")
    text: str = Field(default="", description="输入文本")
    options: dict[str, Any] = Field(default_factory=dict, description="额外选项")


class InferResponse(BaseModel):
    """推理响应。"""
    model: str
    text: str = ""
    markdown: str = ""
    structured: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    """模型信息。"""
    name: str
    pipeline: str
    execution: str
    description: str = ""
    state: str = "cold"


class LoadRequest(BaseModel):
    """模型加载请求。"""
    model: str = Field(description="模型名称")


class EvictRequest(BaseModel):
    """模型驱逐请求。"""
    model: str = Field(description="模型名称")


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    setup_logging()
    global _pool
    _pool = ModelRuntimePool()
    atexit.register(_pool.shutdown)
    logger.info("DocMill Server 启动完成")
    yield
    if _pool:
        _pool.shutdown()
    logger.info("DocMill Server 已关闭")


# --- FastAPI App ---
app = FastAPI(
    title="DocMill",
    description="A unified inference runtime for OCR & VLM document understanding.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """服务健康检查。"""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/models", response_model=list[ModelInfo])
async def list_models():
    """列出所有已注册的模型。"""
    result: list[ModelInfo] = []

    # 已注册的 spec
    for name, spec in _model_specs.items():
        state = "cold"
        if _pool:
            runtime = _pool.registry.get(name)
            if runtime:
                state = runtime.state.value

        result.append(ModelInfo(
            name=name,
            pipeline=spec.pipeline.value,
            execution=spec.execution.value,
            description=spec.description,
            state=state,
        ))

    return result


@app.post("/models/register")
async def register_models_dir(models_dir: str):
    """注册模型目录。"""
    specs = discover_models(models_dir)
    _model_specs.update(specs)
    return {"registered": list(specs.keys()), "total": len(_model_specs)}


@app.post("/models/register_one")
async def register_model(config_path: str):
    """注册单个模型配置。"""
    try:
        spec = load_model_spec(config_path)
        _model_specs[spec.name] = spec
        return {"registered": spec.name, "pipeline": spec.pipeline.value}
    except DocMillError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/models/load")
async def load_model(req: LoadRequest):
    """预热模型（预加载到 GPU）。"""
    if req.model not in _model_specs:
        raise HTTPException(status_code=404, detail=f"模型 '{req.model}' 未注册")

    if _pool is None:
        raise HTTPException(status_code=500, detail="RuntimePool 未初始化")

    try:
        spec = _model_specs[req.model]
        _pool.get_or_load(spec)
        _pool.release(req.model)
        return {"model": req.model, "status": "loaded"}
    except ModelLoadTimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except InsufficientVRAMError as e:
        raise HTTPException(status_code=507, detail=str(e))
    except DocMillError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/evict")
async def evict_model(req: EvictRequest):
    """手动驱逐模型。"""
    if _pool is None:
        raise HTTPException(status_code=500, detail="RuntimePool 未初始化")

    _pool.evict(req.model)
    return {"model": req.model, "status": "evicted"}


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    """执行推理。"""
    if req.model not in _model_specs:
        raise HTTPException(status_code=404, detail=f"模型 '{req.model}' 未注册")

    if _pool is None:
        raise HTTPException(status_code=500, detail="RuntimePool 未初始化")

    spec = _model_specs[req.model]

    try:
        # 获取或加载 Pipeline
        pipeline = _pool.get_or_load(spec)

        # 构建输入
        from docmill.pipelines.base import PipelineInput

        input_kwargs: dict[str, Any] = {"options": req.options}
        if req.file_path:
            input_kwargs["file_path"] = req.file_path
        if req.text:
            input_kwargs["raw_text"] = req.text
        if not req.file_path and not req.text:
            input_kwargs["raw_text"] = " "  # 防止空输入

        pipeline_input = PipelineInput(**input_kwargs)

        # 执行推理
        output = pipeline.run(pipeline_input)

        # 释放
        _pool.release(req.model)

        return InferResponse(
            model=req.model,
            text=output.text,
            markdown=output.markdown,
            structured=output.structured,
            metadata=output.metadata,
        )

    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelLoadTimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except InsufficientVRAMError as e:
        raise HTTPException(status_code=507, detail=str(e))
    except DocMillError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        _pool.release(req.model)
        raise HTTPException(status_code=500, detail=f"推理失败: {e}")


@app.get("/runtimes")
async def list_runtimes():
    """列出所有活跃的运行时信息。"""
    if _pool is None:
        return []
    return _pool.list_runtimes()


def create_app(models_dir: str | Path | None = None) -> FastAPI:
    """创建并配置 FastAPI 应用。

    Args:
        models_dir: 模型目录路径，自动发现并注册模型。

    Returns:
        配置好的 FastAPI 应用。
    """
    if models_dir:
        specs = discover_models(models_dir)
        _model_specs.update(specs)
        logger.info("已注册 %d 个模型", len(specs))

    return app
