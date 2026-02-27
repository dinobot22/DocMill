"""DocMill FastAPI Server V2 — 基于 Engine 架构的统一推理 API。

新版 API 使用 Engine 架构：
- 代码注册模型（非配置文件）
- 自动管理 vLLM sidecar
- 更简洁的接口
"""

from __future__ import annotations

import atexit
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from docmill.core import DocMill
from docmill.engines.base import EngineInput, EngineOutput
from docmill.engines.registry import EngineRegistry
from docmill.utils.logging import get_logger, setup_logging

logger = get_logger("server.v2")

# --- 全局状态 ---
_docmill: DocMill | None = None


# --- Request / Response 模型 ---
class RegisterModelRequest(BaseModel):
    """注册模型请求。"""

    name: str = Field(description="模型名称（唯一标识符）")
    engine_name: str | None = Field(default=None, description="已注册的 Engine 名称")
    vllm_config: dict[str, Any] | None = Field(default=None, description="vLLM sidecar 配置")
    engine_kwargs: dict[str, Any] = Field(default_factory=dict, description="Engine 初始化参数")


class InferRequest(BaseModel):
    """推理请求。"""

    model: str = Field(description="模型名称")
    file_path: str | None = Field(default=None, description="输入文件路径")
    image_bytes: str | None = Field(default=None, description="Base64 编码的图片数据")
    url: str | None = Field(default=None, description="图片 URL")
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
    engine: str
    requires_vllm: bool
    vllm_endpoint: str = ""
    status: str
    is_loaded: bool
    vram_estimate_mb: float


class EngineInfo(BaseModel):
    """Engine 信息。"""

    name: str
    requires_vllm: bool


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    setup_logging()
    global _docmill
    _docmill = DocMill()
    atexit.register(_cleanup)
    logger.info("DocMill Server V2 启动完成")
    yield
    _cleanup()
    logger.info("DocMill Server V2 已关闭")


def _cleanup():
    """清理资源。"""
    global _docmill
    if _docmill:
        _docmill.shutdown()
        _docmill = None


# --- FastAPI App ---
app = FastAPI(
    title="DocMill V2",
    description="A unified inference runtime for OCR & VLM document understanding (Engine-based).",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """服务健康检查。"""
    return {"status": "ok", "version": "0.2.0"}


# --- Engine 相关 ---
@app.get("/engines", response_model=list[EngineInfo])
async def list_engines():
    """列出所有已注册的 Engine 类型。"""
    result = []
    for name in EngineRegistry.list_engines():
        engine_class = EngineRegistry.get(name)
        if engine_class:
            result.append(EngineInfo(
                name=name,
                requires_vllm=engine_class.requires_vllm_sidecar(),
            ))
    return result


# --- 模型相关 ---
@app.get("/models", response_model=list[ModelInfo])
async def list_models():
    """列出所有已注册的模型。"""
    if _docmill is None:
        raise HTTPException(status_code=500, detail="DocMill 未初始化")

    result = []
    for name in _docmill.list_models():
        try:
            info = _docmill.get_model_info(name)
            result.append(ModelInfo(**info))
        except Exception as e:
            logger.warning("获取模型信息失败: %s - %s", name, e)

    return result


@app.post("/models/register")
async def register_model(req: RegisterModelRequest):
    """注册模型。

    示例:
        # PaddleOCR-VL
        {
            "name": "paddle-ocr-vl",
            "engine_name": "paddle_ocr_vl",
            "vllm_config": {
                "model_path": "/models/paddleocr-vl-llm",
                "gpu_memory_utilization": 0.8
            },
            "engine_kwargs": {
                "vllm_model_path": "/models/paddleocr-vl-llm"
            }
        }

        # DeepSeek OCR
        {
            "name": "deepseek-ocr",
            "engine_name": "deepseek_ocr",
            "vllm_config": {
                "model_path": "deepseek-ai/DeepSeek-OCR"
            }
        }
    """
    if _docmill is None:
        raise HTTPException(status_code=500, detail="DocMill 未初始化")

    try:
        _docmill.register_model(
            name=req.name,
            engine_name=req.engine_name,
            vllm_config=req.vllm_config,
            **req.engine_kwargs,
        )
        return {"status": "registered", "model": req.name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"注册失败: {e}")


@app.post("/models/unregister")
async def unregister_model(model: str):
    """注销模型。"""
    if _docmill is None:
        raise HTTPException(status_code=500, detail="DocMill 未初始化")

    try:
        _docmill.unregister_model(model)
        return {"status": "unregistered", "model": model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"注销失败: {e}")


@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """获取模型详情。"""
    if _docmill is None:
        raise HTTPException(status_code=500, detail="DocMill 未初始化")

    try:
        info = _docmill.get_model_info(model_name)
        return ModelInfo(**info)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"模型 '{model_name}' 未注册")


@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """预热模型（预加载）。"""
    if _docmill is None:
        raise HTTPException(status_code=500, detail="DocMill 未初始化")

    try:
        _docmill.ensure_model_ready(model_name)
        return {"status": "loaded", "model": model_name}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"模型 '{model_name}' 未注册")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_name}/unload")
async def unload_model(model_name: str):
    """卸载模型。"""
    if _docmill is None:
        raise HTTPException(status_code=500, detail="DocMill 未初始化")

    try:
        _docmill.unload_model(model_name)
        return {"status": "unloaded", "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"卸载失败: {e}")


# --- 推理相关 ---
@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    """执行 OCR 推理。

    支持三种输入方式：
    - file_path: 本地文件路径
    - image_bytes: Base64 编码的图片数据
    - url: 图片 URL
    """
    if _docmill is None:
        raise HTTPException(status_code=500, detail="DocMill 未初始化")

    # 构建输入
    input_kwargs: dict[str, Any] = {"options": req.options}

    if req.file_path:
        input_kwargs["file_path"] = req.file_path
    elif req.image_bytes:
        import base64
        try:
            input_kwargs["image_bytes"] = base64.b64decode(req.image_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Base64 解码失败: {e}")
    elif req.url:
        input_kwargs["url"] = req.url
    else:
        raise HTTPException(status_code=400, detail="必须提供 file_path、image_bytes 或 url")

    engine_input = EngineInput(**input_kwargs)

    # 执行推理
    try:
        output = _docmill.infer(req.model, engine_input)
        return InferResponse(
            model=req.model,
            text=output.text,
            markdown=output.markdown,
            structured=output.structured,
            metadata=output.metadata,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"模型 '{req.model}' 未注册")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("推理失败: %s", e)
        raise HTTPException(status_code=500, detail=f"推理失败: {e}")


@app.post("/infer/{model_name}", response_model=InferResponse)
async def infer_with_model(model_name: str, req: InferRequest):
    """执行推理（模型名在 URL 中）。"""
    req.model = model_name
    return await infer(req)


# --- 健康检查 ---
@app.get("/health/models")
async def health_check_models():
    """检查所有模型健康状态。"""
    if _docmill is None:
        raise HTTPException(status_code=500, detail="DocMill 未初始化")

    return _docmill.health_check_all()


def create_app(
    models_config: list[dict[str, Any]] | None = None,
    **kwargs,
) -> FastAPI:
    """创建并配置 FastAPI 应用。

    Args:
        models_config: 模型配置列表，每个元素是 register_model 的参数。
        **kwargs: DocMill 初始化参数。

    Returns:
        配置好的 FastAPI 应用。

    Example:
        app = create_app(models_config=[
            {
                "name": "paddle-ocr-vl",
                "engine_name": "paddle_ocr_vl",
                "vllm_config": {
                    "model_path": "/models/paddleocr-vl-llm",
                },
            },
        ])
    """
    # 在 lifespan 中注册模型
    if models_config:

        @app.on_event("startup")
        async def register_models():
            global _docmill
            if _docmill:
                for config in models_config:
                    try:
                        _docmill.register_model(**config)
                        logger.info("已注册模型: %s", config.get("name"))
                    except Exception as e:
                        logger.error("注册模型失败: %s - %s", config.get("name"), e)

    return app