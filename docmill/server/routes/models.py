"""模型管理 API 路由"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from docmill.core import DocMill
from docmill.engines.registry import EngineRegistry
from docmill.utils.logging import get_logger

logger = get_logger("server.routes.models")

router = APIRouter(prefix="/models", tags=["models"])

# 全局 DocMill 实例（由 main.py 注入）
_docmill: DocMill | None = None


def set_docmill(docmill: DocMill) -> None:
    """设置 DocMill 实例。"""
    global _docmill
    _docmill = docmill


# --- Request / Response 模型 ---


class RegisterModelRequest(BaseModel):
    """注册模型请求。"""

    name: str = Field(description="模型名称（唯一标识符）")
    engine_name: str | None = Field(default=None, description="已注册的 Engine 名称")
    vllm_config: dict[str, Any] | None = Field(default=None, description="vLLM sidecar 配置")
    engine_kwargs: dict[str, Any] = Field(default_factory=dict, description="Engine 初始化参数")


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


# --- API 端点 ---


@router.get("", response_model=list[ModelInfo])
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


@router.post("/register")
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


@router.post("/unregister")
async def unregister_model(model: str):
    """注销模型。"""
    if _docmill is None:
        raise HTTPException(status_code=500, detail="DocMill 未初始化")

    try:
        _docmill.unregister_model(model)
        return {"status": "unregistered", "model": model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"注销失败: {e}")


@router.get("/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """获取模型详情。"""
    if _docmill is None:
        raise HTTPException(status_code=500, detail="DocMill 未初始化")

    try:
        info = _docmill.get_model_info(model_name)
        return ModelInfo(**info)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"模型 '{model_name}' 未注册")


@router.post("/{model_name}/load")
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


@router.post("/{model_name}/unload")
async def unload_model(model_name: str):
    """卸载模型。"""
    if _docmill is None:
        raise HTTPException(status_code=500, detail="DocMill 未初始化")

    try:
        _docmill.unload_model(model_name)
        return {"status": "unloaded", "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"卸载失败: {e}")


# --- Engine 相关 ---


@router.get("/engines/list", response_model=list[EngineInfo])
async def list_engines():
    """列出所有已注册的 Engine 类型。"""
    result = []
    for name in EngineRegistry.list_engines():
        engine_class = EngineRegistry.get(name)
        if engine_class:
            result.append(
                EngineInfo(
                    name=name,
                    requires_vllm=engine_class.requires_vllm_sidecar(),
                )
            )
    return result