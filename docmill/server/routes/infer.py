"""推理 API 路由"""

from __future__ import annotations

import base64
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from docmill.core import DocMill
from docmill.engines.base import EngineInput, EngineOutput
from docmill.storage.history_store import HistoryStore
from docmill.storage.file_store import FileStore
from docmill.utils.logging import get_logger

logger = get_logger("server.routes.infer")

router = APIRouter(prefix="/infer", tags=["infer"])

# 全局实例（由 main.py 注入）
_docmill: DocMill | None = None
_history_store: HistoryStore | None = None
_file_store: FileStore | None = None


def set_dependencies(docmill: DocMill, history_store: HistoryStore, file_store: FileStore) -> None:
    """设置依赖实例。"""
    global _docmill, _history_store, _file_store
    _docmill = docmill
    _history_store = history_store
    _file_store = file_store


# --- Request / Response 模型 ---


class InferRequest(BaseModel):
    """推理请求。"""

    model: str = Field(description="模型名称")
    file_id: str | None = Field(default=None, description="已上传文件的 ID")
    file_path: str | None = Field(default=None, description="本地文件路径（服务器端）")
    image_bytes: str | None = Field(default=None, description="Base64 编码的图片数据")
    url: str | None = Field(default=None, description="图片 URL")
    options: dict[str, Any] = Field(default_factory=dict, description="额外选项")
    save_history: bool = Field(default=True, description="是否保存到历史记录")


class InferResponse(BaseModel):
    """推理响应。"""

    id: str | None = None  # 历史记录 ID
    model: str
    text: str = ""
    markdown: str = ""
    structured: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- API 端点 ---


@router.post("", response_model=InferResponse)
async def infer(req: InferRequest):
    """执行 OCR 推理。

    支持四种输入方式：
    - file_id: 已上传文件的 ID（推荐）
    - file_path: 本地文件路径（服务器端）
    - image_bytes: Base64 编码的图片数据
    - url: 图片 URL
    """
    if _docmill is None:
        raise HTTPException(status_code=500, detail="DocMill 未初始化")

    # 构建输入
    input_kwargs: dict[str, Any] = {"options": req.options}
    filename = "unknown"

    if req.file_id:
        # 使用已上传的文件
        if _file_store is None:
            raise HTTPException(status_code=500, detail="文件存储未初始化")

        file_info = _file_store.get(req.file_id)
        if not file_info:
            raise HTTPException(status_code=404, detail=f"文件 '{req.file_id}' 不存在")

        input_kwargs["file_path"] = str(file_info.path)
        filename = file_info.filename

    elif req.file_path:
        input_kwargs["file_path"] = req.file_path
        filename = req.file_path.split("/")[-1]

    elif req.image_bytes:
        try:
            input_kwargs["image_bytes"] = base64.b64decode(req.image_bytes)
            filename = "uploaded_image"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Base64 解码失败: {e}")

    elif req.url:
        input_kwargs["url"] = req.url
        filename = req.url.split("/")[-1] or "remote_image"

    else:
        raise HTTPException(status_code=400, detail="必须提供 file_id、file_path、image_bytes 或 url")

    engine_input = EngineInput(**input_kwargs)

    # 创建历史记录
    history_record = None
    if req.save_history and _history_store:
        history_record = _history_store.create(
            model=req.model,
            file_id=req.file_id or "",
            filename=filename,
        )

    # 执行推理
    try:
        output = _docmill.infer(req.model, engine_input)

        # 更新历史记录
        if history_record and _history_store:
            _history_store.update(
                history_record.id,
                status="completed",
                result_text=output.text,
                result_markdown=output.markdown,
                result_structured=output.structured,
            )

        return InferResponse(
            id=history_record.id if history_record else None,
            model=req.model,
            text=output.text,
            markdown=output.markdown,
            structured=output.structured,
            metadata=output.metadata,
        )

    except KeyError:
        raise HTTPException(status_code=404, detail=f"模型 '{req.model}' 未注册")

    except RuntimeError as e:
        # 更新历史记录为失败
        if history_record and _history_store:
            _history_store.update(history_record.id, status="failed", error=str(e))

        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.exception("推理失败: %s", e)

        if history_record and _history_store:
            _history_store.update(history_record.id, status="failed", error=str(e))

        raise HTTPException(status_code=500, detail=f"推理失败: {e}")


@router.post("/{model_name}", response_model=InferResponse)
async def infer_with_model(model_name: str, req: InferRequest):
    """执行推理（模型名在 URL 中）。"""
    req.model = model_name
    return await infer(req)