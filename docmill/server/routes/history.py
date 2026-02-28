"""历史记录 API 路由"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any

from docmill.storage.history_store import HistoryStore, HistoryRecord
from docmill.utils.logging import get_logger

logger = get_logger("server.routes.history")

router = APIRouter(prefix="/history", tags=["history"])

# 全局实例（由 main.py 注入）
_history_store: HistoryStore | None = None


def set_history_store(store: HistoryStore) -> None:
    """设置历史存储实例。"""
    global _history_store
    _history_store = store


# --- Response 模型 ---


class HistoryRecordResponse(BaseModel):
    """历史记录响应。"""

    id: str
    model: str
    file_id: str
    filename: str
    status: str
    created_at: str
    completed_at: str | None
    result_text: str = ""
    result_markdown: str = ""
    result_structured: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class HistoryListResponse(BaseModel):
    """历史列表响应。"""

    total: int
    items: list[HistoryRecordResponse]


# --- API 端点 ---


@router.get("", response_model=HistoryListResponse)
async def list_history(
    limit: int = 50,
    offset: int = 0,
    model: str | None = None,
    status: str | None = None,
):
    """列出历史记录。

    Args:
        limit: 最大数量（默认 50）。
        offset: 偏移量（默认 0）。
        model: 按模型过滤。
        status: 按状态过滤。
    """
    if _history_store is None:
        raise HTTPException(status_code=500, detail="历史存储未初始化")

    records = _history_store.list(limit=limit, offset=offset, model=model, status=status)
    total = _history_store.count(model=model, status=status)

    return HistoryListResponse(
        total=total,
        items=[_record_to_response(r) for r in records],
    )


@router.get("/{record_id}", response_model=HistoryRecordResponse)
async def get_history(record_id: str):
    """获取历史记录详情。"""
    if _history_store is None:
        raise HTTPException(status_code=500, detail="历史存储未初始化")

    record = _history_store.get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"记录 '{record_id}' 不存在")

    return _record_to_response(record)


@router.delete("/{record_id}")
async def delete_history(record_id: str):
    """删除历史记录。"""
    if _history_store is None:
        raise HTTPException(status_code=500, detail="历史存储未初始化")

    deleted = _history_store.delete(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"记录 '{record_id}' 不存在")

    return {"status": "deleted", "id": record_id}


@router.get("/{record_id}/download")
async def download_result(record_id: str, format: str = "txt"):
    """下载推理结果。

    Args:
        record_id: 记录 ID。
        format: 格式（txt, md, json）。
    """
    if _history_store is None:
        raise HTTPException(status_code=500, detail="历史存储未初始化")

    record = _history_store.get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"记录 '{record_id}' 不存在")

    if record.status != "completed":
        raise HTTPException(status_code=400, detail="记录未完成，无法下载")

    # 根据 format 选择内容
    if format == "txt":
        content = record.result_text
        media_type = "text/plain"
        filename = f"{record.filename}.txt"
    elif format == "md":
        content = record.result_markdown or record.result_text
        media_type = "text/markdown"
        filename = f"{record.filename}.md"
    elif format == "json":
        import json
        content = json.dumps(record.to_dict(), ensure_ascii=False, indent=2)
        media_type = "application/json"
        filename = f"{record.filename}.json"
    else:
        raise HTTPException(status_code=400, detail=f"不支持的格式: {format}")

    from fastapi.responses import Response

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _record_to_response(record: HistoryRecord) -> HistoryRecordResponse:
    """将记录转换为响应。"""
    return HistoryRecordResponse(
        id=record.id,
        model=record.model,
        file_id=record.file_id,
        filename=record.filename,
        status=record.status,
        created_at=record.created_at.isoformat(),
        completed_at=record.completed_at.isoformat() if record.completed_at else None,
        result_text=record.result_text,
        result_markdown=record.result_markdown,
        result_structured=record.result_structured,
        error=record.error,
    )