"""文件上传 API 路由"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from docmill.storage.file_store import FileStore, FileInfo
from docmill.utils.logging import get_logger

logger = get_logger("server.routes.files")

router = APIRouter(prefix="/files", tags=["files"])

# 全局实例（由 main.py 注入）
_file_store: FileStore | None = None


def set_file_store(store: FileStore) -> None:
    """设置文件存储实例。"""
    global _file_store
    _file_store = store


# --- Response 模型 ---


class FileInfoResponse(BaseModel):
    """文件信息响应。"""

    file_id: str
    filename: str
    content_type: str
    size: int
    hash: str
    created_at: str


class FileListResponse(BaseModel):
    """文件列表响应。"""

    items: list[FileInfoResponse]


# --- API 端点 ---


@router.post("/upload", response_model=FileInfoResponse)
async def upload_file(file: UploadFile = File(...)):
    """上传文件。

    支持的文件类型：
    - 图片：png, jpg, jpeg, gif, bmp, webp
    - 文档：pdf

    最大文件大小：50MB
    """
    if _file_store is None:
        raise HTTPException(status_code=500, detail="文件存储未初始化")

    try:
        # 保存文件
        file_info = _file_store.save(
            file_obj=file.file,
            filename=file.filename or "unknown",
            content_type=file.content_type or "application/octet-stream",
        )

        return FileInfoResponse(
            file_id=file_info.file_id,
            filename=file_info.filename,
            content_type=file_info.content_type,
            size=file_info.size,
            hash=file_info.hash,
            created_at=file_info.created_at.isoformat(),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error("上传文件失败: %s", e)
        raise HTTPException(status_code=500, detail=f"上传失败: {e}")


@router.get("", response_model=FileListResponse)
async def list_files(limit: int = 100, offset: int = 0):
    """列出已上传的文件。"""
    if _file_store is None:
        raise HTTPException(status_code=500, detail="文件存储未初始化")

    files = _file_store.list_files(limit=limit, offset=offset)

    return FileListResponse(
        items=[
            FileInfoResponse(
                file_id=f.file_id,
                filename=f.filename,
                content_type=f.content_type,
                size=f.size,
                hash=f.hash,
                created_at=f.created_at.isoformat(),
            )
            for f in files
        ]
    )


@router.get("/{file_id}", response_model=FileInfoResponse)
async def get_file_info(file_id: str):
    """获取文件信息。"""
    if _file_store is None:
        raise HTTPException(status_code=500, detail="文件存储未初始化")

    file_info = _file_store.get(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail=f"文件 '{file_id}' 不存在")

    return FileInfoResponse(
        file_id=file_info.file_id,
        filename=file_info.filename,
        content_type=file_info.content_type,
        size=file_info.size,
        hash=file_info.hash,
        created_at=file_info.created_at.isoformat(),
    )


@router.get("/{file_id}/download")
async def download_file(file_id: str):
    """下载文件。"""
    if _file_store is None:
        raise HTTPException(status_code=500, detail="文件存储未初始化")

    file_info = _file_store.get(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail=f"文件 '{file_id}' 不存在")

    file_path = _file_store.get_path(file_id)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"文件 '{file_id}' 不存在")

    from fastapi.responses import FileResponse

    return FileResponse(
        path=file_path,
        filename=file_info.filename,
        media_type=file_info.content_type,
    )


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """删除文件。"""
    if _file_store is None:
        raise HTTPException(status_code=500, detail="文件存储未初始化")

    deleted = _file_store.delete(file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"文件 '{file_id}' 不存在")

    return {"status": "deleted", "file_id": file_id}