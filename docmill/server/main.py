"""DocMill FastAPI Server - 主入口"""

from __future__ import annotations

import os as _os
import sys as _sys

# 将项目根目录加入 sys.path，使本文件可被直接执行（无需 pip install）
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import atexit
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docmill.core import DocMill
from docmill.storage.file_store import FileStore
from docmill.storage.history_store import HistoryStore
from docmill.server.routes import models, infer, history, files, gpu
from docmill.server.routes import tasks as tasks_routes
from docmill.tasks.task_store import TaskStore
from docmill.tasks.task_manager import AsyncTaskManager
from docmill.utils.logging import get_logger, setup_logging

logger = get_logger("server.main")

# --- 全局状态 ---
_docmill: DocMill | None = None
_file_store: FileStore | None = None
_history_store: HistoryStore | None = None
_task_store: TaskStore | None = None
_task_manager: AsyncTaskManager | None = None

# --- 配置 ---
DATA_DIR = Path("/tmp/docmill")
UPLOAD_DIR = DATA_DIR / "uploads"
HISTORY_DB = DATA_DIR / "history.db"
TASKS_DB = DATA_DIR / "tasks.db"


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    setup_logging()
    global _docmill, _file_store, _history_store, _task_store, _task_manager

    # 初始化存储
    _file_store = FileStore(UPLOAD_DIR)
    _history_store = HistoryStore(HISTORY_DB)
    _task_store = TaskStore(TASKS_DB)

    # 初始化 DocMill
    _docmill = DocMill()
    
    # 初始化任务管理器
    _task_manager = AsyncTaskManager(_task_store)
    _task_manager.set_docmill(_docmill)

    # 注入依赖
    models.set_docmill(_docmill)
    infer.set_dependencies(_docmill, _history_store, _file_store)
    history.set_history_store(_history_store)
    files.set_file_store(_file_store)
    tasks_routes.set_task_dependencies(_task_manager, _task_store)

    atexit.register(_cleanup)

    logger.info("DocMill Server 启动完成")
    yield

    _cleanup()
    logger.info("DocMill Server 已关闭")


def _cleanup():
    """清理资源。"""
    global _docmill
    if _docmill:
        _docmill.shutdown()
        _docmill = None


# --- FastAPI App ---
app = FastAPI(
    title="DocMill",
    description="A unified inference runtime for OCR & VLM document understanding.",
    version="0.2.0",
    lifespan=lifespan,
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 注册路由 ---
app.include_router(models.router)
app.include_router(infer.router)
app.include_router(history.router)
app.include_router(files.router)
app.include_router(gpu.router)
app.include_router(tasks_routes.router)
app.include_router(tasks_routes.queue_router)


# --- 健康检查 ---
@app.get("/health")
async def health_check():
    """服务健康检查。"""
    return {
        "status": "ok",
        "version": "0.2.0",
        "models_registered": len(_docmill.list_models()) if _docmill else 0,
    }


@app.get("/")
async def root():
    """根路径。"""
    return {
        "name": "DocMill",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/health",
    }


# --- 静态文件（生产环境） ---
def mount_static_files(app: FastAPI, static_dir: Path) -> None:
    """挂载静态文件目录（用于服务前端）。

    Args:
        app: FastAPI 应用。
        static_dir: 静态文件目录。
    """
    from fastapi.staticfiles import StaticFiles

    if static_dir.exists():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")
        logger.info("挂载静态文件: %s", static_dir)


def create_app(
    data_dir: str | Path | None = None,
    static_dir: str | Path | None = None,
) -> FastAPI:
    """创建并配置 FastAPI 应用。

    Args:
        data_dir: 数据目录（用于存储上传文件和历史记录）。
        static_dir: 静态文件目录（用于服务前端）。

    Returns:
        配置好的 FastAPI 应用。
    """
    global DATA_DIR, UPLOAD_DIR, HISTORY_DB, TASKS_DB

    if data_dir:
        DATA_DIR = Path(data_dir)
        UPLOAD_DIR = DATA_DIR / "uploads"
        HISTORY_DB = DATA_DIR / "history.db"
        TASKS_DB = DATA_DIR / "tasks.db"

    if static_dir:
        mount_static_files(app, Path(static_dir))

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "docmill.server.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        reload_dirs=[_PROJECT_ROOT],
    )