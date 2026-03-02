"""任务相关 API 路由。
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from docmill.tasks.task_manager import AsyncTaskManager
from docmill.tasks.task_store import Task, TaskStatus, TaskStore

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

# 全局 TaskManager（将在 main.py 中设置）
_task_manager: Optional[AsyncTaskManager] = None
_task_store: Optional[TaskStore] = None


def set_task_dependencies(manager: AsyncTaskManager, store: TaskStore):
    """设置依赖（由 main.py 调用）"""
    global _task_manager, _task_store
    _task_manager = manager
    _task_store = store


def get_task_manager() -> AsyncTaskManager:
    """获取 TaskManager"""
    if _task_manager is None:
        raise HTTPException(status_code=500, detail="TaskManager not initialized")
    return _task_manager


def get_task_store() -> TaskStore:
    """获取 TaskStore"""
    if _task_store is None:
        raise HTTPException(status_code=500, detail="TaskStore not initialized")
    return _task_store


# ========== 请求/响应模型 ==========


class SubmitTaskRequest(BaseModel):
    """提交任务请求"""
    engine: str
    file_path: str
    priority: int = 0
    options: dict = {}


class SubmitTaskResponse(BaseModel):
    """提交任务响应"""
    task_id: str
    status: str


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    status: str
    engine: str
    file_path: str
    result_path: Optional[str] = None
    error_message: Optional[str] = None
    progress: float = 0.0


class TaskStatsResponse(BaseModel):
    """任务统计响应"""
    pending: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0


# ========== API 端点 ==========


@router.post("", response_model=SubmitTaskResponse, status_code=201)
async def submit_task(request: SubmitTaskRequest):
    """提交异步任务
    
    提交一个新的 OCR 推理任务，立即返回 task_id，任务在后台异步执行。
    """
    manager = get_task_manager()
    
    task = await manager.submit(
        engine=request.engine,
        file_path=request.file_path,
        priority=request.priority,
        options=request.options,
    )
    
    return SubmitTaskResponse(
        task_id=task.task_id,
        status=task.status,
    )


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """查询任务状态
    
    获取指定任务的当前状态和进度。
    """
    manager = get_task_manager()
    
    task = await manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    progress = 0.0
    if task.status == TaskStatus.PROCESSING:
        progress = 0.5  # 简化的进度计算
    elif task.status == TaskStatus.COMPLETED:
        progress = 1.0
    
    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        engine=task.engine,
        file_path=task.file_path,
        result_path=task.result_path,
        error_message=task.error_message,
        progress=progress,
    )


@router.get("/{task_id}/result")
async def get_task_result(task_id: str):
    """获取任务结果
    
    获取已完成任务的结果文件路径。
    """
    manager = get_task_manager()
    
    task = await manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed, current status: {task.status}"
        )
    
    if not task.result_path:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return {"result_path": task.result_path}


@router.delete("/{task_id}")
async def cancel_task(task_id: str):
    """取消任务
    
    尝试取消指定任务。如果任务正在执行，可能无法取消。
    """
    manager = get_task_manager()
    
    task = await manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status not in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel task with status: {task.status}"
        )
    
    success = await manager.cancel(task_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to cancel task")
    
    return {"message": "Task cancelled"}


@router.get("", response_model=list[TaskStatusResponse])
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """任务列表
    
    列出所有任务，支持分页和状态筛选。
    """
    manager = get_task_manager()
    
    status_enum = TaskStatus(status) if status else None
    
    tasks = await manager.list_tasks(status_enum, limit, offset)
    
    return [
        TaskStatusResponse(
            task_id=task.task_id,
            status=task.status,
            engine=task.engine,
            file_path=task.file_path,
            result_path=task.result_path,
            error_message=task.error_message,
            progress=1.0 if task.status == TaskStatus.COMPLETED else 0.0,
        )
        for task in tasks
    ]


# ========== 队列统计 ==========


queue_router = APIRouter(prefix="/api/v1/queue", tags=["queue"])


@queue_router.get("/stats", response_model=TaskStatsResponse)
async def get_queue_stats():
    """队列统计
    
    获取当前任务队列的统计信息。
    """
    manager = get_task_manager()
    
    stats = await manager.get_stats()
    
    return TaskStatsResponse(
        pending=stats.get("pending", 0),
        processing=stats.get("processing", 0),
        completed=stats.get("completed", 0),
        failed=stats.get("failed", 0),
        cancelled=stats.get("cancelled", 0),
    )
