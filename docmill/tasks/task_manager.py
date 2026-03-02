"""异步任务管理器 — asyncio 包装 + 事件通知。
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from docmill.tasks.task_store import Task, TaskStore, TaskStatus
from docmill.utils.logging import get_logger

if TYPE_CHECKING:
    from docmill.core import DocMill

logger = get_logger("tasks.manager")


class AsyncTaskManager:
    """异步任务管理器 — asyncio 包装 + 事件通知。
    
    使用示例：
        manager = AsyncTaskManager(store)
        
        # 提交任务
        task = await manager.submit(engine="paddle_ocr_vl", file_path="/path/to/file.pdf")
        
        # Worker 等待任务
        task = await manager.wait_for_task("worker-1")
    """
    
    def __init__(self, store: TaskStore, executor_workers: int = 4):
        self._store = store
        self._executor = ThreadPoolExecutor(max_workers=executor_workers)
        self._new_task_event = asyncio.Event()
        self._docmill: "DocMill" | None = None
        
        logger.info("AsyncTaskManager 初始化完成")
    
    def set_docmill(self, docmill: "DocMill"):
        """设置 DocMill 实例"""
        self._docmill = docmill
    
    async def submit(
        self,
        engine: str,
        file_path: str,
        priority: int = 0,
        options: dict | None = None,
    ) -> Task:
        """提交任务并通知 Worker。
        
        Args:
            engine: Engine 名称
            file_path: 文件路径
            priority: 优先级
            options: 额外选项
        
        Returns:
            创建的 Task 对象
        """
        task = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: self._store.create(engine, file_path, priority, options),
        )
        
        self._new_task_event.set()
        
        logger.info("提交任务: task_id=%s, engine=%s, file=%s", task.task_id, engine, file_path)
        
        return task
    
    async def wait_for_task(self, worker_id: str, timeout: float = 30.0) -> Task | None:
        """Worker 等待新任务（替代轮询）。
        
        Args:
            worker_id: Worker ID
            timeout: 超时时间（秒）
        
        Returns:
            获取到的 Task 或 None
        """
        try:
            await asyncio.wait_for(self._new_task_event.wait(), timeout=timeout)
            self._new_task_event.clear()
        except asyncio.TimeoutError:
            pass
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: self._store.get_next(worker_id),
        )
    
    async def complete(self, task_id: str, result_path: str) -> bool:
        """标记任务完成
        
        Args:
            task_id: 任务 ID
            result_path: 结果文件路径
        
        Returns:
            是否成功
        """
        result = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: self._store.update_status(task_id, TaskStatus.COMPLETED, result_path=result_path),
        )
        
        logger.info("任务完成: task_id=%s, result=%s", task_id, result_path)
        
        if self._docmill and result:
            parent_id = self._store.on_child_completed(task_id)
            if parent_id:
                logger.info("父任务全部完成: parent_id=%s", parent_id)
        
        return result
    
    async def fail(self, task_id: str, error: str) -> bool:
        """标记任务失败
        
        Args:
            task_id: 任务 ID
            error: 错误信息
        
        Returns:
            是否成功
        """
        result = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: self._store.update_status(task_id, TaskStatus.FAILED, error_message=error),
        )
        
        logger.error("任务失败: task_id=%s, error=%s", task_id, error)
        
        if self._docmill and result:
            self._store.on_child_failed(task_id, error)
        
        return result
    
    async def cancel(self, task_id: str) -> bool:
        """取消任务
        
        Args:
            task_id: 任务 ID
        
        Returns:
            是否成功
        """
        result = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: self._store.update_status(task_id, TaskStatus.CANCELLED),
        )
        
        logger.info("任务取消: task_id=%s", task_id)
        
        return result
    
    async def get_task(self, task_id: str) -> Task | None:
        """获取任务
        
        Args:
            task_id: 任务 ID
        
        Returns:
            Task 对象或 None
        """
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: self._store.get(task_id),
        )
    
    async def list_tasks(
        self,
        status: TaskStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Task]:
        """列出任务
        
        Args:
            status: 状态过滤
            limit: 数量限制
            offset: 偏移量
        
        Returns:
            任务列表
        """
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: self._store.list_tasks(status, limit, offset),
        )
    
    async def get_stats(self) -> dict[str, int]:
        """获取任务统计
        
        Returns:
            统计信息
        """
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._store.get_stats,
        )
