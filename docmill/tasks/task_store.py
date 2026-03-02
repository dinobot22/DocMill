"""任务队列与异步处理 — SQLite WAL 任务存储。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from docmill.utils.logging import get_logger

logger = get_logger("tasks.store")


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """任务模型"""
    task_id: str
    engine: str
    file_path: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    options: dict[str, Any] = {}
    result_path: str | None = None
    error_message: str | None = None
    worker_id: str | None = None
    parent_task_id: str | None = None
    is_parent: bool = False
    child_count: int = 0
    child_completed: int = 0
    retry_count: int = 0
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    class Config:
        use_enum_values = True


class TaskStore:
    """任务存储 — 基于 SQLite WAL 模式。
    
    使用示例：
        store = TaskStore("/tmp/docmill/tasks.db")
        
        # 创建任务
        task = store.create(engine="paddle_ocr_vl", file_path="/path/to/file.pdf")
        task_id = task.task_id
        
        # 原子拉取任务
        task = store.get_next("worker-1")
        
        # 更新状态
        store.update_status(task_id, TaskStatus.COMPLETED, result_path="/path/to/result")
    """
    
    def __init__(self, db_path: str | Path):
        """初始化 TaskStore。
        
        Args:
            db_path: SQLite 数据库路径。
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._init_db()
        
        logger.info("TaskStore 初始化完成: %s", self.db_path)
    
    def _init_db(self):
        """初始化数据库表"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                engine TEXT NOT NULL,
                file_path TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 0,
                options TEXT DEFAULT '{}',
                result_path TEXT,
                error_message TEXT,
                worker_id TEXT,
                parent_task_id TEXT,
                is_parent INTEGER DEFAULT 0,
                child_count INTEGER DEFAULT 0,
                child_completed INTEGER DEFAULT 0,
                retry_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON tasks(priority DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_parent ON tasks(parent_task_id)")
        
        conn.commit()
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn
    
    def create(
        self,
        engine: str,
        file_path: str,
        priority: int = 0,
        options: dict[str, Any] | None = None,
        parent_task_id: str | None = None,
    ) -> Task:
        """创建新任务"""
        import uuid
        task_id = str(uuid.uuid4())
        
        is_parent = False
        if parent_task_id is None:
            is_parent = True
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO tasks (task_id, engine, file_path, priority, options, parent_task_id, is_parent, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task_id,
            engine,
            file_path,
            priority,
            json.dumps(options or {}),
            parent_task_id,
            1 if is_parent else 0,
            datetime.now().isoformat(),
        ))
        
        conn.commit()
        
        return Task(
            task_id=task_id,
            engine=engine,
            file_path=file_path,
            priority=priority,
            options=options or {},
            parent_task_id=parent_task_id,
            is_parent=is_parent,
            created_at=datetime.now(),
        )
    
    def get(self, task_id: str) -> Task | None:
        """获取任务"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return self._row_to_task(row)
    
    def get_next(self, worker_id: str) -> Task | None:
        """原子拉取下一个待处理任务"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("BEGIN IMMEDIATE")
            
            cursor.execute("""
                SELECT * FROM tasks
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            """)
            row = cursor.fetchone()
            
            if row is None:
                conn.rollback()
                return None
            
            task_id = row["task_id"]
            
            cursor.execute("""
                UPDATE tasks
                SET status = 'processing', worker_id = ?, started_at = ?
                WHERE task_id = ? AND status = 'pending'
            """, (worker_id, datetime.now().isoformat(), task_id))
            
            if cursor.rowcount == 0:
                conn.rollback()
                return None
            
            conn.commit()
            
            return self._row_to_task(row)
        
        except Exception as e:
            conn.rollback()
            logger.error("拉取任务失败: %s", e)
            return None
    
    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        result_path: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        """更新任务状态"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        completed_at = datetime.now().isoformat() if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED) else None
        
        cursor.execute("""
            UPDATE tasks
            SET status = ?, result_path = ?, error_message = ?, completed_at = ?
            WHERE task_id = ?
        """, (status.value, result_path, error_message, completed_at, task_id))
        
        conn.commit()
        
        return cursor.rowcount > 0
    
    def list_tasks(
        self,
        status: TaskStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Task]:
        """列出任务"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if status:
            cursor.execute("""
                SELECT * FROM tasks
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (status.value, limit, offset))
        else:
            cursor.execute("""
                SELECT * FROM tasks
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
        
        rows = cursor.fetchall()
        return [self._row_to_task(row) for row in rows]
    
    def get_stats(self) -> dict[str, int]:
        """获取任务统计"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM tasks
            GROUP BY status
        """)
        
        return {row["status"]: row["count"] for row in cursor.fetchall()}
    
    def create_child(
        self,
        parent_id: str,
        engine: str,
        file_path: str,
        priority: int = 0,
        options: dict[str, Any] | None = None,
    ) -> Task:
        """创建子任务"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE tasks
            SET child_count = child_count + 1
            WHERE task_id = ? AND is_parent = 1
        """, (parent_id,))
        
        conn.commit()
        
        return self.create(
            engine=engine,
            file_path=file_path,
            priority=priority,
            options=options,
            parent_task_id=parent_id,
        )
    
    def on_child_completed(self, child_id: str) -> str | None:
        """子任务完成回调"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT parent_task_id FROM tasks WHERE task_id = ?", (child_id,))
        row = cursor.fetchone()
        
        if not row or not row["parent_task_id"]:
            return None
        
        parent_id = row["parent_task_id"]
        
        cursor.execute("""
            UPDATE tasks
            SET child_completed = child_completed + 1
            WHERE task_id = ?
        """, (parent_id,))
        
        cursor.execute("SELECT child_count, child_completed FROM tasks WHERE task_id = ?", (parent_id,))
        parent_row = cursor.fetchone()
        
        conn.commit()
        
        if parent_row and parent_row["child_count"] == parent_row["child_completed"]:
            self.update_status(parent_id, TaskStatus.COMPLETED)
            return parent_id
        
        return None
    
    def on_child_failed(self, child_id: str, error: str) -> None:
        """子任务失败回调"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT parent_task_id FROM tasks WHERE task_id = ?", (child_id,))
        row = cursor.fetchone()
        
        if not row or not row["parent_task_id"]:
            return
        
        parent_id = row["parent_task_id"]
        
        cursor.execute("""
            UPDATE tasks
            SET status = 'failed', error_message = ?, completed_at = ?
            WHERE task_id = ? AND is_parent = 1
        """, (error, datetime.now().isoformat(), parent_id))
        
        conn.commit()
    
    def reset_stale(self, timeout_minutes: int = 60) -> int:
        """重置超时的 processing 任务为 pending"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE tasks
            SET status = 'pending', worker_id = NULL, started_at = NULL
            WHERE status = 'processing'
            AND datetime(started_at) < datetime('now', '-' || ? || ' minutes')
        """, (timeout_minutes,))
        
        conn.commit()
        
        return cursor.rowcount
    
    def reset_on_startup(self) -> int:
        """启动时重置所有 processing 任务为 failed"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE tasks
            SET status = 'failed', error_message = 'Worker restarted', completed_at = ?
            WHERE status = 'processing'
        """, (datetime.now().isoformat(),))
        
        conn.commit()
        
        return cursor.rowcount
    
    def cleanup_old(self, days: int = 30) -> int:
        """删除过期任务"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM tasks
            WHERE datetime(created_at) < datetime('now', '-' || ? || ' days')
            AND status IN ('completed', 'failed', 'cancelled')
        """, (days,))
        
        conn.commit()
        
        return cursor.rowcount
    
    def _row_to_task(self, row: sqlite3.Row) -> Task:
        """将数据库行转换为 Task 对象"""
        return Task(
            task_id=row["task_id"],
            engine=row["engine"],
            file_path=row["file_path"],
            status=TaskStatus(row["status"]),
            priority=row["priority"],
            options=json.loads(row["options"]),
            result_path=row["result_path"],
            error_message=row["error_message"],
            worker_id=row["worker_id"],
            parent_task_id=row["parent_task_id"],
            is_parent=bool(row["is_parent"]),
            child_count=row["child_count"],
            child_completed=row["child_completed"],
            retry_count=row["retry_count"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        )
