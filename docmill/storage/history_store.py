"""HistoryStore - 推理历史记录存储。

使用 SQLite 持久化存储历史记录。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from docmill.utils.logging import get_logger

logger = get_logger("storage.history_store")


@dataclass
class HistoryRecord:
    """历史记录。"""

    id: str
    model: str
    file_id: str
    filename: str
    status: str  # pending, processing, completed, failed
    created_at: datetime
    completed_at: datetime | None = None
    result_text: str = ""
    result_markdown: str = ""
    result_structured: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            "id": self.id,
            "model": self.model,
            "file_id": self.file_id,
            "filename": self.filename,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result_text": self.result_text,
            "result_markdown": self.result_markdown,
            "result_structured": self.result_structured,
            "error": self.error,
            "metadata": self.metadata,
        }


class HistoryStore:
    """历史记录存储管理器。

    使用 SQLite 持久化存储。

    使用示例:
        store = HistoryStore("/data/history.db")
        record = store.create(model="paddle-ocr-vl", file_id="xxx", filename="test.png")
        store.update(record.id, status="completed", result_text="OCR 结果")
        records = store.list(limit=10)
    """

    def __init__(self, db_path: str | Path = "/tmp/docmill/history.db"):
        """初始化历史存储。

        Args:
            db_path: 数据库文件路径。
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("HistoryStore 初始化: %s", self.db_path)

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接。"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """初始化数据库表。"""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    result_text TEXT DEFAULT '',
                    result_markdown TEXT DEFAULT '',
                    result_structured TEXT DEFAULT '{}',
                    error TEXT DEFAULT '',
                    metadata TEXT DEFAULT '{}'
                )
            """)
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON history(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON history(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON history(model)")
            conn.commit()
        finally:
            conn.close()

    def create(
        self,
        model: str,
        file_id: str,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> HistoryRecord:
        """创建历史记录。

        Args:
            model: 模型名称。
            file_id: 文件 ID。
            filename: 原始文件名。
            metadata: 额外元数据。

        Returns:
            创建的记录。
        """
        import uuid

        record = HistoryRecord(
            id=uuid.uuid4().hex,
            model=model,
            file_id=file_id,
            filename=filename,
            status="pending",
            created_at=datetime.now(),
            metadata=metadata or {},
        )

        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO history (id, model, file_id, filename, status, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.model,
                    record.file_id,
                    record.filename,
                    record.status,
                    record.created_at.isoformat(),
                    json.dumps(record.metadata),
                ),
            )
            conn.commit()
            logger.info("创建历史记录: %s", record.id)
            return record
        finally:
            conn.close()

    def update(
        self,
        record_id: str,
        status: str | None = None,
        result_text: str | None = None,
        result_markdown: str | None = None,
        result_structured: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> HistoryRecord | None:
        """更新历史记录。

        Args:
            record_id: 记录 ID。
            status: 状态。
            result_text: 结果文本。
            result_markdown: 结果 Markdown。
            result_structured: 结果结构化数据。
            error: 错误信息。

        Returns:
            更新后的记录，不存在返回 None。
        """
        conn = self._get_conn()
        try:
            # 构建更新语句
            updates = []
            params = []

            if status is not None:
                updates.append("status = ?")
                params.append(status)

            if result_text is not None:
                updates.append("result_text = ?")
                params.append(result_text)

            if result_markdown is not None:
                updates.append("result_markdown = ?")
                params.append(result_markdown)

            if result_structured is not None:
                updates.append("result_structured = ?")
                params.append(json.dumps(result_structured))

            if error is not None:
                updates.append("error = ?")
                params.append(error)

            if status in ("completed", "failed"):
                updates.append("completed_at = ?")
                params.append(datetime.now().isoformat())

            if not updates:
                return self.get(record_id)

            params.append(record_id)

            conn.execute(
                f"UPDATE history SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()

            return self.get(record_id)
        finally:
            conn.close()

    def get(self, record_id: str) -> HistoryRecord | None:
        """获取历史记录。

        Args:
            record_id: 记录 ID。

        Returns:
            历史记录，不存在返回 None。
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM history WHERE id = ?",
                (record_id,),
            ).fetchone()

            if row:
                return self._row_to_record(row)
            return None
        finally:
            conn.close()

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        model: str | None = None,
        status: str | None = None,
    ) -> list[HistoryRecord]:
        """列出历史记录。

        Args:
            limit: 最大数量。
            offset: 偏移量。
            model: 按模型过滤。
            status: 按状态过滤。

        Returns:
            记录列表。
        """
        conn = self._get_conn()
        try:
            query = "SELECT * FROM history WHERE 1=1"
            params = []

            if model:
                query += " AND model = ?"
                params.append(model)

            if status:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            return [self._row_to_record(row) for row in rows]
        finally:
            conn.close()

    def delete(self, record_id: str) -> bool:
        """删除历史记录。

        Args:
            record_id: 记录 ID。

        Returns:
            是否成功删除。
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute("DELETE FROM history WHERE id = ?", (record_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info("删除历史记录: %s", record_id)
            return deleted
        finally:
            conn.close()

    def count(self, model: str | None = None, status: str | None = None) -> int:
        """统计记录数量。

        Args:
            model: 按模型过滤。
            status: 按状态过滤。

        Returns:
            记录数量。
        """
        conn = self._get_conn()
        try:
            query = "SELECT COUNT(*) FROM history WHERE 1=1"
            params = []

            if model:
                query += " AND model = ?"
                params.append(model)

            if status:
                query += " AND status = ?"
                params.append(status)

            return conn.execute(query, params).fetchone()[0]
        finally:
            conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> HistoryRecord:
        """将数据库行转换为记录对象。"""
        return HistoryRecord(
            id=row["id"],
            model=row["model"],
            file_id=row["file_id"],
            filename=row["filename"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            result_text=row["result_text"] or "",
            result_markdown=row["result_markdown"] or "",
            result_structured=json.loads(row["result_structured"] or "{}"),
            error=row["error"] or "",
            metadata=json.loads(row["metadata"] or "{}"),
        )