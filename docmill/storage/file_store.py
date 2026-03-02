"""FileStore - 文件存储管理。

支持上传文件的存储和管理。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from docmill.utils.logging import get_logger

logger = get_logger("storage.file_store")


@dataclass
class FileInfo:
    """文件信息。"""

    file_id: str
    filename: str
    content_type: str
    size: int
    hash: str
    created_at: datetime
    path: Path

    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "content_type": self.content_type,
            "size": self.size,
            "hash": self.hash,
            "created_at": self.created_at.isoformat(),
        }


class FileStore:
    """文件存储管理器。

    功能：
    - 存储上传的文件
    - 按文件 ID 检索文件
    - 自动清理过期文件

    使用示例:
        store = FileStore("/data/uploads")
        file_info = store.save(file_obj, "image.png", "image/png")
        file_path = store.get_path(file_info.file_id)
    """

    # 支持的文件类型
    ALLOWED_TYPES = {
        # 图片
        "image/png": [".png"],
        "image/jpeg": [".jpg", ".jpeg"],
        "image/gif": [".gif"],
        "image/bmp": [".bmp"],
        "image/webp": [".webp"],
        # 文档
        "application/pdf": [".pdf"],
    }

    # 最大文件大小 (50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024

    def __init__(self, storage_dir: str | Path = "/tmp/docmill/uploads"):
        """初始化文件存储。

        Args:
            storage_dir: 存储目录路径。
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, FileInfo] = {}
        self._load_existing_files()
        logger.info("FileStore 初始化: %s", self.storage_dir)

    def _load_existing_files(self) -> None:
        """加载已存在的文件信息。"""
        for file_dir in self.storage_dir.iterdir():
            if file_dir.is_dir():
                meta_file = file_dir / "meta.json"
                if meta_file.exists():
                    try:
                        import json
                        with open(meta_file, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # 查找实际文件（带扩展名）
                        actual_file = None
                        for f in file_dir.iterdir():
                            if f.is_file() and f.name != "meta.json":
                                actual_file = f
                                break

                        if actual_file is None:
                            logger.warning("找不到文件: %s", file_dir)
                            continue

                        file_info = FileInfo(
                            file_id=data["file_id"],
                            filename=data["filename"],
                            content_type=data["content_type"],
                            size=data["size"],
                            hash=data["hash"],
                            created_at=datetime.fromisoformat(data["created_at"]),
                            path=actual_file,
                        )
                        self._files[file_info.file_id] = file_info
                    except Exception as e:
                        logger.warning("加载文件信息失败: %s - %s", file_dir, e)

    def save(
        self,
        file_obj: BinaryIO,
        filename: str,
        content_type: str,
    ) -> FileInfo:
        """保存文件。

        Args:
            file_obj: 文件对象（支持 read() 方法）。
            filename: 原始文件名。
            content_type: 文件类型。

        Returns:
            文件信息。

        Raises:
            ValueError: 文件类型不支持或文件过大。
        """
        # 验证文件类型
        if content_type not in self.ALLOWED_TYPES:
            allowed = ", ".join(self.ALLOWED_TYPES.keys())
            raise ValueError(f"不支持的文件类型: {content_type}，允许: {allowed}")

        # 读取文件内容
        content = file_obj.read()
        size = len(content)

        # 验证文件大小
        if size > self.MAX_FILE_SIZE:
            raise ValueError(f"文件过大: {size} bytes，最大: {self.MAX_FILE_SIZE} bytes")

        # 生成文件 ID 和路径
        file_id = uuid.uuid4().hex
        file_hash = hashlib.sha256(content).hexdigest()[:16]
        file_dir = self.storage_dir / file_id
        file_dir.mkdir(parents=True, exist_ok=True)

        # 保留原始文件扩展名
        original_ext = Path(filename).suffix.lower()
        file_name = f"file{original_ext}" if original_ext else "file"
        file_path = file_dir / file_name

        # 保存文件
        with open(file_path, "wb") as f:
            f.write(content)

        # 创建文件信息
        file_info = FileInfo(
            file_id=file_id,
            filename=filename,
            content_type=content_type,
            size=size,
            hash=file_hash,
            created_at=datetime.now(),
            path=file_path,
        )

        # 保存元数据
        self._save_meta(file_info)

        self._files[file_id] = file_info
        logger.info("保存文件: %s (%s, %d bytes)", file_id, filename, size)

        return file_info

    def _save_meta(self, file_info: FileInfo) -> None:
        """保存文件元数据。"""
        import json

        meta_path = self.storage_dir / file_info.file_id / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(file_info.to_dict(), f, ensure_ascii=False, indent=2)

    def get(self, file_id: str) -> FileInfo | None:
        """获取文件信息。

        Args:
            file_id: 文件 ID。

        Returns:
            文件信息，不存在返回 None。
        """
        return self._files.get(file_id)

    def get_path(self, file_id: str) -> Path | None:
        """获取文件路径。

        Args:
            file_id: 文件 ID。

        Returns:
            文件路径，不存在返回 None。
        """
        file_info = self._files.get(file_id)
        if file_info and file_info.path.exists():
            return file_info.path
        return None

    def delete(self, file_id: str) -> bool:
        """删除文件。

        Args:
            file_id: 文件 ID。

        Returns:
            是否成功删除。
        """
        file_info = self._files.pop(file_id, None)
        if file_info:
            file_dir = self.storage_dir / file_id
            if file_dir.exists():
                shutil.rmtree(file_dir)
                logger.info("删除文件: %s", file_id)
            return True
        return False

    def list_files(self, limit: int = 100, offset: int = 0) -> list[FileInfo]:
        """列出文件。

        Args:
            limit: 最大数量。
            offset: 偏移量。

        Returns:
            文件列表。
        """
        files = sorted(self._files.values(), key=lambda x: x.created_at, reverse=True)
        return files[offset : offset + limit]

    def cleanup(self, max_age_hours: int = 24) -> int:
        """清理过期文件。

        Args:
            max_age_hours: 最大保留时间（小时）。

        Returns:
            清理的文件数量。
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        count = 0

        for file_id, file_info in list(self._files.items()):
            if file_info.created_at < cutoff:
                self.delete(file_id)
                count += 1

        logger.info("清理过期文件: %d 个", count)
        return count