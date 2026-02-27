"""DocMill 统一日志配置。"""

import logging
import sys
from pathlib import Path


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> None:
    """初始化 DocMill 统一日志。

    Args:
        level: 日志级别（DEBUG / INFO / WARNING / ERROR）。
        log_file: 可选的日志文件路径。
    """
    global _initialized
    if _initialized:
        return

    root_logger = logging.getLogger("docmill")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root_logger.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """获取 DocMill 子模块 Logger。

    Args:
        name: 模块名称（自动添加 docmill. 前缀）。

    Returns:
        Logger 实例。
    """
    if not name.startswith("docmill."):
        name = f"docmill.{name}"
    return logging.getLogger(name)
