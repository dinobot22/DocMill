"""端口管理工具。"""

from __future__ import annotations

import socket

from docmill.utils.logging import get_logger

logger = get_logger("utils.ports")


def find_free_port(start: int = 10000, end: int = 65535) -> int:
    """查找一个空闲端口。

    Args:
        start: 搜索起始端口。
        end: 搜索结束端口。

    Returns:
        可用端口号。

    Raises:
        RuntimeError: 在指定范围内没有可用端口。
    """
    # 优先使用系统自动分配
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            if start <= port <= end:
                logger.debug("系统自动分配端口: %d", port)
                return port
    except OSError:
        pass

    # 手动扫描范围
    for port in range(start, end + 1):
        if not is_port_in_use(port):
            logger.debug("找到空闲端口: %d", port)
            return port

    raise RuntimeError(f"端口 {start}-{end} 范围内没有可用端口")


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """检查端口是否被占用。

    Args:
        port: 端口号。
        host: 主机地址。

    Returns:
        True 表示端口被占用。
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result == 0
    except OSError:
        return False
