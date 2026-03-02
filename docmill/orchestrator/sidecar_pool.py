"""SidecarPool — vLLM Sidecar 进程池管理。

简化的 Sidecar 管理器：
- 启动/停止 vLLM sidecar 进程
- 管理端点和进程映射
- 提供健康检查
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docmill.orchestrator.launcher import SidecarLauncher, SidecarProcess
from docmill.utils.logging import get_logger

logger = get_logger("orchestrator.pool")


@dataclass
class SidecarEntry:
    """Sidecar 条目。"""

    endpoint: str
    sidecar: SidecarProcess
    ref_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)


class SidecarPool:
    """vLLM Sidecar 进程池。

    管理多个 vLLM sidecar 实例：
    - 按需启动（同一个模型路径复用）
    - 引用计数管理
    - 统一关闭

    使用示例:
        pool = SidecarPool()

        # 获取 sidecar（自动启动）
        entry = pool.acquire(model_path="/models/my-llm")
        endpoint = entry.endpoint  # "http://127.0.0.1:8000/v1"

        # 使用完后释放
        pool.release(endpoint)

        # 关闭所有
        pool.shutdown()
    """

    def __init__(
        self,
        log_dir: str | Path = "/tmp/docmill/sidecar_logs",
        health_check_timeout: float = 180.0,
    ):
        """初始化 SidecarPool。

        Args:
            log_dir: sidecar 日志目录。
            health_check_timeout: 健康检查超时时间（秒）。
        """
        self._launcher = SidecarLauncher(log_dir=log_dir)
        self._health_check_timeout = health_check_timeout

        # endpoint -> SidecarEntry
        self._entries: dict[str, SidecarEntry] = {}
        # model_path -> endpoint (用于复用)
        self._model_to_endpoint: dict[str, str] = {}

        self._lock = threading.Lock()
        logger.info("SidecarPool 初始化完成")

    def acquire(
        self,
        model_path: str,
        port: int | None = None,
        gpu_id: int = 0,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        extra_args: list[str] | None = None,
        reuse: bool = True,
        served_model_name: str | None = None,
    ) -> SidecarEntry:
        """获取 sidecar 实例。

        如果 reuse=True 且已有相同 model_path 的 sidecar，会复用。

        Args:
            model_path: vLLM 模型路径。
            port: 端口号，None 表示自动分配。
            gpu_id: GPU 设备 ID。
            gpu_memory_utilization: 显存使用率。
            max_model_len: 最大序列长度。
            tensor_parallel_size: 张量并行大小。
            trust_remote_code: 是否信任远程代码。
            extra_args: 额外启动参数。
            reuse: 是否复用已有 sidecar。
            served_model_name: vLLM 服务注册的模型名称。

        Returns:
            SidecarEntry 实例。
        """
        with self._lock:
            # 检查是否可以复用
            if reuse and model_path in self._model_to_endpoint:
                endpoint = self._model_to_endpoint[model_path]
                entry = self._entries.get(endpoint)
                if entry and entry.sidecar.is_alive:
                    entry.ref_count += 1
                    entry.last_used_at = time.time()
                    logger.info("复用 sidecar: model=%s, endpoint=%s, ref_count=%d",
                               model_path, endpoint, entry.ref_count)
                    return entry

            # 启动新的 sidecar
            sidecar = self._launcher.launch(
                model_path=model_path,
                port=port,
                gpu_id=gpu_id,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
                extra_args=extra_args,
                served_model_name=served_model_name,
            )

            # 等待就绪
            self._wait_until_ready(sidecar)

            # 创建条目
            entry = SidecarEntry(
                endpoint=sidecar.endpoint,
                sidecar=sidecar,
                ref_count=1,
            )

            self._entries[sidecar.endpoint] = entry
            self._model_to_endpoint[model_path] = sidecar.endpoint

            logger.info("启动 sidecar: model=%s, endpoint=%s, pid=%d",
                       model_path, sidecar.endpoint, sidecar.pid)
            return entry

    def release(self, endpoint: str) -> None:
        """释放 sidecar 引用。

        Args:
            endpoint: sidecar 端点 URL。
        """
        with self._lock:
            entry = self._entries.get(endpoint)
            if not entry:
                logger.warning("未找到 sidecar: %s", endpoint)
                return

            entry.ref_count -= 1
            entry.last_used_at = time.time()
            logger.debug("释放 sidecar: endpoint=%s, ref_count=%d",
                        endpoint, entry.ref_count)

    def get(self, endpoint: str) -> SidecarEntry | None:
        """获取 sidecar 条目。

        Args:
            endpoint: 端点 URL。

        Returns:
            SidecarEntry 或 None。
        """
        return self._entries.get(endpoint)

    def list_all(self) -> list[SidecarEntry]:
        """列出所有 sidecar 条目。"""
        return list(self._entries.values())

    def list_alive(self) -> list[SidecarEntry]:
        """列出所有存活的 sidecar。"""
        return [e for e in self._entries.values() if e.sidecar.is_alive]

    def stop(self, endpoint: str, force: bool = False) -> None:
        """停止指定的 sidecar。

        Args:
            endpoint: 端点 URL。
            force: 是否强制停止（忽略引用计数）。
        """
        with self._lock:
            entry = self._entries.get(endpoint)
            if not entry:
                return

            if not force and entry.ref_count > 0:
                logger.warning("sidecar 仍有引用 (ref_count=%d)，使用 force=True 强制停止",
                              entry.ref_count)
                return

            # 停止进程
            self._launcher.stop(entry.sidecar)

            # 清理映射
            model_path = entry.sidecar.model_path
            self._model_to_endpoint.pop(model_path, None)
            self._entries.pop(endpoint, None)

            logger.info("停止 sidecar: endpoint=%s", endpoint)

    def shutdown(self) -> None:
        """关闭所有 sidecar。"""
        logger.info("关闭所有 sidecar...")
        with self._lock:
            for entry in list(self._entries.values()):
                try:
                    self._launcher.stop(entry.sidecar)
                except Exception as e:
                    logger.warning("停止 sidecar 失败: %s", e)

            self._entries.clear()
            self._model_to_endpoint.clear()

        logger.info("所有 sidecar 已关闭")

    def _wait_until_ready(self, sidecar: SidecarProcess) -> None:
        """等待 sidecar 就绪。

        Args:
            sidecar: Sidecar 实例。

        Raises:
            RuntimeError: 超时未就绪。
        """
        import httpx

        health_url = sidecar.endpoint.replace("/v1", "/health")
        start_time = time.time()
        timeout = self._health_check_timeout

        logger.info("等待 sidecar 就绪: endpoint=%s, timeout=%.0fs",
                   sidecar.endpoint, timeout)

        while time.time() - start_time < timeout:
            # 检查进程是否存活
            if not sidecar.is_alive:
                raise RuntimeError(
                    f"vLLM sidecar 进程异常退出 (pid={sidecar.pid})"
                )

            # 健康检查
            try:
                response = httpx.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    logger.info("sidecar 就绪: endpoint=%s, 耗时=%.1fs",
                               sidecar.endpoint, time.time() - start_time)
                    return
            except Exception:
                pass

            time.sleep(2.0)

        raise RuntimeError(
            f"vLLM sidecar 启动超时 (timeout={timeout}s): {sidecar.endpoint}"
        )

    def __enter__(self) -> "SidecarPool":
        return self

    def __exit__(self, *args) -> None:
        self.shutdown()