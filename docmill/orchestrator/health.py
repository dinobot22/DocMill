"""健康检查器 — 探测 vLLM sidecar 就绪状态。

通过 HTTP 探针（/v1/models 或 /health）检测 sidecar 是否就绪。
不依赖 stdout 日志，因为日志格式可能变化。
"""

from __future__ import annotations

import time
from enum import Enum

import requests

from docmill.utils.errors import HealthCheckError, ModelLoadTimeoutError
from docmill.utils.logging import get_logger

logger = get_logger("orchestrator.health")


class SidecarStatus(str, Enum):
    """Sidecar 状态。"""

    UNKNOWN = "unknown"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class HealthChecker:
    """vLLM Sidecar 健康检查器。"""

    def __init__(
        self,
        probe_interval: float = 2.0,
        probe_timeout: float = 3.0,
    ):
        self.probe_interval = probe_interval
        self.probe_timeout = probe_timeout

    def probe(self, endpoint: str) -> SidecarStatus:
        """单次健康探测。

        Args:
            endpoint: vLLM API base URL (如 http://127.0.0.1:8000/v1)。

        Returns:
            SidecarStatus。
        """
        url = f"{endpoint.rstrip('/')}/models"
        try:
            response = requests.get(url, timeout=self.probe_timeout)
            if response.status_code == 200:
                return SidecarStatus.READY
            else:
                return SidecarStatus.LOADING
        except requests.ConnectionError:
            return SidecarStatus.LOADING
        except requests.Timeout:
            return SidecarStatus.LOADING
        except Exception as e:
            logger.warning("健康探测异常: %s", e)
            return SidecarStatus.UNKNOWN

    def wait_until_ready(
        self,
        endpoint: str,
        timeout: float = 120.0,
        model_name: str = "",
    ) -> None:
        """等待 sidecar 就绪。

        Args:
            endpoint: vLLM API base URL。
            timeout: 最大等待时间（秒）。
            model_name: 模型名称（用于日志和错误消息）。

        Raises:
            ModelLoadTimeoutError: 超时未就绪。
        """
        start = time.time()
        last_status = SidecarStatus.UNKNOWN
        check_count = 0

        logger.info("等待 vLLM 就绪: %s (超时: %.0fs)", endpoint, timeout)

        while time.time() - start < timeout:
            status = self.probe(endpoint)
            check_count += 1

            if status != last_status:
                logger.info("Sidecar 状态: %s → %s (检查 #%d)", last_status.value, status.value, check_count)
                last_status = status

            if status == SidecarStatus.READY:
                elapsed = time.time() - start
                logger.info("vLLM 已就绪！耗时 %.1fs", elapsed)
                return

            if status == SidecarStatus.FAILED:
                raise HealthCheckError(endpoint, "Sidecar 报告 FAILED 状态")

            time.sleep(self.probe_interval)

        raise ModelLoadTimeoutError(model_name or endpoint, timeout)

    def check_health(self, endpoint: str) -> bool:
        """快速健康检查。

        Returns:
            True 表示就绪。
        """
        return self.probe(endpoint) == SidecarStatus.READY
