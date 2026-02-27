"""模型运行时注册表 — 追踪活跃的模型实例。

Registry 是"电话簿"角色：
- 记录哪些模型当前在运行
- 每个实例的状态、显存、端口等信息
- 线程安全操作
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RuntimeState(str, Enum):
    """模型运行时状态。"""

    COLD = "cold"
    LOADING = "loading"
    READY = "ready"
    IDLE = "idle"
    EVICTED = "evicted"
    FAILED = "failed"


@dataclass
class ModelRuntime:
    """单个模型运行时实例。"""

    model_name: str
    spec_hash: str  # 配置唯一标识
    state: RuntimeState = RuntimeState.COLD
    endpoint: str = ""  # LLM API endpoint
    port: int = 0
    pid: int = 0
    estimated_vram_mb: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    active_requests: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """更新最后访问时间。"""
        self.last_access_time = time.time()

    @property
    def idle_seconds(self) -> float:
        """空闲秒数。"""
        return time.time() - self.last_access_time

    @property
    def is_busy(self) -> bool:
        """是否有活跃请求。"""
        return self.active_requests > 0


class RuntimeRegistry:
    """模型运行时注册表（线程安全）。"""

    def __init__(self) -> None:
        self._runtimes: dict[str, ModelRuntime] = {}
        self._lock = threading.RLock()

    def register(self, runtime: ModelRuntime) -> None:
        """注册一个运行时实例。"""
        with self._lock:
            self._runtimes[runtime.model_name] = runtime

    def unregister(self, model_name: str) -> ModelRuntime | None:
        """注销运行时实例。"""
        with self._lock:
            return self._runtimes.pop(model_name, None)

    def get(self, model_name: str) -> ModelRuntime | None:
        """获取运行时实例。"""
        with self._lock:
            return self._runtimes.get(model_name)

    def update_state(self, model_name: str, state: RuntimeState) -> None:
        """更新运行时状态。"""
        with self._lock:
            runtime = self._runtimes.get(model_name)
            if runtime:
                runtime.state = state

    def list_by_state(self, state: RuntimeState) -> list[ModelRuntime]:
        """按状态列出运行时实例。"""
        with self._lock:
            return [r for r in self._runtimes.values() if r.state == state]

    def list_all(self) -> list[ModelRuntime]:
        """列出所有运行时实例。"""
        with self._lock:
            return list(self._runtimes.values())

    def get_idle_sorted_by_lru(self) -> list[ModelRuntime]:
        """获取按 LRU 排序的 IDLE 实例（最久未访问在前）。"""
        with self._lock:
            idle = [r for r in self._runtimes.values() if r.state == RuntimeState.IDLE and not r.is_busy]
            return sorted(idle, key=lambda r: r.last_access_time)

    def total_estimated_vram_mb(self) -> float:
        """当前所有非 EVICTED/COLD 实例的预计显存总量。"""
        with self._lock:
            return sum(
                r.estimated_vram_mb
                for r in self._runtimes.values()
                if r.state not in (RuntimeState.COLD, RuntimeState.EVICTED)
            )

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._runtimes)
