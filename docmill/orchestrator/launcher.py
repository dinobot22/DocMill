"""vLLM Sidecar 启动器 — 通过 subprocess 管理 vLLM 生命周期。

Launcher 是 Orchestrator 的"执行臂"：
- 构建 vLLM 启动命令
- 管理子进程生命周期
- 捕获 stdout/stderr 到日志文件
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docmill.utils.logging import get_logger
from docmill.utils.ports import find_free_port

logger = get_logger("orchestrator.launcher")


@dataclass
class SidecarProcess:
    """vLLM Sidecar 进程信息。"""

    model_path: str
    port: int
    pid: int
    process: subprocess.Popen | None = None
    log_file: str = ""
    started_at: float = 0.0
    extra_info: dict[str, Any] = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        """OpenAI 兼容 API 地址。"""
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def is_alive(self) -> bool:
        """进程是否存活。"""
        if self.process is None:
            return False
        return self.process.poll() is None


class SidecarLauncher:
    """vLLM Sidecar 进程管理器。"""

    def __init__(self, log_dir: str | Path = "/tmp/docmill/sidecar_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._processes: dict[str, SidecarProcess] = {}

    def launch(
        self,
        model_path: str,
        port: int | None = None,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        extra_args: list[str] | None = None,
        gpu_id: int = 0,
        served_model_name: str | None = None,
    ) -> SidecarProcess:
        """启动 vLLM sidecar 进程。

        Args:
            model_path: 模型路径（HuggingFace 格式或本地路径）。
            port: 端口号，None 表示自动分配。
            gpu_memory_utilization: GPU 显存使用率。
            max_model_len: 最大序列长度。
            tensor_parallel_size: 张量并行大小。
            trust_remote_code: 是否信任远程代码。
            extra_args: 额外 vLLM 启动参数。
            gpu_id: GPU 设备索引。
            served_model_name: vLLM 服务注册的模型名称。

        Returns:
            SidecarProcess 实例。
        """
        if port is None:
            port = find_free_port()

        # 展开路径中的 ~ 和环境变量
        model_path = str(Path(model_path).expanduser().resolve())

        # 构建 vLLM 启动命令
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-model-len", str(max_model_len),
            "--tensor-parallel-size", str(tensor_parallel_size),
        ]

        if trust_remote_code:
            cmd.append("--trust-remote-code")

        if served_model_name:
            cmd.extend(["--served-model-name", served_model_name])

        if extra_args:
            cmd.extend(extra_args)

        # 日志文件
        safe_name = model_path.replace("/", "_").replace("\\", "_")
        log_path = self.log_dir / f"vllm_{safe_name}_{port}.log"
        log_file = open(str(log_path), "w", encoding="utf-8")

        # 设置环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        logger.info("启动 vLLM sidecar: %s (port=%d, gpu=%d)", model_path, port, gpu_id)
        logger.debug("命令: %s", " ".join(cmd))

        try:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid,  # 创建新进程组，便于清理
            )
        except Exception as e:
            log_file.close()
            raise RuntimeError(f"vLLM 启动失败: {e}") from e

        sidecar = SidecarProcess(
            model_path=model_path,
            port=port,
            pid=process.pid,
            process=process,
            log_file=str(log_path),
            started_at=time.time(),
        )

        key = self._make_key(model_path, port)
        self._processes[key] = sidecar

        logger.info("vLLM sidecar 已启动: PID=%d, port=%d, log=%s", process.pid, port, log_path)
        return sidecar

    def stop(self, sidecar: SidecarProcess, timeout: float = 15.0) -> None:
        """停止 vLLM sidecar 进程。

        先发送 SIGTERM，超时后 SIGKILL。
        """
        if sidecar.process is None or not sidecar.is_alive:
            logger.info("Sidecar 进程已停止 (model=%s)", sidecar.model_path)
            return

        pid = sidecar.pid
        logger.info("停止 vLLM sidecar: PID=%d (model=%s)", pid, sidecar.model_path)

        try:
            # 发送 SIGTERM 给整个进程组
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            sidecar.process.wait(timeout=timeout)
            logger.info("vLLM sidecar 已优雅停止: PID=%d", pid)
        except subprocess.TimeoutExpired:
            logger.warning("SIGTERM 超时，发送 SIGKILL: PID=%d", pid)
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                sidecar.process.wait(timeout=5)
            except Exception as e:
                logger.error("SIGKILL 失败: PID=%d, %s", pid, e)
        except ProcessLookupError:
            logger.debug("进程已不存在: PID=%d", pid)
        except Exception as e:
            logger.error("停止 sidecar 异常: %s", e)

        # 清理注册
        key = self._make_key(sidecar.model_path, sidecar.port)
        self._processes.pop(key, None)

    def stop_all(self) -> None:
        """停止所有已管理的 sidecar 进程。"""
        for sidecar in list(self._processes.values()):
            self.stop(sidecar)

    def get_process(self, model_path: str, port: int) -> SidecarProcess | None:
        """获取已管理的 sidecar 进程。"""
        key = self._make_key(model_path, port)
        return self._processes.get(key)

    def list_processes(self) -> list[SidecarProcess]:
        """列出所有活跃的 sidecar 进程。"""
        return [p for p in self._processes.values() if p.is_alive]

    @staticmethod
    def _make_key(model_path: str, port: int) -> str:
        return f"{model_path}:{port}"
