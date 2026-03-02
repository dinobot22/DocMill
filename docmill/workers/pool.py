"""Worker 进程池管理器。

管理多个 Worker 进程，支持 GPU 隔离和多进程处理。
"""

from __future__ import annotations

import os
import socket
from multiprocessing import Process
from typing import TYPE_CHECKING

from docmill.config.schema import ServerConfig
from docmill.tasks.task_manager import AsyncTaskManager
from docmill.utils.logging import get_logger

if TYPE_CHECKING:
    from docmill.tasks.task_store import TaskStore

logger = get_logger("workers.pool")


def get_hostname() -> str:
    """获取主机名"""
    return socket.gethostname()


class WorkerPool:
    """Worker 进程池管理器。
    
    使用示例：
        config = ServerConfig(workers_per_gpu=2, gpu_devices=[0, 1])
        pool = WorkerPool(config, task_manager)
        pool.start()  # 启动 Worker 进程
        # ...
        pool.shutdown()  # 关闭所有 Worker
    """
    
    def __init__(
        self,
        config: ServerConfig,
        task_manager: AsyncTaskManager,
        task_store: "TaskStore",
    ):
        self.config = config
        self.task_manager = task_manager
        self.task_store = task_store
        self._processes: list[Process] = []
        self._running = False
        
        logger.info("WorkerPool 初始化: workers_per_gpu=%d, gpu_devices=%s",
                   config.workers_per_gpu, config.gpu_devices)
    
    def start(self):
        """按配置启动 Worker 进程。"""
        if self._running:
            logger.warning("WorkerPool 已经在运行中")
            return
        
        gpu_devices = self._resolve_devices()
        
        logger.info("启动 Worker 进程: %d GPU × %d Worker = %d 进程",
                   len(gpu_devices), self.config.workers_per_gpu,
                   len(gpu_devices) * self.config.workers_per_gpu)
        
        for gpu_id in gpu_devices:
            for i in range(self.config.workers_per_gpu):
                worker_id = f"worker-{get_hostname()}-gpu{gpu_id}-{i}"
                
                p = Process(
                    target=worker_main,
                    args=(gpu_id, worker_id, self.task_store),
                    name=worker_id,
                )
                p.start()
                self._processes.append(p)
                logger.info("启动 Worker: %s (GPU %d)", worker_id, gpu_id)
        
        self._running = True
        logger.info("所有 Worker 进程已启动")
    
    def shutdown(self, timeout: float = 10.0):
        """停止所有 Worker。
        
        Args:
            timeout: 等待进程结束的 timeout（秒）
        """
        if not self._running:
            return
        
        logger.info("正在关闭 Worker 进程...")
        
        for p in self._processes:
            if p.is_alive():
                p.terminate()
        
        for p in self._processes:
            p.join(timeout=timeout)
            if p.is_alive():
                logger.warning("强制终止 Worker: %s", p.name)
                p.kill()
                p.join()
        
        self._processes.clear()
        self._running = False
        logger.info("所有 Worker 已关闭")
    
    def _resolve_devices(self) -> list[int]:
        """解析 GPU 设备列表。
        
        Returns:
            GPU ID 列表
        """
        if isinstance(self.config.gpu_devices, list):
            return self.config.gpu_devices
        
        if self.config.gpu_devices == "auto":
            # 自动检测可用 GPU
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--list-gpus"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    gpu_count = len(result.stdout.strip().split("\n"))
                    return list(range(gpu_count))
            except Exception as e:
                logger.warning("自动检测 GPU 失败: %s", e)
            
            # 回退到单 GPU
            return [0]
        
        raise ValueError(f"无效的 gpu_devices: {self.config.gpu_devices}")
    
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self._running
    
    def get_worker_count(self) -> int:
        """获取 Worker 数量"""
        return len([p for p in self._processes if p.is_alive()])


def worker_main(gpu_id: int, worker_id: str, task_store: "TaskStore"):
    """Worker 子进程入口。
    
    Args:
        gpu_id: GPU 设备 ID
        worker_id: Worker ID
        task_store: 任务存储实例
    """
    # 1. GPU 隔离 — 必须在 import 任何 CUDA 库之前设置
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 2. 设置日志
    from docmill.utils.logging import setup_logging
    setup_logging()
    
    logger = get_logger("worker")
    logger.info("Worker 启动: %s, GPU %d", worker_id, gpu_id)
    
    # 3. 创建独立的 DocMill 实例
    from docmill.core import DocMill
    docmill = DocMill(
        sidecar_log_dir="/tmp/docmill/sidecar_logs",
        default_gpu_id=gpu_id,
    )
    
    # 4. 创建任务管理器
    from docmill.tasks.task_manager import AsyncTaskManager
    task_manager = AsyncTaskManager(task_store)
    task_manager.set_docmill(docmill)
    
    # 5. 任务循环
    from docmill.tasks.task_store import TaskStatus
    
    while True:
        try:
            task = task_manager.wait_for_task(worker_id, timeout=30.0)
            
            if task is None:
                continue
            
            logger.info("开始处理任务: task_id=%s, engine=%s, file=%s",
                       task.task_id, task.engine, task.file_path)
            
            try:
                # 确保模型就绪
                docmill.ensure_model_ready(task.engine)
                
                # 执行推理
                result = docmill.infer(task.engine, task.file_path, **task.options)
                
                # 保存结果
                from pathlib import Path
                result_dir = Path("/tmp/docmill/results") / task.task_id
                result_dir.mkdir(parents=True, exist_ok=True)
                
                result_path = result_dir / "result.md"
                result_path.write_text(result.text, encoding="utf-8")
                
                # 更新任务状态
                task_manager.complete(task.task_id, result_path=str(result_path))
                
                logger.info("任务完成: task_id=%s", task.task_id)
            
            except Exception as e:
                logger.error("任务失败: task_id=%s, error=%s", task.task_id, e)
                task_manager.fail(task.task_id, str(e))
        
        except KeyboardInterrupt:
            logger.info("Worker 收到中断信号，正在退出...")
            break
        except Exception as e:
            logger.error("Worker 异常: %s", e)
            continue
    
    # 6. 清理
    docmill.shutdown()
    logger.info("Worker 退出: %s", worker_id)
