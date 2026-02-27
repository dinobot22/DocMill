"""GPU 硬件检测工具。"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from docmill.utils.logging import get_logger

logger = get_logger("utils.hardware")


@dataclass
class GPUInfo:
    """单个 GPU 的信息。"""

    index: int
    name: str
    total_memory_mb: float
    used_memory_mb: float
    free_memory_mb: float
    utilization_percent: float

    @property
    def usage_ratio(self) -> float:
        """显存使用率 (0.0 ~ 1.0)。"""
        if self.total_memory_mb <= 0:
            return 0.0
        return self.used_memory_mb / self.total_memory_mb


def get_gpu_info_list() -> list[GPUInfo]:
    """通过 nvidia-smi 获取所有 GPU 的信息。

    Returns:
        GPUInfo 列表，检测失败时返回空列表。
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi 执行失败: %s", result.stderr.strip())
            return []

        gpus: list[GPUInfo] = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append(
                GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    total_memory_mb=float(parts[2]),
                    used_memory_mb=float(parts[3]),
                    free_memory_mb=float(parts[4]),
                    utilization_percent=float(parts[5]),
                )
            )
        return gpus

    except FileNotFoundError:
        logger.warning("nvidia-smi 未找到，可能没有 NVIDIA GPU")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi 执行超时")
        return []
    except Exception as e:
        logger.warning("GPU 信息获取失败: %s", e)
        return []


def get_gpu_info(gpu_index: int = 0) -> GPUInfo | None:
    """获取指定 GPU 的信息。

    Args:
        gpu_index: GPU 索引号。

    Returns:
        GPUInfo 或 None（未找到/检测失败时）。
    """
    gpus = get_gpu_info_list()
    for gpu in gpus:
        if gpu.index == gpu_index:
            return gpu
    return None


def get_total_vram_mb(gpu_index: int = 0) -> float:
    """获取指定 GPU 的总显存 (MB)。

    Args:
        gpu_index: GPU 索引号。

    Returns:
        总显存 MB，获取失败返回 0.0。
    """
    gpu = get_gpu_info(gpu_index)
    return gpu.total_memory_mb if gpu else 0.0


def get_free_vram_mb(gpu_index: int = 0) -> float:
    """获取指定 GPU 的可用显存 (MB)。

    Args:
        gpu_index: GPU 索引号。

    Returns:
        可用显存 MB，获取失败返回 0.0。
    """
    gpu = get_gpu_info(gpu_index)
    return gpu.free_memory_mb if gpu else 0.0
