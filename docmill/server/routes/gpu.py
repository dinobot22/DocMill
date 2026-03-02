"""GPU 资源监控 API 路由"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from docmill.utils.logging import get_logger

logger = get_logger("server.routes.gpu")

router = APIRouter(prefix="/gpu", tags=["gpu"])


# --- Response 模型 ---


class GpuInfo(BaseModel):
    """GPU 信息。"""

    index: int = Field(description="GPU 编号")
    name: str = Field(description="GPU 名称")
    memory_total_mb: int = Field(description="总显存 (MB)")
    memory_used_mb: int = Field(description="已用显存 (MB)")
    memory_free_mb: int = Field(description="空闲显存 (MB)")
    memory_utilization: float = Field(description="显存使用率 (0-100)")
    gpu_utilization: float = Field(description="GPU 计算利用率 (0-100)")
    temperature: int = Field(description="温度 (摄氏度)")
    power_draw: float = Field(description="当前功耗 (W)")
    power_limit: float = Field(description="功耗上限 (W)")


class GpuStatus(BaseModel):
    """GPU 状态响应。"""

    available: bool = Field(description="GPU 是否可用")
    count: int = Field(description="GPU 数量")
    gpus: list[GpuInfo] = Field(default_factory=list, description="GPU 列表")
    error: str | None = Field(default=None, description="错误信息")


# --- NVML 工具函数 ---


def _init_nvml() -> bool:
    """初始化 NVML。"""
    try:
        import pynvml

        pynvml.nvmlInit()
        return True
    except ImportError:
        logger.warning("pynvml 未安装，GPU 监控不可用")
        return False
    except Exception as e:
        logger.warning("NVML 初始化失败: %s", e)
        return False


def _shutdown_nvml() -> None:
    """关闭 NVML。"""
    try:
        import pynvml

        pynvml.nvmlShutdown()
    except Exception:
        pass


def _get_gpu_info(index: int) -> GpuInfo | None:
    """获取单个 GPU 的信息。"""
    try:
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(index)

        # 基本信息
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        # 显存信息
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_total_mb = mem_info.total // (1024 * 1024)
        memory_used_mb = mem_info.used // (1024 * 1024)
        memory_free_mb = mem_info.free // (1024 * 1024)
        memory_utilization = (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0

        # GPU 利用率
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = float(util.gpu)
        except Exception:
            gpu_utilization = 0.0

        # 温度
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temperature = 0

        # 功耗
        try:
            power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # mW -> W
        except Exception:
            power_draw = 0.0
            power_limit = 0.0

        return GpuInfo(
            index=index,
            name=name,
            memory_total_mb=memory_total_mb,
            memory_used_mb=memory_used_mb,
            memory_free_mb=memory_free_mb,
            memory_utilization=round(memory_utilization, 1),
            gpu_utilization=round(gpu_utilization, 1),
            temperature=temperature,
            power_draw=round(power_draw, 1),
            power_limit=round(power_limit, 1),
        )
    except Exception as e:
        logger.error("获取 GPU %d 信息失败: %s", index, e)
        return None


# --- API 端点 ---


@router.get("", response_model=GpuStatus)
async def get_gpu_status():
    """获取所有 GPU 的状态信息。"""
    try:
        import pynvml

        # 初始化 NVML
        if not _init_nvml():
            return GpuStatus(
                available=False,
                count=0,
                gpus=[],
                error="pynvml 未安装或 NVML 初始化失败",
            )

        # 获取 GPU 数量
        try:
            gpu_count = pynvml.nvmlDeviceGetCount()
        except Exception as e:
            _shutdown_nvml()
            return GpuStatus(
                available=False,
                count=0,
                gpus=[],
                error=f"获取 GPU 数量失败: {e}",
            )

        # 获取每个 GPU 的信息
        gpus = []
        for i in range(gpu_count):
            info = _get_gpu_info(i)
            if info:
                gpus.append(info)

        _shutdown_nvml()

        return GpuStatus(
            available=True,
            count=gpu_count,
            gpus=gpus,
            error=None,
        )

    except ImportError:
        return GpuStatus(
            available=False,
            count=0,
            gpus=[],
            error="pynvml 未安装，请运行: pip install pynvml",
        )
    except Exception as e:
        logger.error("获取 GPU 状态失败: %s", e)
        return GpuStatus(
            available=False,
            count=0,
            gpus=[],
            error=str(e),
        )