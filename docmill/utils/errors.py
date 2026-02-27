"""DocMill 自定义异常类。"""


class DocMillError(Exception):
    """DocMill 异常基类。"""


class ModelNotFoundError(DocMillError):
    """模型配置或权重文件未找到。"""

    def __init__(self, model_name: str, detail: str = ""):
        self.model_name = model_name
        msg = f"模型 '{model_name}' 未找到"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class ModelLoadTimeoutError(DocMillError):
    """模型加载超时（包括 vLLM sidecar 启动超时）。"""

    def __init__(self, model_name: str, timeout_seconds: float):
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        super().__init__(f"模型 '{model_name}' 在 {timeout_seconds}s 内未就绪")


class InsufficientVRAMError(DocMillError):
    """GPU 显存不足，无法加载模型。"""

    def __init__(self, required_mb: float, available_mb: float, model_name: str = ""):
        self.required_mb = required_mb
        self.available_mb = available_mb
        self.model_name = model_name
        msg = f"显存不足: 需要 {required_mb:.0f}MB, 可用 {available_mb:.0f}MB"
        if model_name:
            msg = f"模型 '{model_name}' {msg}"
        super().__init__(msg)


class SidecarCrashedError(DocMillError):
    """vLLM sidecar 进程异常退出。"""

    def __init__(self, model_name: str, return_code: int | None = None):
        self.model_name = model_name
        self.return_code = return_code
        msg = f"Sidecar 进程崩溃 (模型: '{model_name}')"
        if return_code is not None:
            msg += f", 退出码: {return_code}"
        super().__init__(msg)


class InvalidSpecError(DocMillError):
    """模型配置 (ModelSpec) 校验失败。"""

    def __init__(self, detail: str):
        super().__init__(f"无效的模型配置: {detail}")


class PipelineError(DocMillError):
    """Pipeline 执行过程中的错误。"""


class WorkerError(DocMillError):
    """Worker 推理过程中的错误。"""


class HealthCheckError(DocMillError):
    """健康检查失败。"""

    def __init__(self, endpoint: str, detail: str = ""):
        self.endpoint = endpoint
        msg = f"健康检查失败: {endpoint}"
        if detail:
            msg += f" - {detail}"
        super().__init__(msg)
