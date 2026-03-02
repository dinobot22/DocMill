# DocMill — Next Work（待办工作清单）

> 📅 更新时间: 2026-03-02
>
> 按重要性分级，整理 DocMill 构建成完整可用系统还需要完成的工作。
> 每个任务都是**自包含的**，包含完整的实现规格，可独立开发。

---

## 📦 当前已完成的模块

| 模块 | 文件 | 状态 |
|:-----|:-----|:-----|
| 核心运行时 | `core.py` — DocMill 类（模型注册、生命周期、推理） | ✅ 可用 |
| Engine 抽象 | `engines/base.py` — BaseEngine ABC（load/infer/unload） | ✅ 可用 |
| Engine 注册表 | `engines/registry.py` — 自动注册内置 Engine | ✅ 可用 |
| PaddleOCR-VL Engine | `engines/paddle_ocr_vl.py` — 需要 vLLM sidecar | ✅ 可用 |
| DeepSeek OCR Engine | `engines/deepseek_ocr.py` — 纯 vLLM 模式 | ✅ 可用 |
| LLM 客户端 | `clients/openai_compat.py` — OpenAI 兼容 HTTP 客户端 | ✅ 可用 |
| vLLM 启动器 | `orchestrator/launcher.py` — 自动启动 vLLM 子进程 | ✅ 可用 |
| vLLM 进程池 | `orchestrator/sidecar_pool.py` — 管理多个 vLLM 实例 | ✅ 可用 |
| FastAPI 服务 | `server/main.py` + `routes/` — 基础 API 骨架 | ⚠️ 骨架 |
| 文件存储 | `storage/file_store.py` | ✅ 可用 |
| 历史记录 | `storage/history_store.py` | ✅ 可用 |
| Docker 部署 | `docker/` — Dockerfile + docker-compose 骨架 | ⚠️ 骨架 |
| 前端 | `frontend/` — Vue + Vite 骨架 | ⚠️ 骨架 |

**关键架构约束**（参见 `DESIGN.md`）：
- Engine 不允许直接 import 具体 ML 框架（torch/paddle 只在 Engine 内部出现）
- vLLM Sidecar 只能通过 HTTP 访问，Pipeline/Core 永远不直接操作 vLLM 进程
- 三种推理形态：vision_only / vision_llm / llm_only

---

## 🔴 P0 — 必须做（没有这些无法正常使用）

### 1. 统一配置系统

**优先级: ★★★★★ | 工作量：约 2 天**

> 这是所有后续功能的基石。没有统一配置，用户无法"一键启动"。

**目标**：用 Pydantic 定义配置模型，支持 YAML + ENV + CLI 三层覆盖。

**创建文件**：
```
docmill/config/
  __init__.py
  schema.py         # Pydantic 配置模型
  loader.py         # 多源配置加载器
```

**`schema.py` 完整数据结构**：
```python
from pydantic import BaseModel, Field

class VLLMConfig(BaseModel):
    """vLLM Sidecar 配置"""
    mode: str = "managed"             # "managed"=DocMill自动管理 | "external"=连接外部已有的vLLM
    model_path: str = ""              # managed 模式必填：HuggingFace 模型路径或本地路径
    endpoints: list[str] = []         # external 模式必填：外部 vLLM API 列表
    gpu_id: int = 0
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    served_model_name: str | None = None
    extra_args: list[str] = []

class EngineConfig(BaseModel):
    """单个 Engine 配置"""
    engine: str                       # Engine 注册名（如 "paddle_ocr_vl"）
    auto_start: bool = True           # 是否随系统启动
    vllm: VLLMConfig | None = None    # vLLM 配置（Engine.requires_vllm_sidecar()=True 时必填）
    engine_kwargs: dict = {}          # 传给 Engine 构造函数的额外参数

class ServerConfig(BaseModel):
    """服务器配置"""
    api_port: int = 8080
    data_dir: str = "/tmp/docmill"
    workers_per_gpu: int = 1
    gpu_devices: list[int] | str = "auto"   # "auto" 或 [0, 1, 2]

class DocMillConfig(BaseModel):
    """DocMill 全局配置"""
    server: ServerConfig = ServerConfig()
    engines: dict[str, EngineConfig] = {}
```

**`loader.py` 加载优先级**：
```
YAML 文件 (docmill.yaml) → ENV 环境变量覆盖 → CLI 参数覆盖 → 最终 DocMillConfig
```

**用户配置文件示例** (`docmill.yaml`)：
```yaml
server:
  api_port: 8080
  data_dir: ./data

engines:
  paddle-ocr-vl:
    engine: paddle_ocr_vl
    auto_start: true
    vllm:
      mode: managed
      model_path: /models/PaddleOCR-VL-0.9B
      gpu_id: 0
      served_model_name: PaddleOCR-VL-0.9B

  deepseek-ocr:
    engine: deepseek_ocr
    auto_start: false
    vllm:
      mode: managed
      model_path: deepseek-ai/DeepSeek-OCR
      gpu_id: 1
```

**与现有代码的集成点**：
- `core.py` 的 `register_model()` 方法已有 `vllm_config` 参数，可直接接受 `VLLMConfig.dict()`
- `EngineRegistry.get_or_raise(config.engine)` 获取 Engine 类
- `SidecarPool.acquire(model_path=config.vllm.model_path, ...)` 启动 vLLM

---

### 2. 配置驱动的一键启动器

**优先级: ★★★★★ | 工作量：约 3 天**

> 用户应该只需执行一条命令即可拉起全部服务。

**创建文件**：
```
docmill/launcher/
  __init__.py
  app_launcher.py     # 顶层启动器 + CLI 入口
```

**`app_launcher.py` 核心流程**：
```python
import signal, sys, argparse
from docmill.config.loader import load_config
from docmill.config.schema import DocMillConfig
from docmill.core import DocMill

class AppLauncher:
    def __init__(self, config: DocMillConfig):
        self.config = config
        self.docmill = DocMill(
            sidecar_log_dir=f"{config.server.data_dir}/sidecar_logs",
        )
    
    def start(self):
        """一键启动：注册引擎 → 拉起 vLLM → 启动 API Server"""
        # 1. 遍历配置，注册所有 Engine
        for name, engine_cfg in self.config.engines.items():
            vllm_config = engine_cfg.vllm.dict() if engine_cfg.vllm else None
            self.docmill.register_model(
                name=name,
                engine_name=engine_cfg.engine,
                vllm_config=vllm_config,
                **engine_cfg.engine_kwargs,
            )
        
        # 2. 自动启动 auto_start=True 的引擎（会自动拉起 vLLM sidecar）
        for name, engine_cfg in self.config.engines.items():
            if engine_cfg.auto_start:
                self.docmill.ensure_model_ready(name)
        
        # 3. 启动 FastAPI Server
        import uvicorn
        from docmill.server.main import create_app
        app = create_app(data_dir=self.config.server.data_dir)
        uvicorn.run(app, host="0.0.0.0", port=self.config.server.api_port)
    
    def shutdown(self, signum=None, frame=None):
        """优雅关闭"""
        self.docmill.shutdown()
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="DocMill - OCR & VLM Inference Runtime")
    parser.add_argument("command", choices=["serve"], help="Command to run")
    parser.add_argument("--config", default="docmill.yaml", help="Config file path")
    args = parser.parse_args()
    
    config = load_config(args.config)
    launcher = AppLauncher(config)
    signal.signal(signal.SIGINT, launcher.shutdown)
    signal.signal(signal.SIGTERM, launcher.shutdown)
    launcher.start()
```

**修改 `pyproject.toml`**：
```toml
[project.scripts]
docmill = "docmill.launcher.app_launcher:main"
```

**用户体验**：
```bash
docmill serve --config docmill.yaml
```

---

### 3. 任务队列与异步处理

**优先级: ★★★★★ | 工作量：约 4 天**

> PDF 处理可能耗时几分钟甚至几十分钟，同步接口完全不可用。

#### 技术方案：SQLite WAL + asyncio 事件通知

选择 SQLite（而非 Redis/PostgreSQL）的理由：
- **零依赖**：不需要额外安装任何服务，与"一键启动"理念一致
- **瓶颈不在队列**：DocMill 的瓶颈在 GPU 推理（秒~分钟级），不在队列操作（毫秒级）。即使 8 GPU × 2 Worker = 16 并发，SQLite WAL 也完全够用
- **SQL 灵活**：复杂查询、统计、父子任务关联，一句 SQL 搞定

#### 创建文件

```
docmill/tasks/
  __init__.py
  task_store.py       # SQLite WAL 任务存储
  task_manager.py     # 异步任务管理器（asyncio 包装 + Event 通知）
  task_worker.py      # 后台任务执行器

docmill/server/routes/
  tasks.py            # 任务相关 API 路由
```

#### `task_store.py` 实现规格

**SQLite WAL 模式配置**（在创建连接时设置）：
```python
conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
conn.execute("PRAGMA journal_mode=WAL")       # 读写并发
conn.execute("PRAGMA synchronous=NORMAL")     # 性能与安全平衡
conn.execute("PRAGMA busy_timeout=5000")      # 锁等待 5s
conn.execute("PRAGMA cache_size=-64000")      # 64MB 页缓存
conn.row_factory = sqlite3.Row
```

**数据模型**（用 Pydantic 替代手写 dict）：
```python
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Task(BaseModel):
    task_id: str
    engine: str                          # 使用的 Engine 名称
    file_path: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0                    # 数字越大越优先
    options: dict = {}
    result_path: str | None = None
    error_message: str | None = None
    worker_id: str | None = None
    parent_task_id: str | None = None    # 父任务 ID（PDF 拆分场景）
    is_parent: bool = False
    child_count: int = 0
    child_completed: int = 0
    retry_count: int = 0
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
```

**tasks 表 DDL**：
```sql
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    engine TEXT NOT NULL,
    file_path TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    options TEXT DEFAULT '{}',
    result_path TEXT,
    error_message TEXT,
    worker_id TEXT,
    parent_task_id TEXT,
    is_parent INTEGER DEFAULT 0,
    child_count INTEGER DEFAULT 0,
    child_completed INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_priority ON tasks(priority DESC);
CREATE INDEX IF NOT EXISTS idx_parent ON tasks(parent_task_id);
```

**核心方法**：
```python
class TaskStore:
    def create(self, engine: str, file_path: str, priority: int = 0, options: dict = {}) -> Task
    
    def get(self, task_id: str) -> Task | None
    
    def get_next(self, worker_id: str) -> Task | None
        """原子拉取：BEGIN IMMEDIATE → SELECT WHERE status='pending' ORDER BY priority DESC, created_at ASC LIMIT 1 → UPDATE SET status='processing', worker_id=? WHERE task_id=? AND status='pending' → 检查 rowcount > 0"""
    
    def update_status(self, task_id: str, status: TaskStatus, result_path: str = None, error_message: str = None) -> bool
        """更新时检查前置状态：completed/failed 只能从 processing 转入"""
    
    def list_tasks(self, status: TaskStatus | None = None, limit: int = 100) -> list[Task]
    
    def get_stats(self) -> dict[str, int]
        """SELECT status, COUNT(*) FROM tasks GROUP BY status"""
    
    # 父子任务支持
    def create_child(self, parent_id: str, engine: str, file_path: str, ...) -> Task
    def on_child_completed(self, child_id: str) -> str | None
        """更新父任务 child_completed 计数，全部完成时返回 parent_id"""
    def on_child_failed(self, child_id: str, error: str) -> None
        """标记父任务为 failed"""
    
    # 维护
    def reset_stale(self, timeout_minutes: int = 60) -> int
        """重置超时的 processing 任务为 pending"""
    def reset_on_startup(self) -> int
        """启动时将所有 processing 任务重置为 failed（防止重启后卡死）"""
    def cleanup_old(self, days: int = 30) -> int
        """删除过期任务的文件和记录"""
```

#### `task_manager.py` 实现规格（asyncio 包装）

```python
class AsyncTaskManager:
    """异步任务管理器 — asyncio 包装 + 事件通知。"""
    def __init__(self, store: TaskStore):
        self._store = store
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._new_task_event = asyncio.Event()   # 新任务通知

    async def submit(self, engine: str, file_path: str, **kwargs) -> Task:
        """提交任务并通知 Worker。"""
        task = await asyncio.get_event_loop().run_in_executor(
            self._executor, lambda: self._store.create(engine, file_path, **kwargs)
        )
        self._new_task_event.set()  # 唤醒等待中的 Worker
        return task

    async def wait_for_task(self, worker_id: str, timeout: float = 30.0) -> Task | None:
        """Worker 等待新任务（替代 while+sleep 轮询）。"""
        try:
            await asyncio.wait_for(self._new_task_event.wait(), timeout=timeout)
            self._new_task_event.clear()
        except asyncio.TimeoutError:
            pass
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, lambda: self._store.get_next(worker_id)
        )
```

#### API 路由

```
POST   /api/v1/tasks              → 提交异步任务 → {"task_id": "xxx", "status": "pending"}
GET    /api/v1/tasks/{id}         → 查询任务状态 → {"status": "processing", "progress": 0.6}
GET    /api/v1/tasks/{id}/result  → 获取任务结果
DELETE /api/v1/tasks/{id}         → 取消任务
GET    /api/v1/tasks              → 任务列表（分页、筛选 ?status=pending&limit=20）
GET    /api/v1/queue/stats        → 队列统计 → {"pending": 3, "processing": 1, "completed": 42}
```

---

### 4. Worker Pool 与负载均衡

**优先级: ★★★★☆ | 工作量：约 4 天**

> 多 GPU 并发处理是生产环境的基本要求。

**创建文件**：
```
docmill/workers/
  __init__.py
  pool.py               # Worker 进程池管理器
  worker_process.py     # 单个 Worker 子进程
  gpu_allocator.py      # GPU 设备分配策略
```

#### `pool.py` 实现规格

```python
class WorkerPool:
    """Worker 进程池管理器。"""
    def __init__(self, config: ServerConfig, task_manager: AsyncTaskManager):
        self.config = config
        self.task_manager = task_manager
        self._processes: list[Process] = []

    def start(self):
        """按配置启动 Worker 进程。"""
        gpu_devices = self._resolve_devices()  # "auto" → [0,1,...] 或用户指定
        for gpu_id in gpu_devices:
            for i in range(self.config.workers_per_gpu):
                p = Process(target=worker_main, args=(gpu_id, self.task_manager))
                p.start()
                self._processes.append(p)

    def shutdown(self):
        """停止所有 Worker。"""
        for p in self._processes:
            p.terminate()
            p.join(timeout=10)
```

#### `worker_process.py` 核心逻辑

```python
def worker_main(gpu_id: int, task_manager: AsyncTaskManager):
    """Worker 子进程入口。"""
    # 1. GPU 隔离 — 必须在 import 任何 CUDA 库之前设置
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 2. 创建独立的 DocMill 实例
    docmill = DocMill()
    worker_id = f"worker-{hostname}-gpu{gpu_id}-{os.getpid()}"
    
    # 3. 任务循环
    while True:
        task = task_manager.wait_for_task(worker_id, timeout=30.0)
        if task is None:
            continue
        
        try:
            # 确保模型就绪（如果尚未加载）
            docmill.ensure_model_ready(task.engine)
            # 执行推理
            result = docmill.infer(task.engine, task.file_path, **task.options)
            # 保存结果并更新任务状态
            task_manager.complete(task.task_id, result_path=save_result(result))
        except Exception as e:
            task_manager.fail(task.task_id, error=str(e))
```

**关键设计**：
- 每个 Worker 进程设置独立的 `CUDA_VISIBLE_DEVICES`，实现 GPU 隔离
- 每个 Worker 内部拥有独立的 `DocMill` 实例（Engine 独立加载）
- vLLM Sidecar 是独立进程，多个 Worker 可通过 HTTP 共享同一个 Sidecar
- Worker 通过 `AsyncTaskManager.wait_for_task()` 等待任务（Event 通知，非轮询）

---

### 5. Engine 文件夹化重构

**优先级: ★★★★☆ | 工作量：约 1 天**

> 当前每个 Engine 是单文件（如 `paddle_ocr_vl.py` 296 行）。随着配置、文档、测试的增长，应重构为独立文件夹。

**当前结构 → 目标结构**：
```
# 当前（单文件）                    # 目标（文件夹）
engines/                            engines/
  base.py                             base.py
  registry.py                         registry.py
  paddle_ocr_vl.py  (296行)           paddle_ocr_vl/
  deepseek_ocr.py   (239行)             __init__.py       # 导出 PaddleOCRVLEngine
                                        engine.py         # Engine 核心逻辑（原 .py 文件内容）
                                        defaults.yaml     # 默认 vLLM 配置、模型路径
                                        README.md         # 部署要求、GPU 要求、使用说明
                                      deepseek_ocr/
                                        __init__.py
                                        engine.py
                                        defaults.yaml
                                        README.md
```

**`defaults.yaml` 示例**（`paddle_ocr_vl/defaults.yaml`）：
```yaml
engine_name: paddle_ocr_vl
requires_vllm: true
estimate_vram_mb: 2048

vllm_defaults:
  model_path: PaddlePaddle/PaddleOCR-VL-0.9B
  gpu_memory_utilization: 0.8
  max_model_len: 4096
  served_model_name: PaddleOCR-VL-0.9B

requirements:
  gpu_compute_capability: ">=8.5"
  python_packages:
    - paddlepaddle-gpu>=3.2.0
    - paddleocr[doc-parser]
```

**`__init__.py` 示例**：
```python
from docmill.engines.paddle_ocr_vl.engine import PaddleOCRVLEngine
__all__ = ["PaddleOCRVLEngine"]
```

**重构步骤**：
1. 创建 `engines/paddle_ocr_vl/` 目录
2. 将 `engines/paddle_ocr_vl.py` 移动为 `engines/paddle_ocr_vl/engine.py`
3. 创建 `__init__.py` 导出 Engine 类
4. 从 Engine 代码中提取默认配置到 `defaults.yaml`
5. 创建 `README.md` 说明 GPU 要求和依赖
6. 更新 `engines/registry.py` 的 `_register_builtin_engines()` 导入路径
7. 对 `deepseek_ocr` 重复以上步骤

**好处**：
- 新增 Engine 只需创建一个文件夹 + 实现 `engine.py`，不需要改核心代码
- `defaults.yaml` 可被配置系统（P0.1）自动读取，减少用户手写配置
- 后续自动发现（P1.6）可以扫描 `engines/*/` 目录自动注册

---

## 🟡 P1 — 重要改进（生产可用性）

### 6. Engine 自动发现机制

**优先级: ★★★★☆ | 工作量：约 1.5 天**

> 配合 P0.5 的文件夹化，实现"放目录即注册"。

**改进 `engines/registry.py`**：

```python
class EngineRegistry:
    @classmethod
    def auto_discover(cls, engines_dir: Path = None):
        """扫描 engines/*/ 目录，自动注册所有 Engine。
        
        遍历 engines_dir 下的每个子包：
        1. 检查是否有 __init__.py
        2. 动态 import 模块
        3. 找到 BaseEngine 子类并注册
        4. 如有 defaults.yaml，加载为该 Engine 的默认配置
        """
        if engines_dir is None:
            engines_dir = Path(__file__).parent
        for pkg_dir in engines_dir.iterdir():
            if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
                module = importlib.import_module(f"docmill.engines.{pkg_dir.name}")
                for attr in dir(module):
                    obj = getattr(module, attr)
                    if isinstance(obj, type) and issubclass(obj, BaseEngine) and obj is not BaseEngine:
                        cls.register(obj)

    @classmethod
    def get_defaults(cls, name: str) -> dict:
        """读取 Engine 的 defaults.yaml 作为默认配置。"""
        engine_dir = Path(__file__).parent / name
        defaults_file = engine_dir / "defaults.yaml"
        if defaults_file.exists():
            import yaml
            return yaml.safe_load(defaults_file.read_text())
        return {}
```

**支持第三方 Engine 通过 Entry Points 注册**：
```toml
# 第三方 Engine 的 pyproject.toml
[project.entry-points."docmill.engines"]
my_ocr_engine = "my_package:MyOCREngine"
```

---

### 7. GPU 显存感知调度

**优先级: ★★★☆☆ | 工作量：约 3 天**

> 实现 DESIGN.md 中描述的 LRU + Watermark 调度策略。

**创建文件**：
```
docmill/orchestrator/
  scheduler.py       # GPU 显存感知调度器
  gpu_monitor.py     # GPU 显存监控
```

**`gpu_monitor.py`** — 基于 pynvml：
```python
class GPUMonitor:
    def get_total_vram_mb(self, gpu_id: int) -> float: ...
    def get_used_vram_mb(self, gpu_id: int) -> float: ...
    def get_free_vram_mb(self, gpu_id: int) -> float: ...
    def get_utilization(self, gpu_id: int) -> float: ...
    def list_gpus(self) -> list[dict]: ...
```

**`scheduler.py`** — 模型状态机 + LRU 驱逐：
```python
class GPUScheduler:
    def __init__(self, watermark_ratio: float = 0.9):
        """高水位线：默认 GPU 总显存的 90%"""
    
    def can_load(self, engine: BaseEngine, gpu_id: int) -> bool:
        """检查是否有足够显存加载模型（考虑 vLLM sidecar 显存）"""
    
    def evict_if_needed(self, required_mb: float, gpu_id: int) -> list[str]:
        """显存不足时按 LRU 驱逐 IDLE 模型（永不驱逐有活跃请求的模型）"""
    
    def on_request_start(self, model_name: str): ...   # IDLE → READY
    def on_request_end(self, model_name: str): ...     # READY → IDLE（如果无其他活跃请求）
```

**模型状态机**：
```
COLD → LOADING → READY → IDLE → EVICTED
                   ↑        │
                   └────────┘  (新请求唤醒)
```

---

### 8. 外部 vLLM 端点支持（混合模式）

**优先级: ★★★☆☆ | 工作量：约 2 天**

> 用户可能已有部署好的 vLLM 服务，DocMill 应同时支持"自动拉起"和"连接外部"两种模式。

**已在 P0.1 的 `VLLMConfig.mode` 字段中预留**，此任务需落地到代码：

```yaml
engines:
  paddle-ocr-vl-auto:
    engine: paddle_ocr_vl
    vllm:
      mode: managed               # DocMill 自动启动和管理
      model_path: /models/xxx

  paddle-ocr-vl-external:
    engine: paddle_ocr_vl
    vllm:
      mode: external              # 连接外部已有的 vLLM
      endpoints:
        - http://gpu-server-1:30023/v1
        - http://gpu-server-2:30023/v1
```

**改动范围**：
- `core.py` → `_start_vllm_sidecar()` 根据 mode 分支：managed 走 SidecarPool；external 直接使用 endpoints
- `orchestrator/sidecar_pool.py` → 添加 `register_external(endpoint)` 方法
- 外部端点也纳入健康检查机制

---

### 9. 输出标准化层

**优先级: ★★★☆☆ | 工作量：约 1.5 天**

> 不同 Engine 的输出格式差异较大，需要统一。

**创建文件**：
```
docmill/output/
  __init__.py
  normalizer.py      # 输出标准化器
  formats.py         # 统一输出格式定义
```

**统一输出目录结构**：
```
{output_dir}/{task_id}/
  result.md           # 合并后的 Markdown
  result.json         # 结构化 JSON
  pages/              # 每页的详细结果（可选）
    page_1/
    page_2/
```

**`normalizer.py`** 核心方法：
```python
class OutputNormalizer:
    def normalize(self, engine_output: EngineOutput, output_dir: Path) -> dict:
        """将 EngineOutput 标准化并保存。返回 {"result_path": str, "content": str}"""
    def merge_pages(self, page_markdowns: list[str]) -> str:
        """合并多页 Markdown（用 --- 分隔）"""
```

---

## 🟢 P2 — 锦上添花（体验优化与扩展性）

### 10. vLLM 端点池负载均衡

**优先级: ★★☆☆☆ | 工作量：约 2 天**

> 多个 vLLM 端点之间的智能负载均衡。

**功能**：
- Round-Robin / 最小连接数 / 响应时间感知策略
- 端点健康监测：自动移除不健康端点，恢复后自动加回
- 同时管理 managed 和 external 的端点

---

### 11. PDF 拆分与并行处理

**优先级: ★★☆☆☆ | 工作量：约 3 天**

> 大 PDF 文件（几百上千页）需要拆分后并行处理。

**实现规格**：
- 配置项：`pdf_split_threshold_pages`（默认 500）、`pdf_split_chunk_size`（默认 250）
- 页数超过阈值时自动拆分为多个子 PDF
- 每个子块作为独立子任务（child task）提交到 TaskStore
- 所有子任务完成后自动合并结果（合并 Markdown + JSON）
- 父子任务状态关联：子任务失败 → 父任务标记 failed
- 使用 TaskStore 的 `create_child()` / `on_child_completed()` / `on_child_failed()` 方法

---

### 12. 前端完善

**优先级: ★★☆☆☆ | 工作量：约 5 天**

> 当前 `frontend/` 目录只有 Vue 骨架。

**需要实现的页面**：
- 📄 文件上传页（支持拖拽上传 PDF/图片，选择 Engine）
- 📊 任务管理页（任务列表、状态筛选、进度条、取消/重试）
- 🖥️ 结果预览页（Markdown 渲染、PDF 原文对比查看）
- ⚙️ 系统管理页（GPU 状态、Engine 列表、vLLM Sidecar 状态）
- 📜 历史记录页（检索、下载结果、重新处理）

---

### 13. Docker 部署完善

**优先级: ★★☆☆☆ | 工作量：约 2 天**

> 当前 `docker/` 只有基础骨架。

**需要完善**：
- `Dockerfile.backend`: 多阶段构建、GPU 运行时依赖、模型缓存目录挂载
- `docker-compose.yml`: 环境变量注入、GPU 设备映射（`deploy.resources.reservations.devices`）、健康检查
- `.env.example`: 完整的环境变量说明文档
- `nginx.conf`: 反向代理完善（WebSocket 支持、上传大小限制）

---

### 14. 测试体系建设

**优先级: ★★☆☆☆ | 工作量：约 3 天**

> 当前几乎没有测试代码。

**创建文件**：
```
tests/
  conftest.py                # pytest fixtures（mock DocMill、临时数据库等）
  test_config.py             # 配置加载测试（YAML 解析、ENV 覆盖、默认值）
  test_engine_registry.py    # Engine 注册表测试（注册、发现、获取）
  test_task_store.py         # TaskStore 测试（CRUD、原子拉取、并发安全）
  test_sidecar_pool.py       # SidecarPool 测试（mock vLLM 进程）
  test_api.py                # API 集成测试（FastAPI TestClient）
  test_e2e.py                # 端到端测试（提交任务 → 等待完成 → 验证结果）
```

---

### 15. 日志与监控

**优先级: ★☆☆☆☆ | 工作量：约 1.5 天**

**功能**：
- 结构化日志输出（JSON 格式，可接入 ELK 等日志系统）
- vLLM Sidecar 日志聚合（当前日志写到文件，不便查看）
- GPU 使用率定期上报
- 可选的 Prometheus metrics 端点（`/metrics`）

---

### 16. CLI 工具丰富化

**优先级: ★☆☆☆☆ | 工作量：约 2 天**

> 除了 `docmill serve`，还需要一些管理命令。

```bash
docmill serve   --config docmill.yaml      # 启动服务
docmill status                              # 查看服务状态
docmill models  list                        # 列出已注册模型
docmill models  add    <name> --config ...  # 运行时添加模型
docmill models  remove <name>               # 移除模型
docmill sidecar list                        # 查看 vLLM Sidecar 状态
docmill sidecar logs   <endpoint>           # 查看 Sidecar 日志
docmill infer   --model <name> --file <path> # 命令行推理
docmill config  validate <file>             # 验证配置文件
```

---

## 📋 总览：工作量估算

| 优先级 | 任务数 | 预计工作量 | 说明 |
|:-------|:------:|:---------:|:-----|
| 🔴 P0 | 5 项 | 约 14 天 | 没有这些无法正常使用 |
| 🟡 P1 | 4 项 | 约 8 天 | 生产环境必须 |
| 🟢 P2 | 7 项 | 约 18.5 天 | 体验优化与扩展 |
| **合计** | **16 项** | **约 40.5 天** | -- |

> [!TIP]
> **建议执行顺序**: 1 → 2 → 3 → 4 → 5 → 6 → 8 → 7 → 9 → 11 → 10 → 12 → 13 → 14 → 15 → 16
>
> P0（1-5）全部完成后，DocMill 即可跑通"一键启动 + 异步任务 + 多 GPU 并发"的核心流程。

---

## 🗂️ 目标目录结构

完成所有工作后，DocMill 的理想目录结构：

```
docmill/
  __init__.py
  core.py                    # ← 已有

  config/                    # 🔴 P0.1 新增
    __init__.py
    schema.py                # Pydantic 配置模型
    loader.py                # 多源配置加载器

  launcher/                  # 🔴 P0.2 新增
    __init__.py
    app_launcher.py          # 一键启动器 + CLI 入口

  engines/                   # P0.5 重构为文件夹 + P1.6 增强
    __init__.py
    base.py
    registry.py              # + auto_discover(), entry points
    paddle_ocr_vl/           # 🔴 P0.5: 从单文件重构为文件夹
      __init__.py
      engine.py
      defaults.yaml
      README.md
    deepseek_ocr/            # 🔴 P0.5: 同上
      __init__.py
      engine.py
      defaults.yaml
      README.md

  clients/                   # ← 已有
    __init__.py
    base.py
    openai_compat.py

  orchestrator/              # ← 已有，P1.7 增强
    __init__.py
    launcher.py
    sidecar_pool.py
    scheduler.py             # 🟡 P1.7 新增: GPU 感知调度
    gpu_monitor.py           # 🟡 P1.7 新增: 显存监控

  tasks/                     # 🔴 P0.3 新增
    __init__.py
    task_store.py
    task_manager.py
    task_worker.py

  workers/                   # 🔴 P0.4 新增
    __init__.py
    pool.py
    worker_process.py
    gpu_allocator.py

  output/                    # 🟡 P1.9 新增
    __init__.py
    normalizer.py
    formats.py

  server/                    # ← 已有，需扩展 task 相关路由
    __init__.py
    main.py
    routes/
      ...
      tasks.py               # 🔴 P0.3 新增

  storage/                   # ← 已有
    __init__.py
    file_store.py
    history_store.py

  utils/                     # ← 已有
    __init__.py
    errors.py
    logging.py
    ports.py
```
