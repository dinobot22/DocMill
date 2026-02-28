# DocMill

**DocMill** is a unified inference runtime for OCR & VLM document understanding — not a general-purpose AI platform.

It focuses exclusively on OCR and VLM workloads, designed to run, manage, and orchestrate document understanding models as a single runtime.

## ✨ 特性

- 🚀 **vLLM 自动编排** — 自动拉起/管理 vLLM sidecar 进程，用户无感知
- 🧠 **GPU 资源调度** — LRU + Watermark 策略，自动管理模型生命周期
- 🔌 **Engine 注册机制** — 基于代码注册模型，灵活可扩展
- 📦 **统一 API** — 一个 FastAPI 入口，不同模型统一接口
- 🖥️ **Vue 前端** — 现代化 Web UI，支持文件上传、模型管理、历史记录

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────┐
│                   Vue Frontend                       │
│         (Home / Models / History)                   │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP API
┌─────────────────────▼───────────────────────────────┐
│              FastAPI Server                          │
│    /models /infer /history /files                   │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                 DocMill Core                         │
│         Engine Registry + Model Manager             │
└──────────┬───────────────────────┬──────────────────┘
           │                       │
┌──────────▼──────────┐   ┌────────▼─────────┐
│  Engines            │   │  vLLM Sidecar    │
│  PaddleOCR-VL      │   │  Pool Manager    │
│  DeepSeek OCR      │   │  Launcher        │
│  ...               │   │                  │
└─────────────────────┘   └──────────────────┘
```

## 📁 项目结构

```
DocMill/
├── docmill/                    # Python 后端包
│   ├── core.py                # 主类
│   ├── engines/               # Engine 实现
│   │   ├── base.py            # 基类
│   │   ├── registry.py        # 注册表
│   │   ├── paddle_ocr_vl.py
│   │   └── deepseek_ocr.py
│   ├── orchestrator/          # vLLM Sidecar 管理
│   │   ├── launcher.py
│   │   └── sidecar_pool.py
│   ├── server/                # FastAPI 服务
│   │   ├── main.py
│   │   └── routes/
│   │       ├── models.py      # 模型管理
│   │       ├── infer.py       # 推理
│   │       ├── history.py     # 历史记录
│   │       └── files.py       # 文件上传
│   ├── storage/               # 存储
│   │   ├── file_store.py
│   │   └── history_store.py
│   ├── clients/               # LLM 客户端
│   └── utils/                 # 工具
├── frontend/                   # Vue 前端
│   ├── src/
│   │   ├── views/
│   │   │   ├── Home.vue       # OCR 上传
│   │   │   ├── Models.vue     # 模型管理
│   │   │   └── History.vue    # 历史记录
│   │   ├── api/               # API 封装
│   │   ├── stores/            # Pinia 状态
│   │   └── router.ts
│   ├── package.json
│   └── vite.config.ts
├── docker/                     # Docker 部署
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   ├── nginx.conf
│   └── docker-compose.yml
├── examples/                   # 使用示例
├── pyproject.toml              # Python 项目配置
└── README.md
```

## 🚀 Quick Start

### 方式一：开发模式

**后端：**
```bash
pip install -e .
uvicorn docmill.server.main:app --reload --port 8080
```

**前端：**
```bash
cd frontend
npm install
npm run dev
# 访问 http://localhost:5173
```

### 方式二：Docker 部署

```bash
cd docker
docker-compose up -d
# 访问 http://localhost:80
```

## 📐 API 使用

### 注册模型

```bash
# 注册 DeepSeek OCR 模型
curl -X POST http://localhost:8080/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "deepseek-ocr",
    "engine_name": "deepseek_ocr",
    "vllm_config": {
      "model_path": "/models/deepseek-ocr",
      "gpu_memory_utilization": 0.8
    }
  }'

# 注册 PaddleOCR-VL 模型
curl -X POST http://localhost:8080/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "paddle-ocr-vl",
    "engine_name": "paddle_ocr_vl",
    "vllm_config": {
      "model_path": "/models/paddleocr-vl-llm"
    },
    "engine_kwargs": {
      "vllm_model_path": "/models/paddleocr-vl-llm"
    }
  }'
```

### 上传文件并推理

```bash
# 1. 上传文件
FILE_ID=$(curl -s -X POST http://localhost:8080/files/upload \
  -F "file=@test.png" | jq -r '.file_id')

# 2. 执行推理
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"deepseek-ocr\", \"file_id\": \"$FILE_ID\"}"
```

### API 端点

| 端点 | 方法 | 说明 |
|:-----|:-----|:-----|
| `/health` | GET | 健康检查 |
| `/models` | GET | 列出模型 |
| `/models/register` | POST | 注册模型 |
| `/models/{name}/load` | POST | 加载模型 |
| `/models/{name}/unload` | POST | 卸载模型 |
| `/models/engines/list` | GET | 列出 Engine 类型 |
| `/infer` | POST | 执行推理 |
| `/files/upload` | POST | 上传文件 |
| `/history` | GET | 历史记录列表 |
| `/history/{id}/download` | GET | 下载结果 |

## 🔧 添加新 Engine

1. 创建 Engine 类继承 `BaseEngine`:

```python
# backend/docmill/engines/my_engine.py
from docmill.engines.base import BaseEngine, EngineInput, EngineOutput

class MyEngine(BaseEngine):
    @classmethod
    def name(cls) -> str:
        return "my_engine"

    @classmethod
    def requires_vllm_sidecar(cls) -> bool:
        return True  # 或 False

    def _do_infer(self, input: EngineInput) -> EngineOutput:
        # 实现推理逻辑
        return EngineOutput(text="result")
```

2. 注册 Engine:

```python
# backend/docmill/engines/__init__.py
from .my_engine import MyEngine
EngineRegistry.register(MyEngine)
```

## 📦 已集成 Engine

| Engine | 需要 vLLM | 特点 |
|:-------|:---------|:-----|
| deepseek_ocr | ✅ | DeepSeek OCR，纯 vLLM，高性能 |
| paddle_ocr_vl | ✅ | PaddleOCR-VL，NaViT + vLLM |

## License

GPL-3.0