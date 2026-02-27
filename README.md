# DocMill

**DocMill** is a unified inference runtime for OCR & VLM document understanding — not a general-purpose AI platform.

It focuses exclusively on OCR and VLM workloads, designed to run, manage, and orchestrate document understanding models as a single runtime.

## ✨ 特性

- 🔧 **声明式配置** — 模型通过 YAML 声明，而非硬编码
- 🚀 **vLLM 自动编排** — 自动拉起/管理 vLLM sidecar 进程，用户无感知
- 🧠 **GPU 资源调度** — LRU + Watermark 策略，自动管理模型生命周期
- 🔌 **三种推理形态** — vision_only / vision_llm / llm_only，覆盖所有 OCR 场景
- 📦 **统一 API** — 一个 FastAPI 入口，不同模型统一接口

## 🏗️ 架构

```
┌──────────────────────────────┐
│       FastAPI Server         │  ← API 入口
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│     Pipeline Engine          │  ← 稳定：3 种推理形态
│  vision_only / vision_llm    │
│         / llm_only           │
└──────┬───────────┬───────────┘
       │           │
┌──────▼───────┐ ┌─▼────────────┐
│  Workers     │ │  LLM Client  │  ← 变化：可替换组件
│ (local GPU)  │ │ (HTTP API)   │
└──────────────┘ └──────┬───────┘
                        │
┌───────────────────────▼──────┐
│    Orchestrator              │  ← 核心：vLLM 生命周期
│  planner / launcher / pool   │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│   vLLM / GPU Process         │  ← Runtime
└──────────────────────────────┘
```

## 🚀 Quick Start

### 安装

```bash
pip install -e .
```

### 启动服务

```bash
# 注册模型目录并启动
uvicorn docmill.server.api:app --host 0.0.0.0 --port 8080
```

### 注册模型

```bash
# 注册模型目录
curl -X POST "http://localhost:8080/models/register?models_dir=docmill/models"

# 或注册单个模型
curl -X POST "http://localhost:8080/models/register_one?config_path=docmill/models/deepseek_ocr/config.yaml"
```

### 推理

```bash
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-ocr", "file_path": "/path/to/image.png"}'
```

## 📁 项目结构

```
docmill/
├── config/          # 配置系统 (Pydantic Schema + YAML 加载)
├── pipelines/       # Pipeline 抽象 (vision_only / vision_llm / llm_only)
├── execution/       # 执行策略 (local / hybrid / remote)
├── orchestrator/    # vLLM Sidecar 编排 (launcher / health / planner / pool)
├── workers/         # 本地推理 Worker (torch / paddle)
├── llm_clients/     # LLM HTTP 客户端 (OpenAI 兼容)
├── models/          # 模型配置 + Hooks
│   ├── mineru_ocr/      # MinerU: hybrid (Vision + vLLM)
│   ├── paddle_ocr_vl/   # PaddleOCR-VL: hybrid (Paddle + vLLM)
│   └── deepseek_ocr/    # DeepSeek OCR: remote (纯 vLLM)
├── server/          # FastAPI 入口
└── utils/           # 工具 (硬件检测 / 端口管理 / 日志 / 异常)
```

## 📐 已集成模型

| 模型 | Pipeline | 执行方式 | 特点 |
|:-----|:---------|:--------|:-----|
| MinerU OCR | vision_llm | hybrid | 两阶段：Layout 分析 + VLM 精细识别 |
| PaddleOCR-VL | vision_llm | hybrid | NaViT Vision Encoder + vLLM Decoder |
| DeepSeek OCR | llm_only | remote | 3B MoE，纯 vLLM，2500 tok/s |

## 🔧 添加新模型

1. 在 `docmill/models/` 下创建目录
2. 编写 `config.yaml`（声明 pipeline / execution / 资源参数）
3. 编写 `hooks.py`（pre/post 处理逻辑）
4. 注册模型并使用

```yaml
# config.yaml 示例
name: "my-ocr-model"
pipeline: "vision_llm"    # vision_only / vision_llm / llm_only
execution: "hybrid"       # local / hybrid / remote
vision:
  framework: "torch"
  model_path: "/path/to/vision/weights"
llm:
  backend: "vllm"
  model_path: "/path/to/llm/weights"
resources:
  gpu_memory_utilization: 0  # 0 = 自动计算
```

## License

GPL-3.0
