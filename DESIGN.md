# DocMill 架构设计文档

> DocMill is a unified inference runtime for OCR & VLM document understanding —
> not a general-purpose AI platform.

## 1. 定位与边界

### 做什么
- 统一管理 OCR 和 VLM 模型的推理生命周期
- 自动编排 vLLM sidecar 进程（用户无感知）
- GPU 显存感知的模型调度（LRU + Watermark）
- 配置驱动的模型声明（YAML，非硬编码）

### 明确不做
- ❌ 视频帧抽取 / 生物学解析 / 音频处理
- ❌ 模型训练 / 微调
- ❌ 多租户 RBAC / 复杂用户系统
- ❌ Kubernetes / Ray / 微服务编排
- ❌ 模型下载市场 / Hub

## 2. 架构总览

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

## 3. 三条架构戒律

1. **Pipeline 不允许出现在 `models/` 中。**
   模型目录只放 `config.yaml` + `hooks.py`（pre/post 适配逻辑）。
   流程拓扑由 `pipelines/` 统一定义。

2. **Core 不允许直接 import 具体 ML 框架。**
   `torch`、`paddle` 等只在 `workers/` 中出现。
   Pipeline 和 Orchestrator 对框架无感知。

3. **Sidecar 只能通过 HTTP 访问。**
   vLLM 是"外部运行时"，等同于数据库。
   Pipeline 永远不直接操作 vLLM 进程。

## 4. 三种推理形态（Inference Shape）

| Pipeline      | 本地 Worker | LLM Sidecar | 典型模型         |
|:-------------|:----------:|:-----------:|:----------------|
| vision_only  | ✅          | ❌           | 纯 OCR / Layout  |
| vision_llm   | ✅          | ✅           | PaddleOCR-VL    |
| llm_only     | ❌          | ✅           | DeepSeek OCR    |

## 5. 三种执行策略（Execution Strategy）

| 策略            | 本地进程 | Sidecar | 调用方式        |
|:---------------|:------:|:------:|:--------------|
| LocalExecution  | ✅      | ❌      | Python call   |
| HybridExecution | ✅      | ✅      | Python + HTTP |
| RemoteExecution | ❌      | ✅      | HTTP only     |

## 6. 模型运行时状态机

```
COLD → LOADING → READY → IDLE → EVICTED
                   ↑        │
                   └────────┘  (新请求唤醒)
```

- **COLD**: 未加载
- **LOADING**: 权重加载中 / vLLM 启动中
- **READY**: 可接受请求
- **IDLE**: 空闲但未卸载（等待复用）
- **EVICTED**: 已释放显存 / 进程已终止

## 7. 调度策略：LRU + Watermark

- 高水位线：默认 GPU 总显存的 90%
- 加载新模型前：估算所需显存
- 超出水位 → 按 LRU 顺序驱逐 IDLE 模型
- 永不驱逐 READY（有活跃请求的）模型
- 显存不足且无可驱逐 → 返回 `InsufficientVRAM` 错误

## 8. 目标模型映射

| 模型            | Pipeline     | Execution | 特点                           |
|:---------------|:------------|:----------|:------------------------------|
| MinerU         | vision_llm  | hybrid    | 两阶段：layout 分析 + VLM 精细识别 |
| PaddleOCR-VL   | vision_llm  | hybrid    | Vision Encoder + vLLM decoder  |
| DeepSeek OCR   | llm_only    | remote    | 纯 vLLM（DeepEncoder + MoE）   |
