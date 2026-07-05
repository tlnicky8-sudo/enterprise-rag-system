# QA Projector

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

面向中文劳动法问答的 RAG 系统。提供 Flask Web 聊天界面，使用 Milvus + BGE-M3 混合检索与重排序，并通过 OpenAI-compatible 接口调用 DashScope / Qwen / DeepSeek 等模型生成答案。

> 将 README 中的 `OWNER/REPO` 替换为你的 GitHub 仓库路径后，CI 徽章即可正常显示。

## Documentation

| 文档 | 说明 |
|------|------|
| **[docs/data-and-cache-usage.md](docs/data-and-cache-usage.md)** | **数据处理、FAQ 入库、问答缓存 — 怎么用（推荐先看）** |
| [models/README.md](models/README.md) | 本地模型路径与意图分类训练 |

## Features

- **双层 QA 流水线** — Redis FAQ 快路径 + Milvus RAG 回退
- **可信回答** — Grounding gate、证据引用 `[1][2]`、低置信度拒答
- **智能检索策略** — 直接检索 / HyDE / 子查询 / 回溯问题（LLM 自动选择）
- **六步语料入库** — 解析 → 清洗 → 切块 → 增强 → 索引 → 血缘报告
- **FAQ 高频问答** — JSONL 入库 Redis，语义缓存 + BM25 双路命中
- **统一后端配置** — `config.ini`（基础设施）+ `config/runtime.yaml`（策略开关）+ `config/prompts/`（提示词）

## Table of Contents

- [Documentation](#documentation)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Data Ingestion](#data-ingestion)
- [Run](#run)
- [Evaluation](#evaluation)
- [Project Layout](#project-layout)
- [Contributing](#contributing)
- [License](#license)

## Architecture

```text
Query
  → FAQ (Redis semantic cache → BM25)
  → miss → Intent (通用知识 / 专业咨询)
      → 通用知识: LLM 直接回答
      → 专业咨询: strategy → Milvus hybrid retrieval → grounding gate → LLM (JSON)
  → cache write (if grounded + score ≥ threshold)
  → answer + citations
```

**数据存储**

| 数据 | 存储 | 入库脚本 |
|------|------|----------|
| 法律语料（检索用） | Milvus | `setup_data.py` |
| 高频问答对 | Redis | `setup_faq_data.py` |

详细入库说明见 **[docs/data-and-cache-usage.md](docs/data-and-cache-usage.md)**。

## Quick Start

### Prerequisites

- Python 3.10 – 3.13
- [Milvus](https://milvus.io/)（默认 `localhost:19530`）
- LLM API Key（DashScope / DeepSeek 等 OpenAI-compatible 服务）
- Redis（FAQ 快路径；不可用时降级为纯 RAG）
- 本地检索模型：`bge-m3`、`bge-reranker-large`（意图分类 BERT 可选，缺失时走大模型兜底；见 [models/README.md](models/README.md)）

### Install

```bash
git clone https://github.com/OWNER/REPO.git
cd rag_projector

uv venv && uv sync --extra retrieval --extra documents --extra faq --extra dev
# 或: pip install -r requirements.txt && pip install -r requirements-dev.txt
```

### Configure

```bash
cp config.example.ini config.ini
cp config/runtime.example.yaml config/runtime.yaml
cp .env.example .env
```

在 `.env` 中填写 API Key：

```bash
DASHSCOPE_API_KEY=your_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### Ingest Data

```bash
# 1. 将法律文档放入 data/labor_law_data/
python setup_data.py

# 2. 准备 FAQ 并入库 Redis
mkdir -p data/faq_data
cp examples/faq_pairs.example.jsonl data/faq_data/faq_pairs.jsonl
python setup_faq_data.py
```

### Run

```bash
python web_app.py    # http://localhost:5000
python rag_main.py   # CLI
```

## Configuration

| 文件 | 用途 |
|------|------|
| `.env` | API Key、敏感连接信息（优先于 config.ini） |
| `config.ini` | Redis / Milvus / 模型路径 / 分块参数 |
| `config/runtime.yaml` | 功能开关、召回数、阈值、策略默认值 |
| `config/prompts/*.txt` | RAG / HyDE / 策略选择等提示词模板 |

`config.ini` 与 `config/runtime.yaml` 不会提交到 Git，请从 `*.example.*` 复制。

意图识别优先加载 `models/bert_outputs/best_intent_classifier.pt`。如果本地没有训练模型，会自动使用大模型按 `config/prompts/intent.txt` 分类；大模型不可用时默认走 `专业咨询`。

环境变量覆盖示例：

```bash
RUNTIME_RETRIEVAL_RETRIEVAL_K=8
RUNTIME_GROUNDING_MIN_RERANK_SCORE=0.4
RUNTIME_CONFIG=/path/to/runtime.yaml
```

## Data Ingestion

> 完整操作说明（含缓存命中逻辑、写回策略、调参建议）见 **[docs/data-and-cache-usage.md](docs/data-and-cache-usage.md)**。

### RAG 语料入库 — `setup_data.py`

将 PDF / Word / Markdown 等放入 `data/labor_law_data/`，执行六步流水线写入 Milvus：

```bash
python setup_data.py
python setup_data.py --dry-run          # 仅校验流程
python setup_data.py --enhance          # 开启关键词 / 假设问题增强
python setup_data.py --skip-if-exists   # 集合已有数据则跳过
```

### 高频 FAQ 入库 — `setup_faq_data.py`

将问答对写入 `data/faq_data/faq_pairs.jsonl`（JSONL，每行 `{"question":"...","answer":"..."}`），导入 Redis 并预热语义缓存：

```bash
mkdir -p data/faq_data
cp examples/faq_pairs.example.jsonl data/faq_data/faq_pairs.jsonl
python setup_faq_data.py
python setup_faq_data.py --replace       # 清空旧数据后重导
python setup_faq_data.py --dry-run       # 仅解析校验
```

### 问答缓存

- **读**：用户提问 → 语义缓存 → BM25 → 未命中再走 RAG（见使用说明第四节）
- **写**：RAG 高质量回答自动写回 Redis（`enable_cache_write`，默认开启）
- **维护**：`python scripts/preheat_faq_cache.py` 可手动重新预热向量缓存

## Run

| 入口 | 命令 | 说明 |
|------|------|------|
| Web | `python web_app.py` | Flask 聊天界面，SSE 流式输出 |
| CLI | `python rag_main.py` | 命令行问答 |

会话历史保存在进程内存中，重启后清空。

## Evaluation

### Live pipeline eval

真实调用 `QAPipeline.answer()`，覆盖 FAQ、Milvus、rerank、grounding gate 和 LLM：

```bash
python scripts/live_eval.py
python scripts/live_eval.py --limit 3
python scripts/live_eval.py --fail-under-pass-rate 0.8
```

Golden set：`data/assesment_data/live_eval_golden.jsonl`  
结果输出：`data/assesment_data/live_eval_results/`

### RAGAS offline eval

```bash
python rag_evaluate.py
```

使用静态 JSON，不调用实时流水线，适合离线分析。

## Project Layout

```text
.
├── base/                    # 配置、日志、Prompt 注册
├── config/
│   ├── prompts/             # 提示词模板（.txt）
│   └── runtime.example.yaml # 运行时策略示例
├── core/
│   ├── qa_pipeline.py       # FAQ + RAG 统一编排
│   ├── rag_system.py        # 意图 / 策略 / 生成 / grounding
│   ├── vector_store.py      # Milvus + BGE-M3 + Reranker
│   ├── faq/                 # Redis FAQ / BM25 / 缓存
│   └── ingest/              # 六步入库流水线
├── data/
│   ├── labor_law_data/      # RAG 原始语料（本地创建）
│   └── faq_data/            # FAQ 问答对（本地创建）
├── docs/
│   └── data-and-cache-usage.md  # 数据处理 / FAQ / 缓存
├── examples/
│   └── faq_pairs.example.jsonl  # FAQ 示例数据
├── scripts/                 # live_eval 等辅助脚本
├── setup_data.py            # RAG 语料入库入口
├── setup_faq_data.py        # FAQ 入库入口
├── web_app.py               # Web 入口
└── tests/                   # Smoke tests
```

## Tests

```bash
pytest
```

轻量测试不加载 Milvus、大模型或 DashScope API。

## Contributing

请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。提交 PR 前请运行 `pytest`，勿提交 `.env`、`config.ini`、模型权重或完整语料。

## License

[MIT](LICENSE)

## Publishing Checklist

首次公开仓库前确认未提交：

- `.venv/`、`.idea/`、`.env`、`config.ini`、`config/runtime.yaml`
- `logs/`、`models/` 下的模型权重
- `data/` 下的完整语料与 FAQ 数据
