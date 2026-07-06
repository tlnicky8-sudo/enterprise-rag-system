# Intent Classifier (BERT)

可选的 **二分类意图识别** 训练模块，用于区分用户问题是走「通用知识」还是「专业咨询」。

> **不训练也能用。** 模型权重与训练语料**不提交到 Git**。若本地没有 `models/bert_outputs/best_intent_classifier.pt`，主项目会自动通过 **LLM API** 做意图分类；API 不可用时默认按 **专业咨询** 处理。

## Role in the QA Pipeline

```text
用户问题
  → [1] 本地 BERT 分类（若存在 models/bert_outputs/best_intent_classifier.pt）
  → [2] LLM API 分类（config/prompts/intent.txt，默认开启）
  → [3] 兜底：专业咨询
```

| 分类结果 | 运行时行为 |
|----------|------------|
| **通用知识** | 跳过 Milvus 检索，由 LLM 直接回答 |
| **专业咨询** | 进入检索策略 → Milvus → Grounding → LLM 生成 |

实现入口：[`core/query_classifier.py`](../core/query_classifier.py)

## When to Train

建议在以下情况训练本地 BERT 分类器：

- 希望降低意图识别对 LLM API 的依赖与延迟
- 有稳定的标注数据，且分类边界已明确
- 生产环境需要可离线、可复现的意图判断

若只是本地体验或 Demo，**可跳过训练**，配置好 `DASHSCOPE_API_KEY` 即可。

## Labels

| Label | ID | 示例 |
|-------|----|------|
| 通用知识 | `0` | 「5×9 等于多少？」「怎么和陌生人聊天？」 |
| 专业咨询 | `1` | 「年假有多少天？」「VPN 怎么配置？」 |

## Quick Start

### 1. Prepare training data

在本地生成或准备 `train_data/`（不提交 Git）：

```text
intent_classification/train_data/
  train.jsonl
  test.jsonl
```

```bash
python scripts/generate_intent_data.py
```

每行 JSON 示例：

```json
{"query": "天恒科技年假有多少天？", "label": "专业咨询"}
{"query": "今天天气怎么样？", "label": "通用知识"}
```

### 2. Configure paths

编辑项目根目录的 `config.ini`（推荐）或 [`config.py`](config.py)：

| 项 | 配置项 |
|----|--------|
| 预训练 BERT | `config.ini` → `[models]` → `bert_base_path` |
| 输出 checkpoint | `models/bert_outputs/best_intent_classifier.pt` |

训练脚本会**自动读取 `config.ini` 的 `bert_base_path`**，模型可以放在仓库外任意目录（例如 `/Users/tl/Documents/models/bert-base-chinese`），不必复制到 `rag_projector/models/` 下。

仅当本地路径和 `config.ini` 都未配置时，才会回退到 HuggingFace Hub 的 `bert-base-chinese`。

### 3. Train

```bash
cd intent_classification
python train_intent_classifier.py
```

训练过程中会按 test 集 F1 保存最优模型到：

```text
models/bert_outputs/best_intent_classifier.pt
```

### 4. Quantize (optional)

训练完成后，可将模型动态量化为 INT8，减小体积并加快 **CPU** 推理：

```bash
cd intent_classification
python quantize_intent_classifier.py
```

默认输出：

```text
models/bert_outputs/best_intent_classifier_int8.pt
```

脚本会对比量化前后的 F1、准确率和文件大小。量化模型主要用于 CPU 部署；主项目运行时仍默认加载浮点 `best_intent_classifier.pt`。

### 5. Run the main app

回到项目根目录启动 Web / CLI，流水线会自动加载上述 checkpoint：

```bash
python web_app.py
```

## Runtime Fallback (No Local Model)

未训练或未放置模型时，行为由 [`config/runtime.example.yaml`](../config/runtime.example.yaml) 控制：

```yaml
features:
  enable_llm_intent_fallback: true   # 无 BERT 时走 LLM API

intent:
  fallback_category: 专业咨询        # API 失败时的安全默认值
```

相关提示词：[`config/prompts/intent.txt`](../config/prompts/intent.txt)

## Files

| File | Description |
|------|-------------|
| `config.py` | 训练超参与路径 |
| `bert_model.py` | BERT + 线性分类头 |
| `data_preprocessing.py` | JSONL 加载与 DataLoader |
| `train_intent_classifier.py` | 训练与评估入口 |
| `quantize_intent_classifier.py` | INT8 量化入口 |

## See Also

- [Main README](../README.md) — 项目总览与快速开始
- [models/README.md](../models/README.md) — 本地模型目录说明
- [docs/data-and-cache-usage.md](../docs/data-and-cache-usage.md) — 数据入库与缓存使用
