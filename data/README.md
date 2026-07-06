# Data

本地语料与评估数据目录。**完整语料不会提交到 Git**，仓库只保留这个占位说明。

## 使用说明（推荐）

数据处理、FAQ 入库、问答缓存的完整操作指南：

**[docs/data-and-cache-usage.md](../docs/data-and-cache-usage.md)**

## 目录说明

| 路径 | 用途 | 入库命令 |
|------|------|----------|
| `enterprise_data/` | RAG 企业文档语料（PDF / Word / MD 等） | `python setup_data.py` |
| `faq_data/faq_pairs.jsonl` | 高频问答对（本地自建） | `python setup_faq_data.py` |
| `assessment_data/` | 评估集与评估输出（本地自建） | `python scripts/live_eval.py` |
| `processed/` | 入库中间产物（Markdown） | `setup_data.py` 自动生成 |
| `ingest_reports/` | 入库血缘报告 | `setup_data.py` 自动生成 |

## 快速开始

```bash
# 1. 企业文档 → data/enterprise_data/
python setup_data.py

# 2. FAQ 问答对
mkdir -p data/faq_data
cp examples/faq_pairs.example.jsonl data/faq_data/faq_pairs.jsonl
# 编辑 faq_pairs.jsonl 后：
python setup_faq_data.py

# 3. 可选：仅重新预热语义缓存
python scripts/preheat_faq_cache.py
```

## 意图识别训练数据

```text
intent_classification/train_data/
  train.jsonl
  test.jsonl
```

```bash
python scripts/generate_intent_data.py
```
