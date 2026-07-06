# Models

本地模型目录。模型权重**不提交到 Git**（体积过大），请在本机或 `config.ini` 配置路径。

## 意图分类（可选）

训练产出保存在 `models/bert_outputs/`：

| 文件 | 说明 |
|------|------|
| `best_intent_classifier.pt` | 浮点 checkpoint（运行时默认加载） |
| `best_intent_classifier_int8.pt` | INT8 量化版（`quantize_intent_classifier.py` 生成） |

训练与量化见 [intent_classification/README.md](../intent_classification/README.md)。  
若本地没有 checkpoint，主项目会回退到 LLM API 做意图分类。

## 检索模型（必需）

请在 `config.ini` 配置本地路径：

| 模型 | 配置项 |
|------|--------|
| `bge-m3` | `bge_m3_path` |
| `bge-reranker-large` | `bge_reranker_path` |
| `bert-base-chinese` | `bert_base_path`（仅训练意图分类时需要） |

```ini
[models]
bge_m3_path = /path/to/bge-m3
bge_reranker_path = /path/to/bge-reranker-large
bert_base_path = /path/to/bert-base-chinese
bert_classifier_path = models/bert_outputs
```

## 标签映射

- `0` = 通用知识
- `1` = 专业咨询
