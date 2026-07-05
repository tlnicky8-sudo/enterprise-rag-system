# Models

This directory is for local model files. Model weights are intentionally not committed.

Expected retrieval model paths:

- `models/bge-m3`
- `models/bge-reranker-large`

## Intent classifier

The BERT intent classifier is optional. If no trained checkpoint exists under
`models/bert_outputs/`, the QA pipeline falls back to LLM-based intent
classification and finally defaults to `专业咨询`.

Train under `Bert_2classfication/`:

```bash
cd Bert_2classfication
python train_intent_classifier.py
```

The best checkpoint is saved to:

```text
models/bert_outputs/best_intent_classifier.pt
```

The QA pipeline loads this checkpoint automatically through `core/query_classifier.py`
when it exists.

Label mapping:

- `0` = 通用知识
- `1` = 专业咨询
