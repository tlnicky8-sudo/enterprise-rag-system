"""Runtime intent classifier used by the QA pipeline."""

from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer


class BertClassifierModel(nn.Module):
    """BERT + linear head, matching intent_classification training output."""

    def __init__(self, bert_source, num_labels=2):
        super().__init__()
        # Prefer config-only init when loading a full state_dict later.
        if Path(bert_source).is_file() and Path(bert_source).suffix == ".pt":
            raise ValueError("bert_source must be a model directory, not a .pt file")

        config_path = Path(bert_source)
        if (config_path / "config.json").exists():
            bert_config = BertConfig.from_pretrained(bert_source)
            self.bert = BertModel(bert_config)
        else:
            self.bert = BertModel.from_pretrained(bert_source)

        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = self.dropout(outputs.pooler_output)
        return self.classifier(pooled_output)


def resolve_checkpoint_path(model_path):
    path = Path(model_path)
    if path.is_file() and path.suffix == ".pt":
        return path
    candidates = [
        path / "best_intent_classifier.pt",
        path / "pytorch_model.bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_tokenizer_source(model_path, bert_base_path=None):
    path = Path(model_path)
    model_dir = path.parent if path.is_file() else path

    # Prefer local tokenizer files next to the checkpoint.
    if (model_dir / "tokenizer.json").exists() or (model_dir / "vocab.txt").exists():
        return str(model_dir)
    if bert_base_path and Path(bert_base_path).exists():
        return bert_base_path
    return str(model_dir)


def load_intent_classifier(model_path, bert_base_path=None, device="cpu", num_labels=2):
    """Load trained intent classifier from directory or .pt checkpoint."""
    model_path = Path(model_path)
    checkpoint = resolve_checkpoint_path(model_path)
    tokenizer_source = resolve_tokenizer_source(model_path, bert_base_path=bert_base_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_source)

    # Architecture source: checkpoint dir config, else base BERT.
    architecture_source = (
        str(model_path.parent if model_path.is_file() else model_path)
        if (Path(model_path.parent if model_path.is_file() else model_path) / "config.json").exists()
        else (bert_base_path or tokenizer_source)
    )

    model = BertClassifierModel(architecture_source, num_labels=num_labels)
    if checkpoint is None:
        raise FileNotFoundError(
            f"Intent checkpoint not found under {model_path}. "
            "Expected best_intent_classifier.pt"
        )

    try:
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer, str(checkpoint)
