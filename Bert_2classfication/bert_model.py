import torch.nn as nn
from transformers import BertModel

from config import Config

config = Config()


class BertClassifierModel(nn.Module):
    """BERT + 线性分类头，二分类意图识别。"""

    def __init__(self, freeze_backbone=None):
        super().__init__()
        freeze_backbone = config.freeze_backbone if freeze_backbone is None else freeze_backbone

        self.bert = BertModel.from_pretrained(config.bert_model_path)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_labels)

        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
