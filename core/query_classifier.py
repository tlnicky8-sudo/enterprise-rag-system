import json
import os

from base import logger
from core.intent_runtime import load_intent_classifier
from core.prompts import RAGPrompts


VALID_INTENT_LABELS = {"通用知识", "专业咨询"}


def _resolve_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class QueryClassifier:
    """Binary intent classifier: 通用知识 vs 专业咨询."""

    def __init__(self, model_path=None, bert_base_path=None, conf=None):
        from base import Config

        self.conf = conf or Config()
        if model_path is None:
            model_path = self.conf.BERT_CLASSIFIER_PATH
        if bert_base_path is None:
            bert_base_path = getattr(self.conf, "BERT_BASE_PATH", None)

        self.model_path = model_path
        self.bert_base_path = bert_base_path
        self.model = None
        self.tokenizer = None
        self.checkpoint_path = None
        self.device = "cpu"
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        self.id_to_label = {0: "通用知识", 1: "专业咨询"}
        self.llm_client = None
        logger.info(f"意图识别使用设备: {self.device}")
        self.load_model()

    def load_model(self):
        label_map_path = os.path.join(
            self.model_path if os.path.isdir(self.model_path) else os.path.dirname(self.model_path),
            "label_map.json",
        )
        if os.path.exists(label_map_path):
            with open(label_map_path, "r", encoding="utf-8") as file:
                self.label_map = json.load(file)
                self.id_to_label = {int(v): k for k, v in self.label_map.items()}

        try:
            self.device = _resolve_device()
            self.model, self.tokenizer, self.checkpoint_path = load_intent_classifier(
                model_path=self.model_path,
                bert_base_path=self.bert_base_path,
                device=self.device,
                num_labels=2,
            )
            logger.info(f"加载意图识别模型成功: {self.checkpoint_path}")
        except Exception as exc:
            logger.warning(f"加载意图识别模型失败，将使用 LLM 兜底: {exc}")
            self.model = None
            self.tokenizer = None

    def _normalize_label(self, label):
        text = str(label or "").strip()
        if text in VALID_INTENT_LABELS:
            return text
        if "专业咨询" in text:
            return "专业咨询"
        if "通用知识" in text:
            return "通用知识"
        return None

    def _fallback_category(self):
        return self._normalize_label(self.conf.INTENT_FALLBACK_CATEGORY) or "专业咨询"

    def _get_llm_client(self):
        if self.llm_client is None:
            from openai import OpenAI

            self.llm_client = OpenAI(
                api_key=self.conf.DASHSCOPE_API_KEY,
                base_url=self.conf.DASHSCOPE_BASE_URL,
            )
        return self.llm_client

    def _predict_with_llm(self, query):
        if not self.conf.ENABLE_LLM_INTENT_FALLBACK:
            return self._fallback_category()

        try:
            prompt = RAGPrompts.intent_prompt(self.conf).format(query=query)
            completion = self._get_llm_client().chat.completions.create(
                model=self.conf.LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.conf.INTENT_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.conf.INTENT_LLM_TEMPERATURE,
            )
            content = completion.choices[0].message.content if completion.choices else ""
            category = self._normalize_label(content)
            if category:
                logger.info(f"LLM 意图识别结果：{category}")
                return category
            logger.warning(f"LLM 意图识别返回无效标签: {content!r}")
        except Exception as exc:
            logger.warning(f"LLM 意图识别失败，使用兜底分类: {exc}")

        return self._fallback_category()

    def _predict_with_bert(self, query):
        if self.model is None or self.tokenizer is None:
            return None

        import torch

        encoding = self.tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            prediction = int(torch.argmax(logits, dim=-1).item())

        return self.id_to_label.get(prediction)

    def predict_category(self, query):
        try:
            category = self._predict_with_bert(query)
            if category:
                return category
        except Exception as exc:
            logger.warning(f"BERT 意图识别失败，将使用 LLM 兜底: {exc}")

        return self._predict_with_llm(query)


if __name__ == "__main__":
    classifier = QueryClassifier()
    test_queries = [
        "天恒科技年假有多少天？",
        "加班费如何计算？",
        "5*9等于多少？",
        "公司 VPN 怎么配置？",
        "怎么和陌生人聊天不尴尬？",
    ]
    for query in test_queries:
        category = classifier.predict_category(query)
        print(f"查询: {query} -> 分类: {category}")
