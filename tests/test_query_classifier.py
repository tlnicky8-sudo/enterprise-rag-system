from core.query_classifier import QueryClassifier


class _Conf:
    BERT_CLASSIFIER_PATH = "missing/intent-model"
    BERT_BASE_PATH = "missing/bert-base"
    ENABLE_LLM_INTENT_FALLBACK = True
    INTENT_FALLBACK_CATEGORY = "专业咨询"
    INTENT_SYSTEM_MESSAGE = "intent"
    INTENT_LLM_TEMPERATURE = 0.0
    LLM_MODEL = "test-model"
    DASHSCOPE_API_KEY = ""
    DASHSCOPE_BASE_URL = "http://example.test/v1"


def test_query_classifier_falls_back_to_llm_when_model_missing(monkeypatch):
    def _raise_load_error(*args, **kwargs):
        raise FileNotFoundError("missing checkpoint")

    monkeypatch.setattr("core.query_classifier.load_intent_classifier", _raise_load_error)
    monkeypatch.setattr(QueryClassifier, "_predict_with_llm", lambda self, query: "通用知识")

    classifier = QueryClassifier(conf=_Conf())

    assert classifier.model is None
    assert classifier.predict_category("今天天气怎么样？") == "通用知识"


def test_query_classifier_uses_safe_default_when_llm_disabled(monkeypatch):
    class Conf(_Conf):
        ENABLE_LLM_INTENT_FALLBACK = False

    def _raise_load_error(*args, **kwargs):
        raise FileNotFoundError("missing checkpoint")

    monkeypatch.setattr("core.query_classifier.load_intent_classifier", _raise_load_error)

    classifier = QueryClassifier(conf=Conf())

    assert classifier.predict_category("试用期最长多久？") == "专业咨询"
