from core.cache_policy import should_write_cache
from core.qa_pipeline import QAPipeline
from core.rag_system import RAGSystem


class _Conf:
    FAQ_RERANK_WEIGHT = 0.6
    FAQ_LLM_WEIGHT = 0.4
    FAQ_CACHE_WRITE_THRESHOLD = 0.8
    FAQ_MIN_ANSWER_LENGTH = 20
    CACHE_TIME_SENSITIVE_PATTERN = "今天|本周|本月|今年|现在|实时|当前|几点|天气"


class _FAQService:
    def search(self, query):
        return "cached answer", False, [{"id": 1, "title": "年假制度", "excerpt": "证据"}]


class _SessionStore:
    def __init__(self):
        self.saved = []

    def get_history_text(self, session_id):
        return ""

    def get_pairs(self, session_id):
        return []

    def clear_session(self, session_id):
        return None

    def save_exchange(self, session_id, question, answer, source="rag"):
        self.saved.append((session_id, question, answer, source))


def test_cache_policy_requires_grounded_citations():
    answer = "这是一个足够长的企业知识库回答内容，用于验证缓存写入策略。"
    ok, _ = should_write_cache(
        query="试用期多久",
        answer=answer,
        source="rag",
        has_context=True,
        rerank_score=0.9,
        llm_confidence=0.9,
        citations=[],
        grounded=True,
        conf=_Conf(),
    )
    assert ok is False

    ok, _ = should_write_cache(
        query="试用期多久",
        answer=answer,
        source="rag",
        has_context=True,
        rerank_score=0.9,
        llm_confidence=0.9,
        citations=[{"id": 1}],
        grounded=True,
        conf=_Conf(),
    )
    assert ok is True


def test_grounding_decision_rejects_empty_or_low_score_context(monkeypatch):
    system = RAGSystem.__new__(RAGSystem)
    monkeypatch.setattr("core.rag_system.conf.REQUIRE_CONTEXT_FOR_KB_QA", True)
    monkeypatch.setattr("core.rag_system.conf.MIN_RERANK_SCORE", 0.5)

    no_context = system._grounding_decision("rag", False, 0.0, [])
    assert no_context.should_answer is False
    assert "未检索到" in no_context.refusal_reason

    low_score = system._grounding_decision("rag", True, 0.2, [{"id": 1}])
    assert low_score.should_answer is False
    assert "检索相关性不足" in low_score.refusal_reason

    enough = system._grounding_decision("rag", True, 0.8, [{"id": 1}])
    assert enough.should_answer is True
    assert enough.grounded is True


def test_pipeline_propagates_faq_citations():
    session_store = _SessionStore()
    pipeline = QAPipeline(
        faq_service=_FAQService(),
        rag_system=None,
        session_store=session_store,
    )

    result = pipeline.answer("试用期多久", "s1")

    assert result.source == "faq"
    assert result.grounded is True
    assert result.citations[0]["title"] == "年假制度"
    assert session_store.saved[0][3] == "faq"
