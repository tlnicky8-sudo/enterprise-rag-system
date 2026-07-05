from dataclasses import dataclass, field

from base import Config, logger
from core.faq_service import FAQService
from core.llm_utils import collect_llm_output
from core.rag_models import GenerationResult
from core.rag_system import RAGSystem
from core.session_store import SessionStore

conf = Config()


@dataclass
class QAResult:
    answer: str
    source: str
    stream: bool = False
    token_iterator: object = None
    generation: GenerationResult | None = None
    citations: list = field(default_factory=list)
    grounded: bool = False
    refusal_reason: str = ""


class QAPipeline:
    """FAQ fast path first, then RAG fallback, with session persistence."""

    def __init__(self, faq_service=None, rag_system=None, session_store=None):
        self.faq_service = faq_service if faq_service is not None else FAQService(conf)
        self.rag_system = rag_system
        self.session_store = session_store if session_store is not None else SessionStore()

    def get_history(self, session_id):
        return self.session_store.get_pairs(session_id)

    def clear_session(self, session_id):
        self.session_store.clear_session(session_id)

    def _maybe_cache_answer(self, query, generation: GenerationResult | None):
        if not conf.ENABLE_CACHE_WRITE or generation is None:
            return
        self.faq_service.maybe_cache(
            query=query,
            answer=generation.answer,
            source=generation.source,
            has_context=generation.has_context,
            rerank_score=generation.rerank_score,
            llm_confidence=generation.llm_confidence,
            citations=generation.citations,
            grounded=generation.grounded,
        )

    def answer(self, query, session_id, source_filter=None, stream=False):
        history_text = self.session_store.get_history_text(session_id)

        faq_answer, need_rag, faq_citations = (None, True, [])
        skip_faq = source_filter is not None and conf.SKIP_FAQ_WHEN_SOURCE_FILTER
        if conf.ENABLE_FAQ and not skip_faq:
            faq_answer, need_rag, faq_citations = self.faq_service.search(query)
        elif skip_faq:
            logger.info("Pipeline skipped FAQ layer because source_filter is set")
        elif not conf.ENABLE_FAQ:
            logger.info("Pipeline skipped FAQ layer because enable_faq is false")

        if faq_answer and not need_rag:
            self.session_store.save_exchange(session_id, query, faq_answer, source="faq")
            logger.info("Pipeline answered from FAQ layer")
            return QAResult(
                answer=faq_answer,
                source="faq",
                stream=False,
                citations=faq_citations,
                grounded=bool(faq_citations),
            )

        if self.rag_system is None:
            fallback = "抱歉，当前无法连接到 RAG 服务，请稍后重试。"
            self.session_store.save_exchange(session_id, query, fallback, source="rag")
            return QAResult(answer=fallback, source="rag", stream=False)

        if stream and conf.ENABLE_STREAM:
            token_iterator, generation = self.rag_system.generate_answer_stream(
                query,
                source_filter=source_filter,
                history=history_text,
            )
            return QAResult(
                answer="",
                source=generation.source,
                stream=True,
                token_iterator=token_iterator,
                generation=generation,
                citations=list(generation.citations),
                grounded=generation.grounded,
                refusal_reason=generation.refusal_reason,
            )

        generation = self.rag_system.generate_answer(
            query,
            source_filter=source_filter,
            history=history_text,
        )
        self._maybe_cache_answer(query, generation)
        self.session_store.save_exchange(
            session_id,
            query,
            generation.answer,
            source=generation.source,
        )
        return QAResult(
            answer=generation.answer,
            source=generation.source,
            generation=generation,
            citations=list(generation.citations),
            grounded=generation.grounded,
            refusal_reason=generation.refusal_reason,
        )

    def save_streamed_answer(self, session_id, query, answer, source="rag", generation=None):
        if generation is not None:
            generation = GenerationResult(
                answer=answer,
                source=generation.source,
                llm_confidence=generation.llm_confidence,
                rerank_score=generation.rerank_score,
                has_context=generation.has_context,
                citations=list(generation.citations),
                grounded=generation.grounded,
                refusal_reason=generation.refusal_reason,
            )
            self._maybe_cache_answer(query, generation)
            source = generation.source
        self.session_store.save_exchange(session_id, query, answer, source=source)

    @staticmethod
    def collect_stream(result: QAResult) -> str:
        if not result.stream or result.token_iterator is None:
            return result.answer
        return collect_llm_output(result.token_iterator)
