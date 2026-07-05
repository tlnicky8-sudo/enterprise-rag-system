from base import logger
from core.cache_policy import should_write_cache
from core.faq.redis_cache import FAQRedisCache
from core.faq.search import FAQSearch


class FAQService:
    """Redis-only FAQ layer with graceful degradation."""

    def __init__(self, conf=None):
        self.enabled = False
        self.faq_search = None
        self.conf = conf

        try:
            from base import Config

            self.conf = conf or Config()
            if not self.conf.ENABLE_FAQ:
                logger.info("FAQ service disabled by runtime config (enable_faq=false)")
                return
            redis_cache = FAQRedisCache(self.conf)
            self.faq_search = FAQSearch(redis_cache=redis_cache, conf=self.conf)
            self.enabled = True
            logger.info("FAQ service initialized (Redis only)")
        except Exception as exc:
            logger.warning(f"FAQ service disabled: {exc}")

    def search(self, query):
        if not self.enabled or self.faq_search is None:
            return None, True, []
        return self.faq_search.search(query)

    def maybe_cache(
        self,
        query,
        answer,
        source,
        has_context,
        rerank_score,
        llm_confidence,
        citations=None,
        grounded=False,
    ):
        if not self.enabled or self.faq_search is None:
            return False, 0.0

        ok, write_score = should_write_cache(
            query=query,
            answer=answer,
            source=source,
            has_context=has_context,
            rerank_score=rerank_score,
            llm_confidence=llm_confidence,
            citations=citations,
            grounded=grounded,
            conf=self.conf,
        )
        if not ok:
            logger.info(f"FAQ cache skipped, write_score={write_score:.3f}")
            return False, write_score

        try:
            self.faq_search.add_to_semantic_cache(
                query,
                answer,
                citations=citations,
                grounded=grounded,
            )
            logger.info(f"FAQ cache written, write_score={write_score:.3f}")
            return True, write_score
        except Exception as exc:
            logger.warning(f"FAQ cache write failed: {exc}")
            return False, write_score

    def preheat_cache(self, qa_pairs=None, ttl_map=None):
        if not self.enabled or self.faq_search is None:
            logger.warning("FAQ preheat skipped: service unavailable")
            return 0
        if qa_pairs is None:
            qa_pairs = self.faq_search.redis_cache.get_qa_pairs()
        if not qa_pairs:
            logger.warning("FAQ preheat skipped: no Q&A pairs")
            return 0
        if ttl_map is None:
            ttl_map = {}
        return self.faq_search.preheat_cache(qa_pairs, ttl_map=ttl_map)

    def reload(self, version=None):
        if not self.enabled or self.faq_search is None:
            return
        self.faq_search.reload_faq_data(version=version)

    def clear(self):
        if not self.enabled or self.faq_search is None:
            return 0
        redis_cache = self.faq_search.redis_cache
        count = len(redis_cache.get_qa_pairs())
        redis_cache.clear_faq_data()
        self.faq_search.reload_faq_data()
        return count

    def close(self):
        return
