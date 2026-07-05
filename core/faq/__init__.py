__all__ = ["FAQSearch", "FAQRedisCache"]


def __getattr__(name):
    if name == "FAQSearch":
        from core.faq.search import FAQSearch

        return FAQSearch
    if name == "FAQRedisCache":
        from core.faq.redis_cache import FAQRedisCache

        return FAQRedisCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
