import math
import re

from base import Config


def _is_time_sensitive(text, conf):
    return re.search(conf.CACHE_TIME_SENSITIVE_PATTERN, text) is not None


def normalize_rerank_score(raw_score):
    """将 CrossEncoder 原始分数映射到 0~1。"""
    try:
        value = float(raw_score)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def clamp_confidence(value, default=0.0):
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, score))


def compute_cache_write_score(rerank_score, llm_confidence, conf=None):
    conf = conf or Config()
    rerank_weight = conf.FAQ_RERANK_WEIGHT
    llm_weight = conf.FAQ_LLM_WEIGHT
    total = rerank_weight + llm_weight
    if total <= 0:
        return (float(rerank_score) + float(llm_confidence)) / 2.0
    return (rerank_weight * float(rerank_score) + llm_weight * float(llm_confidence)) / total


def should_write_cache(
    query,
    answer,
    source,
    has_context,
    rerank_score,
    llm_confidence,
    citations=None,
    grounded=False,
    conf=None,
):
    """判断是否将问答对写入 Redis 缓存。"""
    conf = conf or Config()

    if source != "rag" or not has_context:
        return False, 0.0

    if not grounded or not citations:
        return False, 0.0

    text = f"{query}\n{answer}"
    if _is_time_sensitive(text, conf):
        return False, 0.0

    if not answer or len(answer.strip()) < conf.FAQ_MIN_ANSWER_LENGTH:
        return False, 0.0

    write_score = compute_cache_write_score(rerank_score, llm_confidence, conf=conf)
    threshold = conf.FAQ_CACHE_WRITE_THRESHOLD
    return write_score >= threshold, write_score
