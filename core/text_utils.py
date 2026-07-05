from base import logger


def preprocess_text(text):
    """Tokenize Chinese text for BM25 retrieval."""
    try:
        import jieba

        return jieba.lcut(str(text).lower())
    except ImportError:
        logger.warning("jieba is not installed; falling back to whitespace tokenization")
        return str(text).lower().split()
    except (AttributeError, TypeError) as exc:
        logger.error(f"文本预处理失败: {exc}")
        return []


def normalize_question(value):
    """Normalize FAQ question values loaded from Redis or raw strings."""
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value)
