"""Preheat FAQ pairs from Redis (or JSONL fallback) into semantic cache."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import Config, logger
from core.faq.loader import load_faq_pairs, to_question_answer_pairs
from core.faq_service import FAQService


def resolve_data_path(conf, source_path=None):
    if source_path:
        path = Path(source_path)
        if not path.is_absolute():
            path = conf.PROJECT_ROOT / path
        return path
    return conf.PROJECT_ROOT / "data" / "faq_data" / "faq_pairs.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Preheat FAQ semantic cache")
    parser.add_argument("--version", type=str, default=None, help="Optional FAQ version tag")
    parser.add_argument(
        "--source",
        default=None,
        help="Fallback FAQ data file when Redis has no pairs",
    )
    args = parser.parse_args()

    conf = Config()
    service = FAQService(conf)
    if not service.enabled:
        logger.error("FAQ service unavailable; check Redis/model configuration")
        return 1

    if args.version:
        service.reload(version=args.version)

    pairs = service.faq_search.redis_cache.get_qa_pairs()
    if not pairs:
        data_path = resolve_data_path(conf, args.source)
        logger.info(f"No FAQ pairs in Redis, loading from {data_path}")
        pairs, _ = load_faq_pairs(data_path, default_subject=conf.DOMAIN_NAME)

    if not pairs:
        logger.error("No FAQ pairs available to preheat")
        return 1

    count = service.preheat_cache(pairs)
    print(f"Preheated {count} FAQ pairs for domain '{conf.DOMAIN_NAME}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
