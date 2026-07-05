import hashlib

import numpy as np
from rank_bm25 import BM25Okapi

from base import Config, logger
from core.faq.redis_cache import FAQRedisCache
from core.faq.loader import to_question_answer_pairs
from core.text_utils import normalize_question, preprocess_text


class FAQSearch:
    """Redis semantic cache + BM25 over Redis FAQ pairs."""

    def __init__(self, redis_cache=None, conf=None):
        self.conf = conf or Config()
        self.redis_cache = redis_cache or FAQRedisCache(self.conf)
        self.embed_model = None
        self.vector_dim = None
        self.bm25 = None
        self.original_questions = []
        self.question_to_record = {}
        self._load_bm25_data()

    def _load_embed_model(self):
        if self.embed_model is not None:
            return
        from sentence_transformers import SentenceTransformer

        model_path = self.conf.BGE_M3_PATH
        logger.info(f"Loading FAQ embedding model from {model_path}")
        self.embed_model = SentenceTransformer(model_path)
        self.vector_dim = self.embed_model.get_sentence_embedding_dimension()
        self.redis_cache.ensure_vector_index(self.vector_dim)

    def _load_bm25_data(self):
        pairs = self.redis_cache.get_qa_pairs()
        if not pairs:
            logger.warning("FAQ layer: no Q&A pairs loaded from Redis")
            self.original_questions = []
            self.question_to_record = {}
            self.bm25 = None
            return

        records = self.redis_cache.get_qa_records()
        self.original_questions = [normalize_question(item["question"]) for item in records]
        self.question_to_record = {
            normalize_question(item["question"]): item for item in records
        }
        tokenized_questions = [preprocess_text(question) for question in self.original_questions]
        self.bm25 = BM25Okapi(tokenized_questions)
        logger.info(f"FAQ BM25 index initialized with {len(records)} pairs")

    def reload_faq_data(self, version=None):
        if version is not None:
            self.redis_cache.set_faq_version(version)
        self._load_bm25_data()

    @staticmethod
    def _softmax(scores):
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()

    def _encode_query(self, query):
        self._load_embed_model()
        vector = self.embed_model.encode(query, normalize_embeddings=True)
        return vector.astype(np.float32).tobytes()

    def add_to_semantic_cache(self, question, answer, ttl=None, citations=None, grounded=False):
        vector_bytes = self._encode_query(question)
        self.redis_cache.add_semantic_cache(
            question,
            answer,
            vector_bytes,
            ttl=ttl,
            citations=citations,
            grounded=grounded,
        )
        added = self.redis_cache.add_qa_pair(
            question,
            answer,
            citations=citations,
            grounded=grounded,
        )
        self._load_bm25_data()
        return added

    def preheat_cache(self, qa_pairs, ttl_map=None):
        qa_pairs = to_question_answer_pairs(qa_pairs)
        merged = {
            item["question"]: item
            for item in self.redis_cache.get_qa_records()
            if item.get("question")
        }
        for question, answer in qa_pairs:
            existing = merged.get(question, {})
            merged[question] = {
                "question": question,
                "answer": answer,
                "citations": existing.get("citations") or [],
                "grounded": bool(existing.get("grounded", False)),
            }
        self.redis_cache.set_qa_pairs(list(merged.values()))

        self._load_embed_model()
        count = 0
        for question, answer in qa_pairs:
            ttl = ttl_map.get(question) if ttl_map else None
            vector_bytes = self._encode_query(question)
            self.redis_cache.add_semantic_cache(question, answer, vector_bytes, ttl=ttl)
            count += 1

        self._load_bm25_data()
        logger.info(f"FAQ semantic cache preheated with {count} pairs")
        return count

    def search(self, query, semantic_threshold=None, bm25_threshold=None):
        if not query or not isinstance(query, str):
            logger.error("FAQ search received invalid query")
            return None, True, []

        semantic_threshold = (
            semantic_threshold if semantic_threshold is not None else self.conf.FAQ_SEMANTIC_THRESHOLD
        )
        bm25_threshold = (
            bm25_threshold if bm25_threshold is not None else self.conf.FAQ_BM25_THRESHOLD
        )
        bm25_min_score = self.conf.FAQ_BM25_MIN_SCORE

        try:
            self._load_embed_model()
            query_vector = self._encode_query(query)
            answer, similarity, citations, grounded = self.redis_cache.semantic_search(
                query_vector,
                semantic_threshold,
            )
            if answer:
                logger.info(f"FAQ semantic cache hit, similarity={similarity:.3f}")
                return answer, False, citations
        except Exception as exc:
            logger.warning(f"FAQ semantic search unavailable, fallback to BM25: {exc}")

        if not self.bm25 or not self.original_questions:
            return None, True, []

        try:
            query_tokens = preprocess_text(query)
            scores = self.bm25.get_scores(query_tokens)
            if len(scores) == 0:
                return None, True, []

            softmax_scores = self._softmax(scores)
            best_idx = int(softmax_scores.argmax())
            best_softmax = float(softmax_scores[best_idx])
            best_raw_score = float(scores[best_idx])

            if best_raw_score >= bm25_min_score and best_softmax >= bm25_threshold:
                original_question = self.original_questions[best_idx]
                record = self.question_to_record.get(original_question)
                if record and record.get("answer"):
                    logger.info(
                        "FAQ BM25 hit, raw=%.3f, softmax=%.3f",
                        best_raw_score,
                        best_softmax,
                    )
                    return record["answer"], False, record.get("citations") or []

            logger.info(
                "FAQ miss, best raw=%.3f, softmax=%.3f",
                best_raw_score,
                best_softmax,
            )
            return None, True, []
        except Exception as exc:
            logger.error(f"FAQ BM25 search failed: {exc}")
            return None, True, []
