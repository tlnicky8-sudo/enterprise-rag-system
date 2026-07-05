import hashlib
import json
import struct

import redis

from base import Config, logger


class FAQRedisCache:
    """Redis cache for FAQ Q&A pairs and semantic vector search."""

    QA_PAIRS_KEY = "faq:qa_pairs"
    VERSION_KEY = "faq_version"
    INDEX_NAME = "semantic_cache_idx"
    SEMANTIC_PREFIX = "semantic_cache:"

    def __init__(self, conf=None):
        self.conf = conf or Config()
        redis_kwargs = {
            "host": self.conf.REDIS_HOST,
            "port": self.conf.REDIS_PORT,
            "password": self.conf.REDIS_PASSWORD or None,
            "db": self.conf.REDIS_DB,
        }
        self.text_client = redis.StrictRedis(**redis_kwargs, decode_responses=True)
        self.vector_client = redis.StrictRedis(**redis_kwargs, decode_responses=False)
        self.text_client.ping()
        logger.info("Redis FAQ cache connected")

    @staticmethod
    def _vector_bytes_to_list(vector_bytes):
        count = len(vector_bytes) // 4
        return list(struct.unpack(f"{count}f", vector_bytes))

    @staticmethod
    def _vector_list_to_bytes(vector_list):
        return struct.pack(f"{len(vector_list)}f", *vector_list)

    @staticmethod
    def _to_bool(value):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def get_json(self, key):
        data = self.text_client.get(key)
        return json.loads(data) if data else None

    def set_json(self, key, value):
        self.text_client.set(key, json.dumps(value, ensure_ascii=False))

    def get_faq_version(self):
        return self.text_client.get(self.VERSION_KEY)

    def set_faq_version(self, version):
        self.text_client.set(self.VERSION_KEY, str(version))

    def get_qa_pairs(self):
        return [
            (item["question"], item["answer"])
            for item in self.get_qa_records()
            if item.get("question") and item.get("answer")
        ]

    def get_qa_records(self):
        payload = self.get_json(self.QA_PAIRS_KEY)
        if not payload:
            return []
        records = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            answer = str(item.get("answer") or "").strip()
            if question and answer:
                records.append(
                    {
                        "question": question,
                        "answer": answer,
                        "citations": item.get("citations") or [],
                        "grounded": bool(item.get("grounded", False)),
                    }
                )
        return records

    def set_qa_pairs(self, pairs):
        payload = []
        for item in pairs:
            if isinstance(item, dict):
                payload.append(
                    {
                        "question": item.get("question"),
                        "answer": item.get("answer"),
                        "citations": item.get("citations") or [],
                        "grounded": bool(item.get("grounded", False)),
                    }
                )
            else:
                question, answer = item
                payload.append(
                    {
                        "question": question,
                        "answer": answer,
                        "citations": [],
                        "grounded": False,
                    }
                )
        self.set_json(self.QA_PAIRS_KEY, payload)

    def add_qa_pair(self, question, answer, citations=None, grounded=False):
        records = self.get_qa_records()
        normalized = question.strip()
        payload = {
            "question": normalized,
            "answer": answer,
            "citations": citations or [],
            "grounded": bool(grounded),
        }
        for idx, existing in enumerate(records):
            if existing["question"] == normalized:
                records[idx] = payload
                self.set_qa_pairs(records)
                return False
        records.append(payload)
        self.set_qa_pairs(records)
        return True

    def clear_faq_data(self):
        for key in self.vector_client.scan_iter(match=f"{self.SEMANTIC_PREFIX}*"):
            self.vector_client.delete(key)
        self.text_client.delete(self.QA_PAIRS_KEY)
        try:
            self.vector_client.ft(self.INDEX_NAME).dropindex(delete_documents=True)
        except Exception:
            pass

    def ensure_vector_index(self, vector_dim):
        try:
            self.vector_client.ft(self.INDEX_NAME).info()
            return
        except Exception:
            pass

        from redis.commands.search.field import TextField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType

        schema = (
            TextField("$.question", as_name="question"),
            TextField("$.answer", as_name="answer"),
            VectorField(
                "$.vector",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": vector_dim,
                    "DISTANCE_METRIC": "COSINE",
                },
                as_name="vector",
            ),
        )
        definition = IndexDefinition(prefix=[self.SEMANTIC_PREFIX], index_type=IndexType.JSON)
        self.vector_client.ft(self.INDEX_NAME).create_index(fields=schema, definition=definition)
        logger.info("Redis semantic FAQ index created")

    def add_semantic_cache(self, question, answer, vector_bytes, ttl=None, citations=None, grounded=False):
        question_hash = hashlib.md5(question.encode("utf-8")).hexdigest()
        cache_key = f"{self.SEMANTIC_PREFIX}{question_hash}"
        vector_list = self._vector_bytes_to_list(vector_bytes)
        payload = {
            "question": question,
            "answer": answer,
            "vector": vector_list,
            "citations": citations or [],
            "grounded": bool(grounded),
        }
        self.vector_client.json().set(cache_key, "$", payload)
        self.vector_client.expire(cache_key, ttl if ttl is not None else self.conf.FAQ_CACHE_TTL)

    def semantic_search(self, query_vector_bytes, threshold):
        from redis.commands.search.query import Query

        base_query = "*=>[KNN 1 @vector $query_vec AS score]"
        query = (
            Query(base_query)
            .return_fields("question", "answer", "citations", "grounded", "score")
            .sort_by("score")
            .dialect(2)
        )
        results = self.vector_client.ft(self.INDEX_NAME).search(
            query,
            query_params={"query_vec": query_vector_bytes},
        )
        if results.total <= 0:
            return None, 0.0

        best_match = results.docs[0]
        similarity = 1 - float(best_match.score)
        if similarity >= threshold:
            answer = best_match.answer
            if isinstance(answer, bytes):
                answer = answer.decode("utf-8")
            citations = getattr(best_match, "citations", "[]") or "[]"
            if isinstance(citations, bytes):
                citations = citations.decode("utf-8")
            try:
                citations = json.loads(citations)
            except (TypeError, ValueError):
                citations = []
            grounded = self._to_bool(getattr(best_match, "grounded", False))
            return answer, similarity, citations, grounded
        return None, similarity, [], False
