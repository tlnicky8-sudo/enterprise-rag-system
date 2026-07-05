from core.faq.redis_cache import FAQRedisCache


class _FakeTextClient:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value


def _cache():
    cache = FAQRedisCache.__new__(FAQRedisCache)
    cache.text_client = _FakeTextClient()
    return cache


def test_qa_records_preserve_citations_and_grounded_flag():
    cache = _cache()
    citations = [{"id": 1, "title": "第四十七条", "excerpt": "经济补偿"}]

    added = cache.add_qa_pair(
        "解除赔偿怎么算",
        "根据第四十七条计算。[1]",
        citations=citations,
        grounded=True,
    )

    assert added is True
    records = cache.get_qa_records()
    assert records[0]["citations"] == citations
    assert records[0]["grounded"] is True
    assert cache.get_qa_pairs() == [("解除赔偿怎么算", "根据第四十七条计算。[1]")]


def test_qa_record_update_replaces_provenance():
    cache = _cache()
    cache.add_qa_pair("Q", "A1", citations=[{"id": 1}], grounded=True)
    added = cache.add_qa_pair("Q", "A2", citations=[{"id": 2}], grounded=True)

    assert added is False
    assert cache.get_qa_records()[0]["answer"] == "A2"
    assert cache.get_qa_records()[0]["citations"] == [{"id": 2}]
