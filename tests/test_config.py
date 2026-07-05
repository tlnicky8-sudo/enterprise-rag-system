from pathlib import Path

from base.config import Config


def _write_config(path: Path) -> None:
    path.write_text(
        """
[redis]
host = redis.local
port = 6380
password = file-redis
db = 2

[milvus]
host = milvus.local
port = 19531
database_name = qa_db
collection_name = qa_collection

[llm]
model = qwen-plus

[retrieval]
parent_chunk_size = 1000
child_chunk_size = 250
chunk_overlap = 40
retrieval_k = 4
candidate_m = 2

[logger]
log_file = logs/test.log

[app]
valid_sources = ["labor_law", "contract"]
customer_service_phone = 10086

[models]
bge_m3_path = models/bge-m3
bge_reranker_path = models/bge-reranker-large
bert_classifier_path = models/bert_outputs
""".strip(),
        encoding="utf-8",
    )


def test_config_loads_file_values(tmp_path, monkeypatch):
    config_file = tmp_path / "config.ini"
    _write_config(config_file)
    for name in ("REDIS_HOST", "REDIS_PORT", "MILVUS_COLLECTION_NAME"):
        monkeypatch.delenv(name, raising=False)

    config = Config(config_file=config_file)

    assert config.REDIS_HOST == "redis.local"
    assert config.REDIS_PORT == 6380
    assert config.MILVUS_COLLECTION_NAME == "qa_collection"
    assert config.VALID_SOURCES == ["labor_law", "contract"]
    assert config.CHILD_CHUNK_SIZE == 250
    assert config.DOMAIN_NAME == "labor_law"
    assert config.FAQ_BM25_THRESHOLD == 0.85


def test_environment_variables_override_sensitive_values(tmp_path, monkeypatch):
    config_file = tmp_path / "config.ini"
    _write_config(config_file)
    monkeypatch.setenv("REDIS_PASSWORD", "env-redis-password")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-api-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.test/v1")

    config = Config(config_file=config_file)

    assert config.REDIS_PASSWORD == "env-redis-password"
    assert config.DASHSCOPE_API_KEY == "env-api-key"
    assert config.DASHSCOPE_BASE_URL == "https://example.test/v1"


def test_legacy_dashscope_environment_names_are_supported(tmp_path, monkeypatch):
    config_file = tmp_path / "config.ini"
    _write_config(config_file)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    monkeypatch.setenv("API-KEY", "legacy-key")
    monkeypatch.setenv("BASE-URL", "https://legacy.test/v1")

    config = Config(config_file=config_file)

    assert config.DASHSCOPE_API_KEY == "legacy-key"
    assert config.DASHSCOPE_BASE_URL == "https://legacy.test/v1"
