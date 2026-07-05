from pathlib import Path

from base.config import Config
from core.prompts import RAGPrompts


def _write_runtime(path: Path) -> None:
    path.write_text(
        """
features:
  enable_faq: false
  enable_grounding_gate: true
  enable_strategy_selector: false
  enable_llm_intent_fallback: true
  enable_cache_write: true
  enable_stream: false
  skip_faq_when_source_filter: false

retrieval:
  retrieval_k: 7
  candidate_m: 3
  hybrid_dense_weight: 1.1
  hybrid_sparse_weight: 0.6
  milvus_nprobe: 12
  subquery_max_count: 2
  short_circuit_rerank_score: 0.9

faq:
  semantic_threshold: 0.91
  bm25_threshold: 0.84
  bm25_min_score: 0.2
  cache_ttl: 3600
  cache_write_threshold: 0.75
  rerank_weight: 0.5
  llm_weight: 0.5
  min_answer_length: 30

grounding:
  require_context_for_legal_qa: false
  min_rerank_score: 0.4

generation:
  max_context_chars: 5000
  max_prompt_length: 3000

strategy:
  default_strategy: 直接检索
  llm_temperature: 0.2
  system_message: test-system

intent:
  fallback_category: 专业咨询
  llm_temperature: 0.0
  system_message: test-intent-system

cache:
  time_sensitive_pattern: "今天|实时"

prompts:
  dir: config/prompts
""".strip(),
        encoding="utf-8",
    )


def test_runtime_yaml_overrides_defaults(tmp_path, monkeypatch):
    config_file = tmp_path / "config.ini"
    runtime_file = tmp_path / "runtime.yaml"
    config_file.write_text(
        """
[redis]
host = localhost
port = 6379
password =
db = 0

[milvus]
host = localhost
port = 19530
database_name = labor_law
collection_name = labor_law_rag

[llm]
model = qwen-plus

[retrieval]
parent_chunk_size = 1200
child_chunk_size = 300
chunk_overlap = 50
retrieval_k = 5
candidate_m = 2

[logger]
log_file = logs/app.log

[app]
valid_sources = ["labor_law"]
customer_service_phone = 12345678

[models]
bert_classifier_path = models/bert_outputs
""".strip(),
        encoding="utf-8",
    )
    _write_runtime(runtime_file)

    config = Config(config_file=config_file, runtime_file=runtime_file)

    assert config.ENABLE_FAQ is False
    assert config.ENABLE_STREAM is False
    assert config.RETRIEVAL_K == 7
    assert config.CANDIDATE_M == 3
    assert config.HYBRID_DENSE_WEIGHT == 1.1
    assert config.FAQ_BM25_MIN_SCORE == 0.2
    assert config.MIN_RERANK_SCORE == 0.4
    assert config.MAX_CONTEXT_CHARS == 5000
    assert config.STRATEGY_SYSTEM_MESSAGE == "test-system"
    assert config.ENABLE_LLM_INTENT_FALLBACK is True
    assert config.INTENT_FALLBACK_CATEGORY == "专业咨询"
    assert config.INTENT_SYSTEM_MESSAGE == "test-intent-system"
    assert config.CACHE_TIME_SENSITIVE_PATTERN == "今天|实时"


def test_prompt_registry_loads_templates():
    config = Config()
    rag_prompt = RAGPrompts.rag_prompt(config)
    assert "劳动法助手" in rag_prompt.template
    assert "phone" in rag_prompt.input_variables

    strategy_prompt = RAGPrompts.strategy_prompt(config)
    assert "直接检索" in strategy_prompt.template

    intent_prompt = RAGPrompts.intent_prompt(config)
    assert "通用知识" in intent_prompt.template
    assert "query" in intent_prompt.input_variables
