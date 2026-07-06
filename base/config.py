import ast
import configparser
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is optional for smoke tests
    load_dotenv = None

from base.prompt_registry import PromptRegistry
from base.runtime_loader import RuntimeLoader


class Config:
    """Application configuration loaded from config files and environment variables."""

    def __init__(self, config_file=None, runtime_file=None):
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent
        explicit_config_file = config_file is not None

        if config_file is None:
            local_config = self.PROJECT_ROOT / "config.ini"
            example_config = self.PROJECT_ROOT / "config.example.ini"
            config_file = local_config if local_config.exists() else example_config

        if load_dotenv is not None and not explicit_config_file:
            load_dotenv(self.PROJECT_ROOT / ".env")

        self.CONFIG_FILE = Path(config_file)
        self.config = configparser.ConfigParser()
        self.config.read(self.CONFIG_FILE, encoding="utf-8")

        self.runtime = RuntimeLoader(self.PROJECT_ROOT, runtime_file=runtime_file)
        self.RUNTIME_FILE = self.runtime.runtime_file

        # Redis 配置
        self.REDIS_HOST = self._get("redis", "host", "localhost", "REDIS_HOST")
        self.REDIS_PORT = self._getint("redis", "port", 6379, "REDIS_PORT")
        self.REDIS_PASSWORD = self._get("redis", "password", "", "REDIS_PASSWORD")
        self.REDIS_DB = self._getint("redis", "db", 0, "REDIS_DB")

        # Milvus 配置
        self.MILVUS_HOST = self._get("milvus", "host", "localhost", "MILVUS_HOST")
        self.MILVUS_PORT = self._get("milvus", "port", "19530", "MILVUS_PORT")
        self.MILVUS_DATABASE_NAME = self._get(
            "milvus", "database_name", "enterprise_rag", "MILVUS_DATABASE_NAME"
        )
        self.MILVUS_COLLECTION_NAME = self._get(
            "milvus", "collection_name", "enterprise_rag", "MILVUS_COLLECTION_NAME"
        )

        # LLM 配置
        self.LLM_MODEL = self._get("llm", "model", "deepseek-v4-pro", "LLM_MODEL")
        self.DASHSCOPE_API_KEY = self._env("DASHSCOPE_API_KEY", "API-KEY", default="")
        self.DASHSCOPE_BASE_URL = self._env(
            "DASHSCOPE_BASE_URL",
            "BASE-URL",
            default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 入库分块参数（数据管道）
        self.PARENT_CHUNK_SIZE = self._getint("retrieval", "parent_chunk_size", 1200)
        self.CHILD_CHUNK_SIZE = self._getint("retrieval", "child_chunk_size", 300)
        self.CHUNK_OVERLAP = self._getint("retrieval", "chunk_overlap", 50)

        # 应用配置
        valid_sources = self._get("app", "valid_sources", '["enterprise"]', "VALID_SOURCES")
        self.VALID_SOURCES = self._parse_list(valid_sources)
        self.CUSTOMER_SERVICE_PHONE = self._get(
            "app", "customer_service_phone", "12345678", "CUSTOMER_SERVICE_PHONE"
        )

        # 日志文件路径 (相对路径则基于项目根目录解析)
        log_file = self._get("logger", "log_file", "logs/app.log", "LOG_FILE")
        self.LOG_FILE = log_file if os.path.isabs(log_file) else str(self.PROJECT_ROOT / log_file)

        # 模型路径
        self.BERT_CLASSIFIER_PATH = self._resolve_path(
            self._get("models", "bert_classifier_path", "models/bert_outputs", "BERT_CLASSIFIER_PATH")
        )
        self.BERT_BASE_PATH = self._resolve_path(
            self._get(
                "models",
                "bert_base_path",
                "models/bert-base-chinese",
                "BERT_BASE_PATH",
            )
        )

        # 领域配置
        self.DOMAIN_NAME = self._get("domain", "name", "enterprise", "DOMAIN_NAME")
        self.DOMAIN_DISPLAY_NAME = self._get(
            "domain", "display_name", "企业知识库", "DOMAIN_DISPLAY_NAME"
        )

        self._load_runtime_settings()
        self._init_prompt_registry()

    def _load_runtime_settings(self):
        runtime = self.runtime

        # Feature switches
        self.ENABLE_FAQ = runtime.get_bool("features", "enable_faq", True)
        self.ENABLE_GROUNDING_GATE = runtime.get_bool("features", "enable_grounding_gate", True)
        self.ENABLE_STRATEGY_SELECTOR = runtime.get_bool("features", "enable_strategy_selector", True)
        self.ENABLE_LLM_INTENT_FALLBACK = runtime.get_bool(
            "features", "enable_llm_intent_fallback", True
        )
        self.ENABLE_CACHE_WRITE = runtime.get_bool("features", "enable_cache_write", True)
        self.ENABLE_STREAM = runtime.get_bool("features", "enable_stream", True)
        self.SKIP_FAQ_WHEN_SOURCE_FILTER = runtime.get_bool(
            "features", "skip_faq_when_source_filter", True
        )

        # Retrieval
        self.RETRIEVAL_K = self._runtime_or_ini_int("retrieval", "retrieval_k", "retrieval", "retrieval_k", 5)
        self.CANDIDATE_M = self._runtime_or_ini_int("retrieval", "candidate_m", "retrieval", "candidate_m", 2)
        self.HYBRID_DENSE_WEIGHT = runtime.get_float("retrieval", "hybrid_dense_weight", 1.0)
        self.HYBRID_SPARSE_WEIGHT = runtime.get_float("retrieval", "hybrid_sparse_weight", 0.7)
        self.MILVUS_NPROBE = runtime.get_int("retrieval", "milvus_nprobe", 10)
        self.SUBQUERY_MAX_COUNT = runtime.get_int("retrieval", "subquery_max_count", 3)
        self.SHORT_CIRCUIT_RERANK_SCORE = runtime.get_float(
            "retrieval", "short_circuit_rerank_score", 0.85
        )

        # FAQ
        self.FAQ_SEMANTIC_THRESHOLD = self._runtime_or_ini_float(
            "faq", "semantic_threshold", "faq", "semantic_threshold", 0.92
        )
        self.FAQ_BM25_THRESHOLD = self._runtime_or_ini_float(
            "faq", "bm25_threshold", "faq", "bm25_threshold", 0.85
        )
        self.FAQ_BM25_MIN_SCORE = self._runtime_or_ini_float(
            "faq", "bm25_min_score", "faq", "bm25_min_score", 0.1
        )
        self.FAQ_CACHE_TTL = self._runtime_or_ini_int("faq", "cache_ttl", "faq", "cache_ttl", 86400)
        self.FAQ_CACHE_WRITE_THRESHOLD = self._runtime_or_ini_float(
            "faq", "cache_write_threshold", "faq", "cache_write_threshold", 0.8
        )
        self.FAQ_RERANK_WEIGHT = self._runtime_or_ini_float(
            "faq", "rerank_weight", "faq", "rerank_weight", 0.6
        )
        self.FAQ_LLM_WEIGHT = self._runtime_or_ini_float("faq", "llm_weight", "faq", "llm_weight", 0.4)
        self.FAQ_MIN_ANSWER_LENGTH = runtime.get_int("faq", "min_answer_length", 20)

        # Grounding
        # Grounding — 支持旧键 require_context_for_legal_qa 向后兼容
        kb_qa = self.runtime.get("grounding", "require_context_for_kb_qa", None)
        if kb_qa is None:
            kb_qa = self.runtime.get("grounding", "require_context_for_legal_qa", None)
        if kb_qa is not None:
            from base.runtime_loader import _coerce_bool

            self.REQUIRE_CONTEXT_FOR_KB_QA = _coerce_bool(kb_qa)
        elif self.config.has_option("grounding", "require_context_for_kb_qa"):
            self.REQUIRE_CONTEXT_FOR_KB_QA = self._getbool("grounding", "require_context_for_kb_qa", True)
        elif self.config.has_option("grounding", "require_context_for_legal_qa"):
            self.REQUIRE_CONTEXT_FOR_KB_QA = self._getbool("grounding", "require_context_for_legal_qa", True)
        else:
            self.REQUIRE_CONTEXT_FOR_KB_QA = True
        self.MIN_RERANK_SCORE = self._runtime_or_ini_float(
            "grounding", "min_rerank_score", "grounding", "min_rerank_score", 0.35
        )

        # Generation
        self.MAX_CONTEXT_CHARS = runtime.get_int("generation", "max_context_chars", 6000)
        self.MAX_PROMPT_LENGTH = runtime.get_int("generation", "max_prompt_length", 4096)
        self.GENERATION_SYSTEM_MESSAGE = runtime.get_str(
            "generation",
            "system_message",
            "你是一个专业的企业知识助手，请基于提供的公司内部制度文档上下文回答用户问题。",
        )

        # Strategy selector
        self.DEFAULT_STRATEGY = runtime.get_str("strategy", "default_strategy", "直接检索")
        self.STRATEGY_LLM_TEMPERATURE = runtime.get_float("strategy", "llm_temperature", 0.1)
        self.STRATEGY_SYSTEM_MESSAGE = runtime.get_str(
            "strategy", "system_message", "你是一个有用的助手。"
        )
        self.STRATEGY_ENABLE_HEURISTIC_FALLBACK = runtime.get_bool(
            "strategy", "enable_heuristic_fallback", True
        )
        self.STRATEGY_HEURISTIC_OVERRIDE_DIRECT = runtime.get_bool(
            "strategy", "heuristic_override_direct", True
        )

        # Intent classifier fallback
        self.INTENT_FALLBACK_CATEGORY = runtime.get_str("intent", "fallback_category", "专业咨询")
        self.INTENT_LLM_TEMPERATURE = runtime.get_float("intent", "llm_temperature", 0.0)
        self.INTENT_SYSTEM_MESSAGE = runtime.get_str(
            "intent", "system_message", "你是一个严格的意图分类器。"
        )

        # Cache policy
        self.CACHE_TIME_SENSITIVE_PATTERN = runtime.get_str(
            "cache",
            "time_sensitive_pattern",
            "今天|本周|本月|今年|现在|实时|当前|几点|天气",
        )

    def _init_prompt_registry(self):
        prompts_dir = self.runtime.get_str("prompts", "dir", "config/prompts")
        prompts_path = Path(prompts_dir)
        if not prompts_path.is_absolute():
            prompts_path = self.PROJECT_ROOT / prompts_path
        if not prompts_path.exists():
            prompts_path = self.PROJECT_ROOT / "config" / "prompts"
        self.PROMPTS_DIR = str(prompts_path)
        self.prompts = PromptRegistry(prompts_path)

    def _runtime_or_ini_int(self, runtime_section, runtime_key, ini_section, ini_option, fallback):
        value = self.runtime.get(runtime_section, runtime_key, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback
        return self._getint(ini_section, ini_option, fallback)

    def _runtime_or_ini_float(self, runtime_section, runtime_key, ini_section, ini_option, fallback):
        value = self.runtime.get(runtime_section, runtime_key, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback
        return self._getfloat(ini_section, ini_option, fallback)

    def _runtime_or_ini_bool(self, runtime_section, runtime_key, ini_section, ini_option, fallback):
        value = self.runtime.get(runtime_section, runtime_key, None)
        if value is not None:
            try:
                from base.runtime_loader import _coerce_bool

                return _coerce_bool(value)
            except ValueError:
                return fallback
        return self._getbool(ini_section, ini_option, fallback)

    def _resolve_path(self, path_value):
        path = Path(path_value)
        if not path.is_absolute():
            path = self.PROJECT_ROOT / path
        return str(path)

    def _env(self, *names, default=None):
        for name in names:
            if name in os.environ:
                return os.environ[name]
        return default

    def _get(self, section, option, fallback, env_name=None):
        env_value = self._env(env_name, default=None) if env_name else None
        if env_value is not None:
            return env_value
        return self.config.get(section, option, fallback=fallback)

    def _getint(self, section, option, fallback, env_name=None):
        value = self._get(section, option, str(fallback), env_name)
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _getfloat(self, section, option, fallback, env_name=None):
        value = self._get(section, option, str(fallback), env_name)
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _getbool(self, section, option, fallback, env_name=None):
        value = str(self._get(section, option, str(fallback), env_name)).strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off"}:
            return False
        return fallback

    @property
    def BGE_M3_PATH(self):
        return self._resolve_path(
            self._get("models", "bge_m3_path", "models/bge-m3", "BGE_M3_PATH")
        )

    @property
    def BGE_RERANKER_PATH(self):
        return self._resolve_path(
            self._get(
                "models", "bge_reranker_path", "models/bge-reranker-large", "BGE_RERANKER_PATH"
            )
        )

    @staticmethod
    def _parse_list(value):
        if isinstance(value, list):
            return value
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = [item.strip() for item in str(value).split(",") if item.strip()]
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
        return [str(parsed)]


if __name__ == "__main__":
    conf = Config()
    print(conf.CHILD_CHUNK_SIZE)
    print(conf.DASHSCOPE_API_KEY)
    print(conf.DASHSCOPE_BASE_URL)
    print(conf.RETRIEVAL_K)
    print(conf.RUNTIME_FILE)
