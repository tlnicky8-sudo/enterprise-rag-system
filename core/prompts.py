from base import Config


class RAGPrompts:
    """Prompt templates loaded from config/prompts/*.txt."""

    _registry = None

    @classmethod
    def _get_registry(cls, conf=None):
        conf = conf or Config()
        if cls._registry is None or str(conf.PROMPTS_DIR) != str(conf.prompts.prompts_dir):
            cls._registry = conf.prompts
        return cls._registry

    @classmethod
    def rag_prompt(cls, conf=None):
        return cls._get_registry(conf).get_template("rag")

    @classmethod
    def hyde_prompt(cls, conf=None):
        return cls._get_registry(conf).get_template("hyde")

    @classmethod
    def subquery_prompt(cls, conf=None):
        return cls._get_registry(conf).get_template("subquery")

    @classmethod
    def backtracking_prompt(cls, conf=None):
        return cls._get_registry(conf).get_template("backtrack")

    @classmethod
    def strategy_prompt(cls, conf=None):
        return cls._get_registry(conf).get_template("strategy")

    @classmethod
    def intent_prompt(cls, conf=None):
        return cls._get_registry(conf).get_template("intent")
