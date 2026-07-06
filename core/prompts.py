from base import Config


class RAGPrompts:
    """Prompt templates loaded from config/prompts/*.txt."""

    _registry = None
    _registry_path = None

    @classmethod
    def _get_registry(cls, conf=None):
        conf = conf or Config()
        prompts_path = str(conf.PROMPTS_DIR)
        if cls._registry is None or cls._registry_path != prompts_path:
            cls._registry = conf.prompts
            cls._registry_path = prompts_path
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
