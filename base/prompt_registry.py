from pathlib import Path

from langchain_core.prompts import PromptTemplate

_PROMPT_VARIABLES = {
    "rag": ["context", "history", "question", "phone"],
    "hyde": ["query"],
    "subquery": ["query"],
    "backtrack": ["query"],
    "strategy": ["query"],
    "intent": ["query"],
}


class PromptRegistry:
    """Load prompt templates from config/prompts/*.txt."""

    def __init__(self, prompts_dir):
        self.prompts_dir = Path(prompts_dir)

    def _read(self, name):
        path = self.prompts_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    def get_template(self, name):
        return PromptTemplate(
            template=self._read(name),
            input_variables=_PROMPT_VARIABLES[name],
        )

    def get(self, name):
        return self._read(name)
