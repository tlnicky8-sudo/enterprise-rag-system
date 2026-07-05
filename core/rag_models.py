from dataclasses import dataclass, field


@dataclass
class GroundingDecision:
    should_answer: bool
    grounded: bool = False
    refusal_reason: str = ""


@dataclass
class GenerationResult:
    answer: str
    source: str
    llm_confidence: float = 0.0
    rerank_score: float = 0.0
    has_context: bool = False
    citations: list = field(default_factory=list)
    grounded: bool = False
    refusal_reason: str = ""

    @property
    def citation_count(self):
        return len(self.citations)