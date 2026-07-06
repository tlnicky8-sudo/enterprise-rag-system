from openai import OpenAI

from base import Config, logger
from core.prompts import RAGPrompts


VALID_STRATEGIES = ("直接检索", "假设问题检索", "子查询检索", "回溯问题检索")
STRATEGY_ALIASES = {
    "hyde": "假设问题检索",
    "HyDE": "假设问题检索",
    "假设答案检索": "假设问题检索",
    "假设文档检索": "假设问题检索",
    "回溯检索": "回溯问题检索",
}

SUBQUERY_PATTERNS = (
    "比较",
    "区别",
    "差异",
    "不同",
    "分别",
    "各自",
    "同时",
    "以及",
    "和",
    "与",
    "、",
    "/",
)
BACKTRACK_PATTERNS = (
    "我现在",
    "我已经",
    "如果",
    "因为",
    "准备",
    "需要注意",
    "怎么办",
    "怎么处理",
    "被退回",
    "卡住",
    "入职",
    "离职",
    "转岗",
    "交接",
    "审批",
)
HYDE_PATTERNS = (
    "体现在哪些方面",
    "如何理解",
    "怎么看",
    "原则",
    "体系",
    "机制",
    "整体要求",
    "最佳实践",
    "核心要求",
    "有哪些方面",
)
SIMPLE_FACT_PATTERNS = (
    "多少",
    "几天",
    "几点",
    "在哪里",
    "谁审批",
    "怎么申请",
    "是否可以",
    "标准是什么",
)


def normalize_strategy(raw_strategy, default_strategy="直接检索"):
    """Normalize LLM output to one of the supported strategy names."""
    text = str(raw_strategy or "").strip().strip('"“”\'`。；;，,：:')
    for strategy in VALID_STRATEGIES:
        if strategy in text:
            return strategy
    for alias, strategy in STRATEGY_ALIASES.items():
        if alias in text:
            return strategy
    return default_strategy


def heuristic_strategy(query, default_strategy="直接检索"):
    """Choose a strategy using lightweight enterprise-query signals."""
    text = str(query or "").strip()
    if not text:
        return default_strategy

    normalized = text.replace("？", "?")
    clause_count = sum(normalized.count(mark) for mark in ("，", ",", "；", ";", "?"))
    multi_signal_count = sum(1 for pattern in SUBQUERY_PATTERNS if pattern in text)

    if any(pattern in text for pattern in BACKTRACK_PATTERNS) and (
        clause_count >= 2 or multi_signal_count >= 1 or len(text) >= 28
    ):
        return "回溯问题检索"

    if any(pattern in text for pattern in HYDE_PATTERNS):
        return "假设问题检索"

    if multi_signal_count >= 2 or (
        multi_signal_count >= 1 and any(word in text for word in ("什么", "哪些", "流程", "标准", "条件"))
    ):
        return "子查询检索"

    if len(text) >= 36 and clause_count >= 2:
        return "回溯问题检索"

    if any(pattern in text for pattern in SIMPLE_FACT_PATTERNS):
        return "直接检索"

    return default_strategy


class StrategySelector:
    def __init__(self, conf=None):
        self.conf = conf or Config()
        self.client = OpenAI(
            api_key=self.conf.DASHSCOPE_API_KEY,
            base_url=self.conf.DASHSCOPE_BASE_URL,
        )
        self.strategy_prompt_template = RAGPrompts.strategy_prompt(self.conf)

    def call_dashscope(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.conf.LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.conf.STRATEGY_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.conf.STRATEGY_LLM_TEMPERATURE,
            )
            return (
                completion.choices[0].message.content
                if completion.choices
                else self.conf.DEFAULT_STRATEGY
            )
        except Exception as exc:
            logger.error(f"DashScope API 调用失败: {exc}")
            return self.conf.DEFAULT_STRATEGY

    def select_strategy(self, query):
        raw_strategy = self.call_dashscope(self.strategy_prompt_template.format(query=query)).strip()
        strategy = normalize_strategy(raw_strategy, self.conf.DEFAULT_STRATEGY)
        if raw_strategy != strategy:
            logger.info(f"策略选择结果已归一化: '{raw_strategy}' -> '{strategy}'")

        if self.conf.STRATEGY_ENABLE_HEURISTIC_FALLBACK:
            heuristic = heuristic_strategy(query, self.conf.DEFAULT_STRATEGY)
            should_override = (
                heuristic != strategy
                and (
                    not self.conf.STRATEGY_HEURISTIC_OVERRIDE_DIRECT
                    or strategy == "直接检索"
                )
            )
            if should_override:
                logger.info(f"策略选择结果被启发式兜底覆盖: '{strategy}' -> '{heuristic}'")
                strategy = heuristic

        logger.info(f"为查询 '{query}' 选择的检索策略：{strategy}")
        return strategy


if __name__ == "__main__":
    ss = StrategySelector()
    print(ss.select_strategy("公司年假有多少天，以及离职后社保和公积金怎么处理？"))
