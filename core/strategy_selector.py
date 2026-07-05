from langchain_core.prompts import PromptTemplate
from openai import OpenAI

from base import Config, logger
from core.prompts import RAGPrompts


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
        strategy = self.call_dashscope(self.strategy_prompt_template.format(query=query)).strip()
        logger.info(f"为查询 '{query}' 选择的检索策略：{strategy}")
        return strategy


if __name__ == "__main__":
    ss = StrategySelector()
    print(ss.select_strategy("劳动合同法规定的加班工资标准，以及试用期解除合同的条件是什么？"))
