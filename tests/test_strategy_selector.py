from core.strategy_selector import StrategySelector, heuristic_strategy, normalize_strategy


def test_normalize_strategy_accepts_exact_name():
    assert normalize_strategy("子查询检索") == "子查询检索"


def test_normalize_strategy_extracts_name_from_sentence():
    assert normalize_strategy("策略：子查询检索。") == "子查询检索"


def test_normalize_strategy_supports_aliases():
    assert normalize_strategy("HyDE") == "假设问题检索"


def test_normalize_strategy_falls_back_to_default():
    assert normalize_strategy("无法判断", default_strategy="直接检索") == "直接检索"


def test_heuristic_strategy_keeps_simple_fact_direct():
    assert heuristic_strategy("公司年假有多少天？") == "直接检索"


def test_heuristic_strategy_selects_subquery_for_comparison():
    assert heuristic_strategy("比较专业通道和管理通道的晋升条件有什么区别？") == "子查询检索"


def test_heuristic_strategy_selects_hyde_for_abstract_query():
    assert heuristic_strategy("天恒科技的企业文化体现在哪些方面？") == "假设问题检索"


def test_heuristic_strategy_selects_backtracking_for_complex_scenario():
    query = "我工作三年准备离职，竞业限制、社保转移和交接要注意什么？"
    assert heuristic_strategy(query) == "回溯问题检索"


class _Conf:
    DEFAULT_STRATEGY = "直接检索"
    STRATEGY_ENABLE_HEURISTIC_FALLBACK = True
    STRATEGY_HEURISTIC_OVERRIDE_DIRECT = True


class _Prompt:
    def format(self, query):
        return query


def test_selector_heuristic_overrides_direct(monkeypatch):
    selector = StrategySelector.__new__(StrategySelector)
    selector.conf = _Conf()
    selector.strategy_prompt_template = _Prompt()
    monkeypatch.setattr(selector, "call_dashscope", lambda prompt: "直接检索")

    assert selector.select_strategy("比较专业通道和管理通道的晋升条件有什么区别？") == "子查询检索"


def test_selector_does_not_override_non_direct(monkeypatch):
    selector = StrategySelector.__new__(StrategySelector)
    selector.conf = _Conf()
    selector.strategy_prompt_template = _Prompt()
    monkeypatch.setattr(selector, "call_dashscope", lambda prompt: "假设问题检索")

    assert selector.select_strategy("比较专业通道和管理通道的晋升条件有什么区别？") == "假设问题检索"
