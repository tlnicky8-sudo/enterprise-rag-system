from core.llm_utils import collect_llm_output, iter_llm_output
from core.text_utils import normalize_question, preprocess_text


def test_normalize_question_from_tuple():
    assert normalize_question(("试用期最长多久？",)) == "试用期最长多久？"


def test_preprocess_text_returns_tokens():
    tokens = preprocess_text("劳动合同法")
    assert isinstance(tokens, list)
    assert len(tokens) >= 1


def test_collect_llm_output_from_string():
    assert collect_llm_output("hello") == "hello"


def test_collect_llm_output_from_iterator():
    assert collect_llm_output(iter(["a", "b"])) == "ab"


def test_iter_llm_output_from_string():
    assert list(iter_llm_output("abc")) == ["abc"]
