import importlib.util
from pathlib import Path


def _load_live_eval_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "live_eval.py"
    spec = importlib.util.spec_from_file_location("live_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_cases_and_keyword_recall(tmp_path):
    live_eval = _load_live_eval_module()
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"id":"case1","question":"试用期多久","expected_keywords":["试用期","六个月"]}\n',
        encoding="utf-8",
    )

    cases = live_eval.load_cases(golden)

    assert cases[0].id == "case1"
    assert live_eval.keyword_recall("试用期最长六个月", cases[0].expected_keywords) == 1.0


def test_summarize_records():
    live_eval = _load_live_eval_module()
    records = [
        live_eval.EvalRecord(
            id="a",
            question="q1",
            passed=True,
            source="rag",
            grounded=True,
            refused=False,
            citation_count=2,
            latency_ms=100,
            keyword_recall=1.0,
            answer="a1",
            failure_reasons=[],
        ),
        live_eval.EvalRecord(
            id="b",
            question="q2",
            passed=False,
            source="rag",
            grounded=False,
            refused=True,
            citation_count=0,
            latency_ms=300,
            keyword_recall=0.0,
            answer="a2",
            failure_reasons=["failed"],
        ),
    ]

    summary = live_eval.summarize(records)

    assert summary["total"] == 2
    assert summary["pass_rate"] == 0.5
    assert summary["avg_latency_ms"] == 200
    assert summary["citation_coverage"] == 0.5
    assert summary["refusal_rate"] == 0.5


def test_refusal_detection_uses_answer_text():
    live_eval = _load_live_eval_module()

    class Result:
        refusal_reason = ""

    assert live_eval.is_refused(Result(), "信息不足，无法回答，请联系人工客服。") is True
    assert live_eval.is_refused(Result(), "可以正常回答。") is False
