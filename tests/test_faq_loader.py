import json

from core.faq.loader import load_faq_pairs, to_question_answer_pairs


def test_load_faq_pairs_from_jsonl(tmp_path):
    file_path = tmp_path / "faq.jsonl"
    rows = [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2", "subject_name": "custom"},
    ]
    file_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    pairs, skipped = load_faq_pairs(file_path, default_subject="enterprise")

    assert skipped == 0
    assert pairs == [("enterprise", "Q1", "A1"), ("custom", "Q2", "A2")]


def test_load_faq_pairs_skips_invalid_and_duplicates(tmp_path):
    file_path = tmp_path / "faq.jsonl"
    file_path.write_text(
        "\n".join(
            [
                json.dumps({"question": "Q1", "answer": "A1"}, ensure_ascii=False),
                json.dumps({"question": "", "answer": "A2"}, ensure_ascii=False),
                json.dumps({"question": "Q1", "answer": "A1-new"}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )

    pairs, skipped = load_faq_pairs(file_path, default_subject="enterprise")

    assert pairs == [("enterprise", "Q1", "A1")]
    assert skipped == 2


def test_to_question_answer_pairs_supports_loader_output():
    pairs = [("enterprise", "Q1", "A1"), ("custom", "Q2", "A2")]
    assert to_question_answer_pairs(pairs) == [("Q1", "A1"), ("Q2", "A2")]


def test_to_question_answer_pairs_supports_two_tuple_input():
    pairs = [("Q1", "A1"), ("Q2", "A2")]
    assert to_question_answer_pairs(pairs) == pairs
