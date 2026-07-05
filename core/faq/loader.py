import json
from pathlib import Path


def _normalize_record(record, default_subject=None):
    if not isinstance(record, dict):
        raise ValueError(f"FAQ record must be an object, got {type(record).__name__}")

    question = str(record.get("question") or record.get("q") or "").strip()
    answer = str(record.get("answer") or record.get("a") or "").strip()
    subject = record.get("subject_name") or record.get("domain") or default_subject

    if not question or not answer:
        return None
    if subject is not None:
        subject = str(subject).strip() or default_subject

    return subject, question, answer


def load_faq_pairs(path, default_subject=None):
    """Load FAQ pairs from JSONL or JSON file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"FAQ data file not found: {file_path}")

    suffix = file_path.suffix.lower()
    records = []

    if suffix == ".jsonl":
        with file_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    elif suffix == ".json":
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get("items") or payload.get("data") or payload.get("faqs") or []
            if not isinstance(records, list):
                raise ValueError("JSON FAQ file must contain a list or an items/data/faqs array")
        else:
            raise ValueError("JSON FAQ file must contain a list or object")
    else:
        raise ValueError(f"Unsupported FAQ file format: {suffix}. Use .jsonl or .json")

    pairs = []
    seen = set()
    skipped = 0
    for record in records:
        normalized = _normalize_record(record, default_subject=default_subject)
        if normalized is None:
            skipped += 1
            continue

        subject, question, answer = normalized
        dedupe_key = (subject, question)
        if dedupe_key in seen:
            skipped += 1
            continue
        seen.add(dedupe_key)
        pairs.append((subject, question, answer))

    return pairs, skipped


def to_question_answer_pairs(pairs):
    """Normalize FAQ records to (question, answer) tuples for Redis storage."""
    normalized = []
    for item in pairs:
        if not isinstance(item, (list, tuple)):
            raise ValueError(f"Invalid FAQ pair: {item!r}")
        if len(item) == 3:
            _, question, answer = item
        elif len(item) == 2:
            question, answer = item
        else:
            raise ValueError(f"Invalid FAQ pair arity: {item!r}")

        question = str(question).strip()
        answer = str(answer).strip()
        if question and answer:
            normalized.append((question, answer))
    return normalized
