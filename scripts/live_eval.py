"""Run live end-to-end evaluation through QAPipeline.

This script intentionally exercises the real runtime path: Redis FAQ,
Milvus retrieval, rerank, grounding gate, and the configured LLM.
"""

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from base import Config, logger
from core.qa_pipeline import QAPipeline
from core.rag_system import RAGSystem
from core.vector_store import VectorStore


@dataclass
class EvalCase:
    id: str
    question: str
    expected_source: str | None = None
    expected_grounded: bool | None = None
    expected_refusal: bool | None = None
    expected_keywords: list[str] | None = None
    source_filter: str | None = None


@dataclass
class EvalRecord:
    id: str
    question: str
    passed: bool
    source: str
    grounded: bool
    refused: bool
    citation_count: int
    latency_ms: int
    keyword_recall: float | None
    answer: str
    failure_reasons: list[str]


def default_golden_path(conf):
    return conf.PROJECT_ROOT / "data" / "assessment_data" / "live_eval_golden.jsonl"


def default_output_dir(conf):
    return conf.PROJECT_ROOT / "data" / "assessment_data" / "live_eval_results"


def load_cases(path):
    cases = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not payload.get("question"):
                raise ValueError(f"Missing question at line {line_no}")
            cases.append(
                EvalCase(
                    id=str(payload.get("id") or f"case_{line_no}"),
                    question=str(payload["question"]),
                    expected_source=payload.get("expected_source"),
                    expected_grounded=payload.get("expected_grounded"),
                    expected_refusal=payload.get("expected_refusal"),
                    expected_keywords=payload.get("expected_keywords") or [],
                    source_filter=payload.get("source_filter"),
                )
            )
    return cases


def keyword_recall(answer, keywords):
    if not keywords:
        return None
    hits = sum(1 for keyword in keywords if str(keyword) in answer)
    return hits / len(keywords)


def is_refused(result, answer):
    refusal_markers = ("信息不足", "无法回答", "无法基于", "联系人工客服")
    return bool(result.refusal_reason) or any(marker in answer for marker in refusal_markers)


def evaluate_expectations(case, result, answer):
    reasons = []
    refused = is_refused(result, answer)
    citation_count = len(result.citations)

    if case.expected_source and result.source != case.expected_source:
        reasons.append(f"source expected {case.expected_source}, got {result.source}")
    if case.expected_grounded is not None and result.grounded != case.expected_grounded:
        reasons.append(f"grounded expected {case.expected_grounded}, got {result.grounded}")
    if case.expected_refusal is not None and refused != case.expected_refusal:
        reasons.append(f"refusal expected {case.expected_refusal}, got {refused}")
    if case.expected_grounded and citation_count <= 0:
        reasons.append("grounded answer has no citations")

    recall = keyword_recall(answer, case.expected_keywords or [])
    if recall is not None and recall < 0.5:
        reasons.append(f"keyword recall too low: {recall:.2f}")

    return reasons, recall


def build_pipeline(conf):
    client = OpenAI(api_key=conf.DASHSCOPE_API_KEY, base_url=conf.DASHSCOPE_BASE_URL)

    def call_llm(prompt):
        completion = client.chat.completions.create(
            model=conf.LLM_MODEL,
            messages=[
                {"role": "system", "content": conf.GENERATION_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            timeout=60,
        )
        return completion.choices[0].message.content if completion.choices else ""

    vector_store = VectorStore(
        collection_name=conf.MILVUS_COLLECTION_NAME,
        host=conf.MILVUS_HOST,
        port=conf.MILVUS_PORT,
        database=conf.MILVUS_DATABASE_NAME,
    )
    rag_system = RAGSystem(vector_store, call_llm)
    return QAPipeline(rag_system=rag_system)


def run_eval(pipeline, cases, default_source_filter=None):
    records = []
    for case in cases:
        source_filter = case.source_filter or default_source_filter
        session_id = f"eval-{case.id}-{uuid.uuid4()}"
        start = time.perf_counter()
        result = pipeline.answer(
            case.question,
            session_id,
            source_filter=source_filter,
            stream=False,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        reasons, recall = evaluate_expectations(case, result, result.answer)
        refused = is_refused(result, result.answer)
        records.append(
            EvalRecord(
                id=case.id,
                question=case.question,
                passed=not reasons,
                source=result.source,
                grounded=result.grounded,
                refused=refused,
                citation_count=len(result.citations),
                latency_ms=latency_ms,
                keyword_recall=recall,
                answer=result.answer,
                failure_reasons=reasons,
            )
        )
    return records


def summarize(records):
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "passed": 0,
            "pass_rate": 0.0,
            "avg_latency_ms": 0,
            "citation_coverage": 0.0,
            "refusal_rate": 0.0,
            "grounded_rate": 0.0,
            "source_counts": {},
        }

    source_counts = {}
    for item in records:
        source_counts[item.source] = source_counts.get(item.source, 0) + 1

    return {
        "total": total,
        "passed": sum(1 for item in records if item.passed),
        "pass_rate": sum(1 for item in records if item.passed) / total,
        "avg_latency_ms": int(sum(item.latency_ms for item in records) / total),
        "citation_coverage": sum(1 for item in records if item.citation_count > 0) / total,
        "refusal_rate": sum(1 for item in records if item.refused) / total,
        "grounded_rate": sum(1 for item in records if item.grounded) / total,
        "source_counts": source_counts,
    }


def write_outputs(records, summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    detail_path = output_dir / f"live_eval_details_{timestamp}.jsonl"
    summary_path = output_dir / f"live_eval_summary_{timestamp}.json"

    with detail_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return detail_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="Run live QAPipeline evaluation")
    parser.add_argument("--golden", default=None, help="JSONL golden set path")
    parser.add_argument("--output-dir", default=None, help="Evaluation output directory")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N cases")
    parser.add_argument("--source-filter", default=None, help="Optional source filter")
    parser.add_argument(
        "--fail-under-pass-rate",
        type=float,
        default=0.0,
        help="Exit non-zero if pass rate is lower than this value",
    )
    args = parser.parse_args()

    conf = Config()
    golden_path = Path(args.golden) if args.golden else default_golden_path(conf)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(conf)

    cases = load_cases(golden_path)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError(f"No evaluation cases loaded from {golden_path}")

    logger.info("Starting live eval with %d cases", len(cases))
    pipeline = build_pipeline(conf)
    records = run_eval(pipeline, cases, default_source_filter=args.source_filter)
    summary = summarize(records)
    detail_path, summary_path = write_outputs(records, summary, output_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Details: {detail_path}")
    print(f"Summary: {summary_path}")

    if summary["pass_rate"] < args.fail_under_pass_rate:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
