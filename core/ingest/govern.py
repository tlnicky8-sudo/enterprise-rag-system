from __future__ import annotations

import json
from collections import defaultdict

from base import logger
from core.ingest.models import EnrichedBlock, IngestContext, now_iso


class GovernStage:
    name = "govern"

    def run(self, ctx: IngestContext, chunks: list[EnrichedBlock], written: int) -> dict:
        ctx.report_dir.mkdir(parents=True, exist_ok=True)
        lineage_by_doc: dict[str, dict] = {}
        chunk_count_by_doc = defaultdict(int)

        for item in chunks:
            metadata = item.metadata
            doc_id = metadata["doc_id"]
            chunk_count_by_doc[doc_id] += 1
            lineage_by_doc.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "source": metadata.get("source", ctx.source),
                    "source_file": metadata.get("source_file"),
                    "file_hash": metadata.get("file_hash"),
                    "markdown_path": metadata.get("markdown_path"),
                    "doc_type": metadata.get("doc_type"),
                    "version": ctx.doc_version,
                    "ingest_time": metadata.get("ingest_time"),
                    "chunks": [],
                },
            )
            lineage_by_doc[doc_id]["chunks"].append(
                {
                    "child_id": item.child.child_id,
                    "parent_id": item.child.parent.parent_id,
                    "section_path": item.child.parent.section_path,
                    "content_hash": metadata.get("content_hash"),
                    "keywords": item.keywords,
                    "hypothetical_questions": item.hypothetical_questions,
                }
            )

        report = {
            "run_id": ctx.run_id,
            "source": ctx.source,
            "source_dir": str(ctx.source_dir),
            "started_at": ctx.started_at,
            "finished_at": now_iso(),
            "dry_run": ctx.dry_run,
            "enhance": ctx.enhance,
            "documents": len(lineage_by_doc),
            "chunks_indexed": written,
            "stage_reports": [stage.__dict__ for stage in ctx.stage_reports],
            "errors": ctx.errors,
            "lineage": list(lineage_by_doc.values()),
        }

        report_path = ctx.report_dir / f"ingest_report_{ctx.run_id}.json"
        if not ctx.dry_run:
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            "入库审计 run=%s source=%s docs=%d chunks=%d report=%s",
            ctx.run_id,
            ctx.source,
            len(lineage_by_doc),
            written,
            report_path,
        )
        ctx.add_stage(self.name, input_count=len(chunks), output_count=len(chunks))
        report["stage_reports"] = [stage.__dict__ for stage in ctx.stage_reports]
        return report

