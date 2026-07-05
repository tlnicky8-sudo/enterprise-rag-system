from __future__ import annotations

import uuid
from pathlib import Path

from base import Config, logger
from core.ingest.chunk import ChunkStage
from core.ingest.clean import CleanStage
from core.ingest.enhance import EnhanceStage
from core.ingest.govern import GovernStage
from core.ingest.index import IndexStage
from core.ingest.models import IngestContext
from core.ingest.parse import ParseStage, discover_files
from core.vector_store import VectorStore


class IngestPipeline:
    """Six-stage legal corpus ingestion pipeline."""

    def __init__(self, vector_store: VectorStore | None = None, conf: Config | None = None) -> None:
        self.conf = conf or Config()
        self.parse_stage = ParseStage()
        self.clean_stage = CleanStage()
        self.chunk_stage = ChunkStage(
            parent_chunk_size=self.conf.PARENT_CHUNK_SIZE,
            child_chunk_size=self.conf.CHILD_CHUNK_SIZE,
            chunk_overlap=self.conf.CHUNK_OVERLAP,
        )
        self.enhance_stage = EnhanceStage()
        self.index_stage = IndexStage(vector_store)
        self.govern_stage = GovernStage()

    def run_directory(
        self,
        source_dir: Path,
        *,
        source: str,
        dry_run: bool = False,
        enhance: bool = False,
        doc_version: str = "1",
    ) -> dict:
        source_dir = Path(source_dir)
        ctx = IngestContext(
            run_id=str(uuid.uuid4())[:8],
            project_root=Path(self.conf.PROJECT_ROOT),
            source=source,
            source_dir=source_dir,
            processed_dir=Path(self.conf.PROJECT_ROOT) / "data" / "processed",
            report_dir=Path(self.conf.PROJECT_ROOT) / "data" / "ingest_reports",
            dry_run=dry_run,
            enhance=enhance,
            doc_version=doc_version,
        )

        files = discover_files(source_dir, source)
        logger.info("发现待入库文件 %d 个: %s", len(files), source_dir)
        if not files:
            ctx.add_stage("discover", input_count=0, output_count=0, skipped=0)
            return self.govern_stage.run(ctx, [], written=0)

        parsed = self.parse_stage.run(ctx, files)
        cleaned = self.clean_stage.run(ctx, parsed)
        chunks = self.chunk_stage.run(ctx, cleaned)
        enriched = self.enhance_stage.run(ctx, chunks)
        written = self.index_stage.run(ctx, enriched)
        return self.govern_stage.run(ctx, enriched, written=written)

