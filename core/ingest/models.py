from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".ppt",
    ".pptx",
    ".jpg",
    ".jpeg",
    ".png",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class FileRef:
    path: Path
    source: str
    file_hash: str
    mtime: float

    @classmethod
    def from_path(cls, path: Path, source: str) -> "FileRef":
        stat = path.stat()
        return cls(
            path=path,
            source=source,
            file_hash=sha256_bytes(path.read_bytes()),
            mtime=stat.st_mtime,
        )


@dataclass
class ParsedDocument:
    file_ref: FileRef
    title: str
    markdown: str
    markdown_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanDocument:
    parsed: ParsedDocument
    text: str
    headings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParentBlock:
    parent_id: str
    doc_id: str
    text: str
    section_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChildBlock:
    child_id: str
    parent: ParentBlock
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichedBlock:
    child: ChildBlock
    index_text: str
    keywords: list[str] = field(default_factory=list)
    hypothetical_questions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageReport:
    name: str
    input_count: int
    output_count: int
    skipped: int = 0
    failed: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class IngestContext:
    run_id: str
    project_root: Path
    source: str
    source_dir: Path
    processed_dir: Path
    report_dir: Path
    dry_run: bool = False
    enhance: bool = False
    doc_version: str = "1"
    started_at: str = field(default_factory=now_iso)
    stage_reports: list[StageReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_stage(
        self,
        name: str,
        input_count: int,
        output_count: int,
        skipped: int = 0,
        failed: int = 0,
        notes: list[str] | None = None,
    ) -> None:
        self.stage_reports.append(
            StageReport(
                name=name,
                input_count=input_count,
                output_count=output_count,
                skipped=skipped,
                failed=failed,
                notes=notes or [],
            )
        )

