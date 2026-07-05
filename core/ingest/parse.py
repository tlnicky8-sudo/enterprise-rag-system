from __future__ import annotations

import json
from pathlib import Path

from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader

from base import logger
from core.ingest.models import FileRef, IngestContext, ParsedDocument, SUPPORTED_EXTENSIONS, now_iso
from document_loaders import OCRDOCLoader, OCRIMGLoader, OCRPDFLoader, OCRPPTLoader


LOADER_BY_EXTENSION = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pdf": OCRPDFLoader,
    ".docx": OCRDOCLoader,
    ".ppt": OCRPPTLoader,
    ".pptx": OCRPPTLoader,
    ".jpg": OCRIMGLoader,
    ".jpeg": OCRIMGLoader,
    ".png": OCRIMGLoader,
}


def discover_files(root: Path, source: str) -> list[FileRef]:
    files: list[FileRef] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(FileRef.from_path(path, source=source))
    return files


def _load_text(file_ref: FileRef) -> str:
    suffix = file_ref.path.suffix.lower()
    loader_cls = LOADER_BY_EXTENSION[suffix]
    if suffix == ".txt":
        loader = loader_cls(str(file_ref.path), encoding="utf-8")
    else:
        loader = loader_cls(str(file_ref.path))
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs if doc.page_content)


def _to_markdown(title: str, text: str) -> str:
    text = text.strip()
    if text.startswith("#"):
        return text
    return f"# {title}\n\n{text}\n"


class ParseStage:
    name = "parse"

    def run(self, ctx: IngestContext, files: list[FileRef]) -> list[ParsedDocument]:
        outputs: list[ParsedDocument] = []
        failed = 0
        markdown_root = ctx.processed_dir / ctx.source / "markdown"
        if not ctx.dry_run:
            markdown_root.mkdir(parents=True, exist_ok=True)

        for file_ref in files:
            try:
                title = file_ref.path.stem
                text = _load_text(file_ref)
                if not text.strip():
                    logger.warning(f"解析结果为空，跳过: {file_ref.path}")
                    continue
                markdown = _to_markdown(title, text)
                markdown_path = markdown_root / f"{file_ref.path.stem}.{file_ref.file_hash[:8]}.md"
                metadata_path = markdown_path.with_suffix(".json")
                metadata = {
                    "title": title,
                    "source": file_ref.source,
                    "source_file": str(file_ref.path),
                    "file_hash": file_ref.file_hash,
                    "doc_type": file_ref.path.suffix.lower().lstrip("."),
                    "parsed_at": now_iso(),
                    "markdown_path": str(markdown_path),
                }
                if not ctx.dry_run:
                    markdown_path.write_text(markdown, encoding="utf-8")
                    metadata_path.write_text(
                        json.dumps(metadata, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                outputs.append(
                    ParsedDocument(
                        file_ref=file_ref,
                        title=title,
                        markdown=markdown,
                        markdown_path=markdown_path,
                        metadata=metadata,
                    )
                )
            except Exception as exc:
                failed += 1
                message = f"解析失败 {file_ref.path}: {exc}"
                logger.error(message)
                ctx.errors.append(message)

        ctx.add_stage(self.name, input_count=len(files), output_count=len(outputs), failed=failed)
        return outputs

