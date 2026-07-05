from __future__ import annotations

import re

from langchain_core.documents import Document

from base import Config
from core.ingest.models import ChildBlock, CleanDocument, IngestContext, ParentBlock, content_hash, now_iso
from text_spliter import ChineseRecursiveTextSplitter

conf = Config()

ARTICLE_HEADING = re.compile(r"(?m)^###\s+(第[一二三四五六七八九十百零〇\d]+条)\s*")
HEADING = re.compile(r"(?m)^#{1,6}\s+(.+)$")


def _doc_id(file_path: str, file_hash: str) -> str:
    return content_hash(f"{file_path}:{file_hash}")[:16]


def _split_by_articles(text: str) -> list[tuple[str, str]]:
    matches = list(ARTICLE_HEADING.finditer(text))
    if not matches:
        return []

    parts: list[tuple[str, str]] = []
    prefix = text[: matches[0].start()].strip()
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_title = match.group(1)
        body = text[start:end].strip()
        if prefix and idx == 0:
            body = f"{prefix}\n\n{body}"
        parts.append((section_title, body))
    return parts


def _section_path(headings: list[str], fallback: str, index: int) -> str:
    if fallback:
        return fallback
    if headings:
        return " > ".join(headings[: min(index + 1, len(headings))])
    return "正文"


class ChunkStage:
    name = "chunk"

    def __init__(
        self,
        parent_chunk_size: int = conf.PARENT_CHUNK_SIZE,
        child_chunk_size: int = conf.CHILD_CHUNK_SIZE,
        chunk_overlap: int = conf.CHUNK_OVERLAP,
        min_child_chars: int = 20,
    ) -> None:
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_child_chars = min_child_chars
        self.parent_splitter = ChineseRecursiveTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.child_splitter = ChineseRecursiveTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _parent_units(self, doc: CleanDocument) -> list[tuple[str, str]]:
        article_units = _split_by_articles(doc.text)
        if article_units:
            return article_units
        base_doc = Document(page_content=doc.text, metadata=dict(doc.metadata))
        parents = self.parent_splitter.split_documents([base_doc])
        return [("", item.page_content) for item in parents]

    def run(self, ctx: IngestContext, docs: list[CleanDocument]) -> list[ChildBlock]:
        outputs: list[ChildBlock] = []
        skipped = 0

        for doc in docs:
            file_path = doc.metadata["source_file"]
            file_hash = doc.metadata["file_hash"]
            doc_id = _doc_id(file_path, file_hash)
            units = self._parent_units(doc)
            for parent_index, (section_title, parent_text) in enumerate(units):
                parent_text = parent_text.strip()
                if not parent_text:
                    skipped += 1
                    continue
                parent_id = f"{doc_id}_p{parent_index}"
                section = _section_path(doc.headings, section_title, parent_index)
                parent_metadata = {
                    **doc.metadata,
                    "doc_id": doc_id,
                    "parent_id": parent_id,
                    "section_path": section,
                    "ingest_time": now_iso(),
                    "content_hash": content_hash(parent_text),
                }
                parent = ParentBlock(
                    parent_id=parent_id,
                    doc_id=doc_id,
                    text=parent_text,
                    section_path=section,
                    metadata=parent_metadata,
                )
                child_docs = self.child_splitter.split_documents(
                    [Document(page_content=parent_text, metadata=parent_metadata)]
                )
                for child_index, child_doc in enumerate(child_docs):
                    text = child_doc.page_content.strip()
                    if len(text) < self.min_child_chars:
                        skipped += 1
                        continue
                    child_id = f"{parent_id}_c{child_index}"
                    metadata = {
                        **parent_metadata,
                        "child_id": child_id,
                        "parent_content": parent.text,
                        "chunk_index": child_index,
                        "chunk_type": "child",
                        "content_hash": content_hash(text),
                    }
                    outputs.append(
                        ChildBlock(
                            child_id=child_id,
                            parent=parent,
                            text=text,
                            metadata=metadata,
                        )
                    )

        ctx.add_stage(
            self.name,
            input_count=len(docs),
            output_count=len(outputs),
            skipped=skipped,
            notes=[
                f"parent={self.parent_chunk_size}",
                f"child={self.child_chunk_size}",
                f"overlap={self.chunk_overlap}",
            ],
        )
        return outputs

