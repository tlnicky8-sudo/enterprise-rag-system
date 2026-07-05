from __future__ import annotations

from langchain_core.documents import Document

from core.ingest.models import EnrichedBlock, IngestContext
from core.vector_store import VectorStore


class IndexStage:
    name = "index"

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self.vector_store = vector_store

    def run(self, ctx: IngestContext, chunks: list[EnrichedBlock]) -> int:
        documents = []
        for item in chunks:
            child = item.child
            metadata = {
                **item.metadata,
                "id": child.child_id,
                "parent_id": child.parent.parent_id,
                "parent_content": child.parent.text,
                "source": child.metadata.get("source", ctx.source),
                "timestamp": child.metadata.get("ingest_time"),
                "index_text": item.index_text,
            }
            documents.append(Document(page_content=child.text, metadata=metadata))

        if ctx.dry_run:
            written = len(documents)
        else:
            if self.vector_store is None:
                raise RuntimeError("IndexStage 需要 VectorStore 才能写入 Milvus")
            written = self.vector_store.add_documents(documents)
        ctx.add_stage(self.name, input_count=len(chunks), output_count=written)
        return written

