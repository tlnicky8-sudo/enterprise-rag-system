from pathlib import Path


def _display_title(metadata, fallback_source="未知来源"):
    section_path = str(metadata.get("section_path") or "").strip()
    if section_path:
        return section_path

    source_file = str(metadata.get("source_file") or "").strip()
    if source_file:
        return Path(source_file).name

    source = str(metadata.get("source") or "").strip()
    return source or fallback_source


def build_citations_from_docs(docs, max_excerpt=500):
    """Build frontend-friendly citation payloads from retrieved documents."""
    citations = []
    for index, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", None) or {}
        content = (getattr(doc, "page_content", None) or "").strip()
        if not content:
            continue

        excerpt = content[:max_excerpt]
        if len(content) > max_excerpt:
            excerpt += "…"

        citations.append(
            {
                "id": index,
                "title": _display_title(metadata),
                "source": str(metadata.get("source") or ""),
                "source_file": str(metadata.get("source_file") or ""),
                "section_path": str(metadata.get("section_path") or ""),
                "doc_type": str(metadata.get("doc_type") or ""),
                "parent_id": str(metadata.get("parent_id") or ""),
                "excerpt": excerpt,
            }
        )
    return citations
