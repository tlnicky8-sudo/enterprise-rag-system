from __future__ import annotations

import re

from core.ingest.models import CleanDocument, IngestContext, ParsedDocument, content_hash

# 清洗阶段：去页码噪声、统一标题、脱敏，适用于企业制度/手册 PDF、Word、Markdown。

PAGE_NOISE = re.compile(r"^\s*(第\s*)?\d+\s*(页)?\s*$", re.MULTILINE)
MULTI_BLANK = re.compile(r"\n{3,}")
SPACES = re.compile(r"[ \t]+")
# PDF/Word 导出时常见的「第X章 / 第X节 / 第X条」标题，统一转为 Markdown 方便后续按章节切块
NUMBERED_CHAPTER = re.compile(r"(?m)^\s*(第[一二三四五六七八九十百零〇\d]+章)\s*(.+)?$")
NUMBERED_SECTION = re.compile(r"(?m)^\s*(第[一二三四五六七八九十百零〇\d]+节)\s*(.+)?$")
NUMBERED_ARTICLE = re.compile(r"(?m)^\s*(第[一二三四五六七八九十百零〇\d]+条)\s*")
PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
HEADING = re.compile(r"(?m)^#{1,6}\s+(.+)$")


def _repair_document_headings(text: str) -> str:
    text = NUMBERED_CHAPTER.sub(lambda m: f"## {m.group(1)} {m.group(2) or ''}".rstrip(), text)
    text = NUMBERED_SECTION.sub(lambda m: f"### {m.group(1)} {m.group(2) or ''}".rstrip(), text)
    text = NUMBERED_ARTICLE.sub(lambda m: f"### {m.group(1)} ", text)
    return text


def _dedupe_lines(text: str) -> str:
    seen = set()
    lines = []
    for line in text.splitlines():
        normalized = line.strip()
        if not normalized:
            lines.append("")
            continue
        digest = content_hash(normalized)
        if digest in seen:
            continue
        seen.add(digest)
        lines.append(line)
    return "\n".join(lines)


def clean_text(text: str, desensitize: bool = True) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = PAGE_NOISE.sub("", text)
    text = SPACES.sub(" ", text)
    text = _repair_document_headings(text)
    text = _dedupe_lines(text)
    text = MULTI_BLANK.sub("\n\n", text).strip()
    if desensitize:
        text = PHONE.sub("[手机号]", text)
        text = ID_CARD.sub("[身份证号]", text)
    return text


class CleanStage:
    name = "clean"

    def __init__(self, desensitize: bool = True) -> None:
        self.desensitize = desensitize

    def run(self, ctx: IngestContext, docs: list[ParsedDocument]) -> list[CleanDocument]:
        outputs: list[CleanDocument] = []
        skipped = 0
        seen_docs = set()
        for doc in docs:
            text = clean_text(doc.markdown, desensitize=self.desensitize)
            digest = content_hash(text)
            if not text:
                skipped += 1
                continue
            if digest in seen_docs:
                skipped += 1
                continue
            seen_docs.add(digest)
            headings = [match.group(1).strip() for match in HEADING.finditer(text)]
            metadata = {
                **doc.metadata,
                "clean_hash": digest,
                "headings": headings,
                "desensitized": self.desensitize,
            }
            outputs.append(CleanDocument(parsed=doc, text=text, headings=headings, metadata=metadata))
        ctx.add_stage(self.name, input_count=len(docs), output_count=len(outputs), skipped=skipped)
        return outputs
