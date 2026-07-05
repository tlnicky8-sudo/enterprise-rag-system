from __future__ import annotations

import re
from collections import Counter

from core.ingest.models import ChildBlock, EnrichedBlock, IngestContext

TOKEN = re.compile(r"[\u4e00-\u9fa5A-Za-z0-9]{2,}")
STOPWORDS = {
    "规定",
    "可以",
    "应当",
    "劳动者",
    "用人单位",
    "或者",
    "以及",
    "不得",
    "根据",
}


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    tokens = [token for token in TOKEN.findall(text) if token not in STOPWORDS]
    counts = Counter(tokens)
    return [token for token, _ in counts.most_common(limit)]


def build_hypothetical_questions(keywords: list[str], section_path: str) -> list[str]:
    questions = []
    if section_path and section_path != "正文":
        questions.append(f"{section_path} 是怎么规定的？")
    for keyword in keywords[:3]:
        questions.append(f"{keyword} 相关法律规定是什么？")
    return questions[:4]


class EnhanceStage:
    name = "enhance"

    def run(self, ctx: IngestContext, chunks: list[ChildBlock]) -> list[EnrichedBlock]:
        outputs: list[EnrichedBlock] = []
        for chunk in chunks:
            keywords = extract_keywords(chunk.text) if ctx.enhance else []
            questions = (
                build_hypothetical_questions(keywords, chunk.parent.section_path)
                if ctx.enhance
                else []
            )
            metadata = {
                **chunk.metadata,
                "keywords": keywords,
                "hypothetical_questions": questions,
            }
            index_text = chunk.text
            if questions:
                index_text = f"{chunk.text}\n\n可能问题：\n" + "\n".join(questions)
            outputs.append(
                EnrichedBlock(
                    child=chunk,
                    index_text=index_text,
                    keywords=keywords,
                    hypothetical_questions=questions,
                    metadata=metadata,
                )
            )
        ctx.add_stage(
            self.name,
            input_count=len(chunks),
            output_count=len(outputs),
            notes=["enabled" if ctx.enhance else "disabled"],
        )
        return outputs

