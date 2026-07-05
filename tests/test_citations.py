from langchain_core.documents import Document

from core.citations import build_citations_from_docs


def test_build_citations_prefers_section_path():
    docs = [
        Document(
            page_content="用人单位应当在解除劳动合同时支付经济补偿。",
            metadata={
                "source": "labor_law",
                "source_file": "/data/labor_law_data/劳动合同法.pdf",
                "section_path": "第四十七条",
                "parent_id": "abc_p0",
            },
        )
    ]

    citations = build_citations_from_docs(docs)

    assert len(citations) == 1
    assert citations[0]["title"] == "第四十七条"
    assert citations[0]["source"] == "labor_law"
    assert "经济补偿" in citations[0]["excerpt"]


def test_build_citations_falls_back_to_filename():
    docs = [
        Document(
            page_content="试用期最长不得超过六个月。",
            metadata={"source": "labor_law", "source_file": "/data/劳动法.pdf"},
        )
    ]

    citations = build_citations_from_docs(docs)

    assert citations[0]["title"] == "劳动法.pdf"
