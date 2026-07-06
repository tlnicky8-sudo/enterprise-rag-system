from langchain_core.documents import Document

from core.citations import build_citations_from_docs


def test_build_citations_prefers_section_path():
    docs = [
        Document(
            page_content="员工累计工作满一年可享受带薪年假。",
            metadata={
                "source": "enterprise",
                "source_file": "/data/enterprise_data/employee_handbook.md",
                "section_path": "年假制度",
                "parent_id": "abc_p0",
            },
        )
    ]

    citations = build_citations_from_docs(docs)

    assert len(citations) == 1
    assert citations[0]["title"] == "年假制度"
    assert citations[0]["source"] == "enterprise"
    assert "年假" in citations[0]["excerpt"]


def test_build_citations_falls_back_to_filename():
    docs = [
        Document(
            page_content="试用期最长不得超过六个月。",
            metadata={"source": "enterprise", "source_file": "/data/employee_handbook.md"},
        )
    ]

    citations = build_citations_from_docs(docs)

    assert citations[0]["title"] == "employee_handbook.md"
