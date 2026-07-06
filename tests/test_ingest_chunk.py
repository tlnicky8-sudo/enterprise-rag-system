from core.ingest.chunk import _split_by_markdown_sections, _split_by_numbered_articles


SAMPLE_HANDBOOK = """# 员工手册

## 第一章 考勤管理

### 1.1 工作时间
标准工时 9:00-18:30。

### 1.2 加班管理
加班须提前申请。

## 第二章 假期制度

### 2.1 年假
满一年享 5 天年假。
"""


def test_split_by_markdown_sections_uses_headings():
    parts = _split_by_markdown_sections(SAMPLE_HANDBOOK)

    assert len(parts) >= 4
    titles = [title for title, _ in parts]
    assert "1.1 工作时间" in titles
    assert "2.1 年假" in titles
    assert any("标准工时" in body for _, body in parts)


def test_split_by_numbered_articles_supports_legacy_format():
    text = """# 制度汇编

### 第一条 适用范围
适用于全体员工。

### 第二条 生效时间
自发布之日起执行。
"""
    parts = _split_by_numbered_articles(text)

    assert len(parts) == 2
    assert parts[0][0] == "第一条"
    assert "适用范围" in parts[0][1]
