"""extract_metadata: frontmatter, title, headers, section summaries."""
from pathlib import Path

from lce.indexer.markdown_meta import extract_metadata

FIXTURE = Path("tests/fixtures/sample.md")


def test_fixture_metadata():
    md = extract_metadata(FIXTURE.read_text())
    assert md["title"] == "Sample Note"
    assert md["frontmatter"] == {"title": "Sample Note", "tags": ["lce", "fixture"]}
    assert md["headers"] == [
        {"level": 1, "text": "Sample Note"},
        {"level": 2, "text": "Section One"},
        {"level": 2, "text": "Section Two"},
    ]
    assert md["sections"] == [
        {"header": "Sample Note", "summary": "Intro paragraph."},
        {"header": "Section One", "summary": "Body text one."},
        {"header": "Section Two", "summary": "Body text two."},
    ]


def test_no_frontmatter():
    md = extract_metadata("# Title\n\nBody.\n")
    assert md["title"] == "Title"
    assert md["frontmatter"] == {}


def test_frontmatter_title_fallback():
    md = extract_metadata("---\ntitle: From FM\n---\n\nNo headers here.\n")
    assert md["title"] == "From FM"
    assert md["headers"] == []


def test_empty_text():
    assert extract_metadata("") == {
        "title": None,
        "frontmatter": {},
        "headers": [],
        "sections": [],
    }


def test_summary_truncated_at_200_chars():
    md = extract_metadata("# H\n\n" + "x" * 500 + "\n")
    assert len(md["sections"][0]["summary"]) == 200


def test_fenced_code_lines_are_not_headers():
    text = (
        "# Real Title\n\nIntro.\n\n"
        "```bash\n# not a header\n## also not\n```\n\n"
        "## Real Section\n\nBody.\n"
    )
    md = extract_metadata(text)
    assert [h["text"] for h in md["headers"]] == ["Real Title", "Real Section"]


def test_summary_stops_at_code_fence():
    text = "# T\n\n```python\nx = 1\n```\n\nProse after.\n"
    md = extract_metadata(text)
    assert md["sections"][0]["summary"] == ""
