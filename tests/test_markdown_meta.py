"""extract_metadata: frontmatter, title, headers (outline-only since phase 3)."""
from pathlib import Path

from lce.indexer.markdown_meta import extract_metadata

FIXTURE = Path("tests/fixtures/sample.md")


def test_fixture_metadata():
    md = extract_metadata(FIXTURE.read_text())
    assert set(md) == {"title", "frontmatter", "headers"}
    assert md["title"] == "Sample Note"
    assert md["frontmatter"] == {"title": "Sample Note", "tags": ["lce", "fixture"]}
    assert md["headers"] == [
        {"level": 1, "text": "Sample Note"},
        {"level": 2, "text": "Section One"},
        {"level": 2, "text": "Section Two"},
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
    }


def test_fenced_code_lines_are_not_headers():
    text = (
        "# Real Title\n\nIntro.\n\n"
        "```bash\n# not a header\n## also not\n```\n\n"
        "## Real Section\n\nBody.\n"
    )
    md = extract_metadata(text)
    assert [h["text"] for h in md["headers"]] == ["Real Title", "Real Section"]


