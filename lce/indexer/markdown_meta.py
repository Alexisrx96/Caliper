"""Extract metadata from Markdown notes: frontmatter, headers, sections.

Foundation spec §4. Frontmatter parsing is deliberately minimal (key: value
pairs and [a, b] lists, no nesting) to avoid a yaml dependency.

Note: only ATX headers (`#` ... `######`) are recognized; setext headers
(underlined with `===`/`---`) are not.
"""
from __future__ import annotations

import re

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SUMMARY_LIMIT = 200


def extract_metadata(text: str) -> dict:
    """Return {'title', 'frontmatter', 'headers', 'sections'} for `text`.

    title: first H1, falling back to frontmatter 'title'. sections: one per
    header, summary = first paragraph after it (<=200 chars).
    """
    frontmatter, body = _split_frontmatter(text)
    headers: list[dict] = []
    sections: list[dict] = []
    lines = body.splitlines()
    in_fence = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADER_RE.match(line)
        if m:
            header_text = m.group(2).strip()
            headers.append({"level": len(m.group(1)), "text": header_text})
            sections.append(
                {"header": header_text, "summary": _first_paragraph(lines, i + 1)}
            )
    title = next((h["text"] for h in headers if h["level"] == 1), None)
    if title is None:
        fm_title = frontmatter.get("title")
        title = fm_title if isinstance(fm_title, str) else None
    return {
        "title": title,
        "frontmatter": frontmatter,
        "headers": headers,
        "sections": sections,
    }


def _split_frontmatter(text: str) -> tuple[dict, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return {}, text
    frontmatter: dict = {}
    for line in lines[1:end]:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        frontmatter[key.strip()] = _parse_value(value.strip())
    return frontmatter, "\n".join(lines[end + 1 :])


def _parse_value(value: str):
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [item.strip() for item in inner.split(",")] if inner else []
    return value


def _first_paragraph(lines: list[str], start: int) -> str:
    paragraph: list[str] = []
    for line in lines[start:]:
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            break  # code block: not prose, end of summary candidates
        if _HEADER_RE.match(line):
            break
        if line.strip():
            paragraph.append(line.strip())
        elif paragraph:
            break
    return " ".join(paragraph)[:_SUMMARY_LIMIT]
