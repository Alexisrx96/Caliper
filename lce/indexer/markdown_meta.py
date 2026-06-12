"""Extract metadata from Markdown notes: frontmatter and headers.

Foundation spec §4. Frontmatter parsing is deliberately minimal (key: value
pairs and [a, b] lists, no nesting) to avoid a yaml dependency.

Note: only ATX headers (`#` ... `######`) are recognized; setext headers
(underlined with `===`/`---`) are not.
"""
from __future__ import annotations

import re

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def extract_metadata(text: str) -> dict:
    """Return {'title', 'frontmatter', 'headers'} for `text`.

    title: first H1, falling back to frontmatter 'title'.
    """
    frontmatter, body = _split_frontmatter(text)
    headers: list[dict] = []
    lines = body.splitlines()
    in_fence = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADER_RE.match(line)
        if m:
            headers.append({"level": len(m.group(1)), "text": m.group(2).strip()})
    title = next((h["text"] for h in headers if h["level"] == 1), None)
    if title is None:
        fm_title = frontmatter.get("title")
        title = fm_title if isinstance(fm_title, str) else None
    return {"title": title, "frontmatter": frontmatter, "headers": headers}


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
