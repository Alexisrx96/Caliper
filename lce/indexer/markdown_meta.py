"""Extract metadata from Markdown notes: frontmatter, headers, sections.

Foundation spec §4.
"""
from __future__ import annotations


def extract_metadata(text: str) -> dict:
    """Return {'title', 'frontmatter', 'headers', 'sections'} for `text`."""
    raise NotImplementedError("phase 2 — foundation spec §4")
