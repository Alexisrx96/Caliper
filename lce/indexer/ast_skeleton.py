"""Extract token-lean skeletons from Python source via stdlib ast.

Foundation spec §4: signatures + docstring first-lines, bodies discarded.
"""
from __future__ import annotations


def extract_skeleton(source: str, *, filename: str = "<unknown>") -> str:
    """Return the skeleton of `source`.

    Output: module docstring first line, then each class/function signature
    with its docstring first line, bodies discarded. Raises SyntaxError on
    unparseable source (the caller skips the file, spec §8).
    """
    raise NotImplementedError("phase 2 — foundation spec §4")
