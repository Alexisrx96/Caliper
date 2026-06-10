"""Extract token-lean skeletons from Python source via stdlib ast.

Foundation spec §4: signatures + docstring first-lines, bodies discarded.
"""
from __future__ import annotations

import ast
import copy

_INDENT = "    "


def extract_skeleton(source: str, *, filename: str = "<unknown>") -> str:
    """Return the skeleton of `source`.

    Output: module docstring first line, then each class/function signature
    (decorators omitted) with its docstring first line, indented to reflect
    nesting. Bodies are discarded. Raises SyntaxError on unparseable source
    (the caller skips the file, spec §8).
    """
    tree = ast.parse(source, filename=filename)
    lines: list[str] = []
    _append_docstring(tree, lines, depth=0)
    _walk(tree.body, lines, depth=0)
    return "\n".join(lines)


def _walk(body: list[ast.stmt], lines: list[str], *, depth: int) -> None:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lines.append(_INDENT * depth + _signature_line(node))
            _append_docstring(node, lines, depth=depth + 1)
            _walk(node.body, lines, depth=depth + 1)


def _signature_line(node: ast.AST) -> str:
    # Clone with the body replaced by `pass`, unparse, keep the def/class
    # line — ast.unparse handles every arg form (pos-only, kw-only, *args,
    # defaults, annotations) so we never hand-format signatures.
    clone = copy.deepcopy(node)
    clone.body = [ast.Pass()]
    clone.decorator_list = []
    return ast.unparse(clone).splitlines()[0]


def _append_docstring(node: ast.AST, lines: list[str], *, depth: int) -> None:
    doc = ast.get_docstring(node)
    if doc and doc.strip():
        first = doc.strip().splitlines()[0]
        lines.append(_INDENT * depth + f'"""{first}"""')
