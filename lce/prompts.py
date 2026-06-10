"""ChatML prompt construction — identical task framing across arms.

Spec §6: the system prompt (task + schema + format demand) is the same for
every arm; only the CONTEXT representation differs. Pure functions, no
model dependency.
"""
from __future__ import annotations

from lce.retriever import RetrievedDoc

SYSTEM_PROMPT = (
    "You are a code-navigation router. Given CONTEXT about a codebase and a "
    "user query, respond ONLY with a JSON object of the form "
    '{"action": "open_file" | "search_code" | "explain" | "none", '
    '"target": "<file path, symbol, or search string>", '
    '"confidence": <number between 0 and 1>}. '
    "No prose, no code fences, no explanations."
)


def build_prompt(query: str, docs: list[RetrievedDoc], mode: str) -> str:
    """Render the Qwen ChatML prompt for `mode` ('naive' | 'lean')."""
    if mode == "naive":
        context = "\n\n".join(doc.text for doc in docs)
    elif mode == "lean":
        context = "\n\n".join(f"[{doc.doc_id}]\n{doc.text}" for doc in docs)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'naive' or 'lean'")
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
