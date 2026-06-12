"""ChatML prompt construction — identical task framing across arms.

Spec §6: the system prompt (task + schema + format demand) is the same for
every arm; only the CONTEXT representation differs. Pure functions, no
model dependency.
"""
from __future__ import annotations

from lce.retriever import RetrievedDoc

SYSTEM_PROMPT = (
    "Code-navigation router. Output ONLY "
    '{"action": "open_file" | "search_code" | "explain" | "none", '
    '"target": "<path | symbol | query>", '
    '"confidence": <0-1>}. No prose.'
)


def build_prompt(
    query: str,
    docs: list[RetrievedDoc],
    mode: str,
    doc_cap: int | None = None,
) -> str:
    """Render the Qwen ChatML prompt for `mode` ('naive' | 'lean').

    doc_cap=N truncates each doc to its first N lines before rendering —
    lines, not tokens: deterministic and tokenizer-free (phase-5 spec §4).
    None reproduces the uncapped output byte-for-byte. Validation (N >= 1)
    lives at the CLI boundary.
    """
    texts = [doc.text for doc in docs]
    if doc_cap is not None:
        texts = ["\n".join(t.splitlines()[:doc_cap]) for t in texts]
    if mode == "naive":
        context = "\n\n".join(texts)
    elif mode == "lean":
        context = "\n\n".join(
            f"[{doc.doc_id}]\n{t}" for doc, t in zip(docs, texts)
        )
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'naive' or 'lean'")
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
