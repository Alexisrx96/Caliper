"""ChromaDB store with parallel raw/skeleton collections (CPU embeddings).

Foundation spec §4: both collections are built from identical source
material so naive vs lean comparisons are fair. Embeddings stay on CPU to
reserve all VRAM for the SLM (spec §2).
"""
from __future__ import annotations

from pathlib import Path

RAW_COLLECTION = "lce_raw"
SKELETON_COLLECTION = "lce_skeleton"


class Retriever:
    def __init__(self, persist_dir: str | Path = ".chroma") -> None:
        raise NotImplementedError("phase 2 — foundation spec §4")

    def index_document(
        self,
        *,
        doc_id: str,
        raw: str,
        skeleton: str,
        metadata: dict | None = None,
    ) -> None:
        """Store `raw` and `skeleton` under `doc_id` in their collections."""
        raise NotImplementedError("phase 2 — foundation spec §4")

    def query(self, text: str, *, mode: str = "lean", k: int = 5) -> list[str]:
        """Top-k documents from the collection matching `mode`."""
        raise NotImplementedError("phase 2 — foundation spec §4")
