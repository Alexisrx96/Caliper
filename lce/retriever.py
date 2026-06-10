"""ChromaDB store with parallel raw/skeleton collections (CPU embeddings).

Foundation spec §4: both collections are built from identical source
material so naive vs lean comparisons are fair. Embeddings stay on CPU to
reserve all VRAM for the SLM (spec §2). Metadata values are flattened to
Chroma-safe scalars (str/int/float/bool); lists and dicts are JSON-encoded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

RAW_COLLECTION = "lce_raw"
SKELETON_COLLECTION = "lce_skeleton"
_MODE_TO_COLLECTION = {"naive": RAW_COLLECTION, "lean": SKELETON_COLLECTION}
_CHUNK_CHARS = 1500


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    metadata: dict
    distance: float


class Retriever:
    def __init__(self, persist_dir: str | Path = ".chroma") -> None:
        import chromadb  # deferred: heavy import
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=Settings(anonymized_telemetry=False)
        )
        self._collections = {
            RAW_COLLECTION: self._client.get_or_create_collection(RAW_COLLECTION),
            SKELETON_COLLECTION: self._client.get_or_create_collection(
                SKELETON_COLLECTION
            ),
        }

    def index_document(
        self,
        *,
        doc_id: str,
        raw: str,
        skeleton: str,
        metadata: dict | None = None,
    ) -> None:
        """Store `raw` (chunked) and `skeleton` (whole) under `doc_id`."""
        meta = _flatten(metadata or {}) | {"source_id": doc_id}
        chunks = _chunk(raw)
        self._collections[RAW_COLLECTION].upsert(
            ids=[f"{doc_id}#{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[dict(meta) for _ in chunks],
        )
        self._collections[SKELETON_COLLECTION].upsert(
            ids=[doc_id], documents=[skeleton], metadatas=[dict(meta)]
        )

    def query(self, text: str, *, mode: str = "lean", k: int = 5) -> list[RetrievedDoc]:
        """Top-k documents from the collection matching `mode` (naive|lean)."""
        collection = self._collection(mode)
        if collection.count() == 0:
            return []
        res = collection.query(query_texts=[text], n_results=min(k, collection.count()))
        return [
            RetrievedDoc(doc_id=i, text=d, metadata=m or {}, distance=dist)
            for i, d, m, dist in zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            )
        ]

    def count(self, mode: str = "lean") -> int:
        """Number of stored items in the collection matching `mode`."""
        return self._collection(mode).count()

    def _collection(self, mode: str):
        try:
            return self._collections[_MODE_TO_COLLECTION[mode]]
        except KeyError:
            raise ValueError(
                f"unknown mode {mode!r}; expected one of {sorted(_MODE_TO_COLLECTION)}"
            ) from None


def _flatten(metadata: dict) -> dict:
    flat: dict = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = json.dumps(value, ensure_ascii=False)
    return flat


def _chunk(text: str, limit: int = _CHUNK_CHARS) -> list[str]:
    """Split on blank-line boundaries into ~limit-char chunks.

    A single paragraph longer than `limit` stays one oversized chunk.
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > limit and current:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
