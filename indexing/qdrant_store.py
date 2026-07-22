"""Qdrant-backed hybrid vector store.

Provides collection provisioning (dense cosine vectors + payload indexes),
bulk upserts, and hybrid search that combines dense vector similarity with
sparse keyword / payload filtering. All operations tolerate a mock client so
the store keeps working when a live Qdrant node is idle.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from config.database_config import get_qdrant_client
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single retrieval hit."""

    id: str
    score: float
    text: str
    payload: Dict[str, Any] = field(default_factory=dict)


class QdrantStore:
    """High-level interface over a Qdrant collection."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        client: Any = None,
    ) -> None:
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.dimension = dimension or settings.vector_dimensionality
        self.client = client or get_qdrant_client()
        self._models = self._import_models()

    @staticmethod
    def _import_models() -> Any:
        """Import ``qdrant_client.models`` if available, else ``None``."""
        try:
            from qdrant_client import models

            return models
        except ImportError:
            return None

    @property
    def is_mock(self) -> bool:
        return self.client.__class__.__name__ == "MockQdrantClient"

    # ------------------------------------------------------------------ #
    # Collection lifecycle
    # ------------------------------------------------------------------ #
    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection (and payload indexes) if it does not exist."""
        try:
            exists = self.client.collection_exists(self.collection_name)
            if exists and not recreate:
                logger.info("Collection '%s' already exists.", self.collection_name)
                return

            if self._models is not None:
                vectors_config = self._models.VectorParams(
                    size=self.dimension,
                    distance=self._models.Distance.COSINE,
                )
                if recreate and exists:
                    self.client.delete_collection(self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                )
            else:  # mock client accepts loose kwargs.
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={"size": self.dimension, "distance": "Cosine"},
                )

            self._create_payload_indexes()
            logger.info("Provisioned collection '%s'.", self.collection_name)
        except Exception as exc:
            logger.error("Failed to ensure collection '%s': %s", self.collection_name, exc)
            raise

    def _create_payload_indexes(self) -> None:
        """Index the payload fields used for keyword / metadata filtering."""
        indexed_fields = {
            "source_file": "keyword",
            "page_number": "integer",
            "keywords": "keyword",
        }
        for field_name, schema in indexed_fields.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception as exc:  # index creation is best-effort.
                logger.debug("Payload index for '%s' skipped: %s", field_name, exc)

    # ------------------------------------------------------------------ #
    # Upsert
    # ------------------------------------------------------------------ #
    def upsert_chunks(
        self,
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[Dict[str, Any]],
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 128,
    ) -> int:
        """Bulk-upsert vectors + payloads. Returns the number of points written."""
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must have equal length.")
        ids = list(ids) if ids is not None else [str(uuid.uuid4()) for _ in vectors]

        total = 0
        for start in range(0, len(vectors), batch_size):
            end = start + batch_size
            batch_points = self._build_points(
                ids[start:end], vectors[start:end], payloads[start:end]
            )
            self.client.upsert(collection_name=self.collection_name, points=batch_points)
            total += len(batch_points)
        logger.info("Upserted %d point(s) into '%s'.", total, self.collection_name)
        return total

    def _build_points(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[Dict[str, Any]],
    ) -> List[Any]:
        """Construct PointStruct objects (or dicts for the mock client)."""
        points: List[Any] = []
        for pid, vector, payload in zip(ids, vectors, payloads):
            enriched = {**payload, "keywords": self._extract_keywords(payload.get("text", ""))}
            if self._models is not None:
                points.append(
                    self._models.PointStruct(
                        id=pid, vector=list(vector), payload=enriched
                    )
                )
            else:
                points.append({"id": pid, "vector": list(vector), "payload": enriched})
        return points

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def hybrid_search(
        self,
        query_vector: Sequence[float],
        query_text: str = "",
        top_k: Optional[int] = None,
        source_file: Optional[str] = None,
    ) -> List[SearchResult]:
        """Combine dense similarity with sparse keyword / payload filtering.

        Dense candidates are retrieved from Qdrant, then re-ranked with a
        lightweight BM25-style keyword overlap boost derived from ``query_text``.
        """
        top_k = top_k or settings.retrieval_top_k
        query_filter = self._build_filter(source_file)

        # Over-fetch so the sparse re-rank has candidates to work with.
        fetch_k = max(top_k * 4, top_k)
        raw_hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=list(query_vector),
            limit=fetch_k,
            query_filter=query_filter,
        )

        keyword_set = self._tokenize(query_text)
        results: List[SearchResult] = []
        for hit in raw_hits:
            payload = getattr(hit, "payload", {}) or {}
            text = payload.get("text", "")
            dense_score = float(getattr(hit, "score", 0.0))
            sparse_score = self._keyword_overlap(keyword_set, text)
            # Weighted fusion of dense + sparse signals.
            fused = 0.75 * dense_score + 0.25 * sparse_score
            results.append(
                SearchResult(
                    id=str(getattr(hit, "id", "")),
                    score=fused,
                    text=text,
                    payload=payload,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        top = results[:top_k]
        logger.info("Hybrid search returned %d/%d candidate(s).", len(top), len(raw_hits))
        return top

    def _build_filter(self, source_file: Optional[str]) -> Any:
        """Build a Qdrant payload filter for optional metadata constraints."""
        if source_file is None:
            return None
        if self._models is not None:
            return self._models.Filter(
                must=[
                    self._models.FieldCondition(
                        key="source_file",
                        match=self._models.MatchValue(value=source_file),
                    )
                ]
            )
        return {"must": [{"key": "source_file", "match": {"value": source_file}}]}

    # ------------------------------------------------------------------ #
    # Keyword helpers (sparse signal)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tokenize(text: str) -> set:
        return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}

    @classmethod
    def _extract_keywords(cls, text: str, limit: int = 20) -> List[str]:
        tokens = cls._tokenize(text)
        return sorted(tokens)[:limit]

    @classmethod
    def _keyword_overlap(cls, query_tokens: set, text: str) -> float:
        if not query_tokens:
            return 0.0
        doc_tokens = cls._tokenize(text)
        if not doc_tokens:
            return 0.0
        return len(query_tokens & doc_tokens) / len(query_tokens)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        """Return collection status/point count for health checks."""
        try:
            info = self.client.get_collection(self.collection_name)
            if isinstance(info, dict):
                return info
            return {
                "status": getattr(info, "status", "unknown"),
                "points_count": getattr(info, "points_count", None),
            }
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}
