"""Redis-backed semantic query cache.

Instead of matching queries by exact string, this cache embeds each incoming
query and compares it (via cosine similarity) against the vectors of previously
cached queries. A hit above the configured threshold (default 0.88) returns the
stored response immediately, targeting sub-50ms latency. Misses are recorded so
subsequent semantically-similar queries can be served from cache.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config.database_config import get_redis_client
from config.settings import settings
from indexing.embedding_engine import EmbeddingEngine, get_embedding_engine

logger = logging.getLogger(__name__)


@dataclass
class CacheLookup:
    """Result of a semantic cache lookup."""

    hit: bool
    response: Optional[str] = None
    score: float = 0.0
    latency_ms: float = 0.0
    matched_query: Optional[str] = None
    entry_id: Optional[str] = None


class SemanticCache:
    """Semantic cache storing query vectors + responses in Redis hashes."""

    def __init__(
        self,
        redis_client: Any = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        threshold: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        self.redis = redis_client or get_redis_client()
        self.embedder = embedding_engine or get_embedding_engine()
        self.threshold = threshold if threshold is not None else settings.similarity_threshold
        self.ttl_seconds = ttl_seconds or settings.redis_ttl_seconds
        self.prefix = prefix or settings.semantic_cache_prefix

    # ------------------------------------------------------------------ #
    # Key helpers
    # ------------------------------------------------------------------ #
    def _entry_key(self, entry_id: str) -> str:
        return f"{self.prefix}:entry:{entry_id}"

    def _scan_pattern(self) -> str:
        return f"{self.prefix}:entry:*"

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #
    def lookup(self, query: str) -> CacheLookup:
        """Return a cache hit if a stored query is similar enough."""
        start = time.perf_counter()
        query_vector = self.embedder.embed_query(query)
        best_score = 0.0
        best_entry: Optional[Dict[str, Any]] = None

        for key in self._iter_entry_keys():
            entry = self._read_entry(key)
            if entry is None:
                continue
            score = self.embedder.cosine_similarity(query_vector, entry["vector"])
            if score > best_score:
                best_score = score
                best_entry = entry

        latency_ms = (time.perf_counter() - start) * 1000.0
        if best_entry is not None and best_score >= self.threshold:
            logger.info(
                "Semantic cache HIT (score=%.4f, %.2fms).", best_score, latency_ms
            )
            return CacheLookup(
                hit=True,
                response=best_entry["response"],
                score=best_score,
                latency_ms=latency_ms,
                matched_query=best_entry.get("query"),
                entry_id=best_entry.get("id"),
            )

        logger.info("Semantic cache MISS (best=%.4f, %.2fms).", best_score, latency_ms)
        return CacheLookup(hit=False, score=best_score, latency_ms=latency_ms)

    # ------------------------------------------------------------------ #
    # Store
    # ------------------------------------------------------------------ #
    def store(
        self,
        query: str,
        response: str,
        query_vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist a query/response pair (with its embedding) into the cache."""
        entry_id = str(uuid.uuid4())
        vector = query_vector or self.embedder.embed_query(query)
        key = self._entry_key(entry_id)
        mapping = {
            "id": entry_id,
            "query": query,
            "response": response,
            "vector": json.dumps(vector),
            "metadata": json.dumps(metadata or {}),
            "created_at": str(int(time.time())),
        }
        try:
            self.redis.hset(key, mapping=mapping)
            self.redis.expire(key, self.ttl_seconds)
            logger.debug("Stored semantic cache entry '%s'.", entry_id)
        except Exception as exc:
            logger.warning("Failed to store cache entry: %s", exc)
        return entry_id

    # ------------------------------------------------------------------ #
    # Internal Redis access
    # ------------------------------------------------------------------ #
    def _iter_entry_keys(self) -> List[str]:
        try:
            keys = self.redis.scan_iter(match=self._scan_pattern())
            return list(keys)
        except Exception as exc:
            logger.debug("scan_iter failed, falling back to keys(): %s", exc)
            try:
                return list(self.redis.keys(self._scan_pattern()))
            except Exception:
                return []

    def _read_entry(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            raw = self.redis.hgetall(key)
        except Exception as exc:
            logger.debug("hgetall failed for '%s': %s", key, exc)
            return None
        if not raw:
            return None
        raw = self._decode(raw)
        try:
            return {
                "id": raw.get("id"),
                "query": raw.get("query", ""),
                "response": raw.get("response", ""),
                "vector": json.loads(raw.get("vector", "[]")),
                "metadata": json.loads(raw.get("metadata", "{}")),
            }
        except (json.JSONDecodeError, TypeError) as exc:
            logger.debug("Corrupt cache entry '%s': %s", key, exc)
            return None

    @staticmethod
    def _decode(mapping: Dict[Any, Any]) -> Dict[str, str]:
        """Normalise bytes keys/values (real Redis) to str."""
        decoded: Dict[str, str] = {}
        for k, v in mapping.items():
            key = k.decode() if isinstance(k, bytes) else str(k)
            val = v.decode() if isinstance(v, bytes) else v
            decoded[key] = val
        return decoded

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #
    def clear(self) -> int:
        """Delete all entries owned by this cache. Returns count removed."""
        keys = self._iter_entry_keys()
        if not keys:
            return 0
        try:
            return int(self.redis.delete(*keys))
        except Exception as exc:
            logger.warning("Failed to clear cache: %s", exc)
            return 0

    def size(self) -> int:
        """Return the number of live cache entries."""
        return len(self._iter_entry_keys())
