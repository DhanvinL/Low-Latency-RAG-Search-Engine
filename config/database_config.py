"""Database connection factories for Qdrant and Redis.

This module instantiates live clients when the backing services are reachable
and transparently falls back to in-memory mock clients otherwise. The mocks
implement the subset of the client surface the application depends on, so that
static analysis and local test runs behave identically to a fully wired
deployment.
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from config.settings import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Retry helper
# --------------------------------------------------------------------------- #
def with_retries(
    max_attempts: int = 3,
    base_delay: float = 0.25,
    backoff: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator implementing exponential-backoff retry logic.

    Args:
        max_attempts: Total number of attempts before giving up.
        base_delay: Initial delay (seconds) between attempts.
        backoff: Multiplier applied to the delay after each failure.
        exceptions: Exception types that trigger a retry.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = base_delay
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: PERF203 - explicit retry loop.
                    last_exc = exc
                    logger.warning(
                        "Attempt %d/%d for %s failed: %s",
                        attempt,
                        max_attempts,
                        func.__name__,
                        exc,
                    )
                    if attempt < max_attempts:
                        time.sleep(delay)
                        delay *= backoff
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# Mock clients
# --------------------------------------------------------------------------- #
class MockQdrantClient:
    """In-memory stand-in for :class:`qdrant_client.QdrantClient`.

    Stores points in a per-collection dictionary and performs brute-force
    cosine similarity search so the application remains fully functional when a
    real Qdrant node is unavailable.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._collections: Dict[str, Dict[str, Dict[str, Any]]] = {}
        logger.info("Initialised MockQdrantClient (no live Qdrant node).")

    # -- Collection management -------------------------------------------- #
    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self._collections

    def create_collection(self, collection_name: str, *args: Any, **kwargs: Any) -> bool:
        self._collections.setdefault(collection_name, {})
        logger.info("MockQdrantClient created collection '%s'.", collection_name)
        return True

    def recreate_collection(self, collection_name: str, *args: Any, **kwargs: Any) -> bool:
        self._collections[collection_name] = {}
        return True

    def delete_collection(self, collection_name: str, *args: Any, **kwargs: Any) -> bool:
        self._collections.pop(collection_name, None)
        return True

    def create_payload_index(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def get_collection(self, collection_name: str) -> Dict[str, Any]:
        points = self._collections.get(collection_name, {})
        return {"status": "green", "points_count": len(points)}

    # -- Data operations --------------------------------------------------- #
    def upsert(self, collection_name: str, points: List[Any], *args: Any, **kwargs: Any) -> Dict[str, str]:
        store = self._collections.setdefault(collection_name, {})
        for point in points:
            point_id = getattr(point, "id", None)
            vector = getattr(point, "vector", None)
            payload = getattr(point, "payload", None)
            if isinstance(point, dict):  # tolerate raw dict points.
                point_id = point.get("id", point_id)
                vector = point.get("vector", vector)
                payload = point.get("payload", payload)
            point_id = point_id or str(uuid.uuid4())
            store[str(point_id)] = {"vector": vector or [], "payload": payload or {}}
        return {"status": "completed", "operation_id": str(uuid.uuid4())}

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 3,
        query_filter: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> List["_MockScoredPoint"]:
        store = self._collections.get(collection_name, {})
        scored: List[_MockScoredPoint] = []
        for pid, record in store.items():
            score = _cosine_similarity(query_vector, record["vector"])
            if _passes_filter(record["payload"], query_filter):
                scored.append(_MockScoredPoint(pid, score, record["payload"]))
        scored.sort(key=lambda p: p.score, reverse=True)
        return scored[:limit]

    def close(self) -> None:  # parity with the real client API.
        self._collections.clear()


class _MockScoredPoint:
    """Mimics ``qdrant_client.models.ScoredPoint``."""

    def __init__(self, point_id: str, score: float, payload: Dict[str, Any]) -> None:
        self.id = point_id
        self.score = score
        self.payload = payload

    def __repr__(self) -> str:  # pragma: no cover - debug helper.
        return f"_MockScoredPoint(id={self.id!r}, score={self.score:.4f})"


class MockRedisClient:
    """In-memory stand-in for :class:`redis.Redis`.

    Implements the hash, key/value, TTL, and scan primitives used by the
    semantic cache. Data is process-local and non-persistent.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._store: Dict[str, Any] = {}
        self._hashes: Dict[str, Dict[str, str]] = {}
        logger.info("Initialised MockRedisClient (no live Redis node).")

    def ping(self) -> bool:
        return True

    # -- String ops -------------------------------------------------------- #
    def set(self, name: str, value: Any, ex: Optional[int] = None) -> bool:
        self._store[name] = value
        return True

    def get(self, name: str) -> Any:
        return self._store.get(name)

    # -- Hash ops ---------------------------------------------------------- #
    def hset(self, name: str, mapping: Optional[Dict[str, str]] = None, **kwargs: str) -> int:
        bucket = self._hashes.setdefault(name, {})
        data = dict(mapping or {})
        data.update(kwargs)
        bucket.update(data)
        return len(data)

    def hgetall(self, name: str) -> Dict[str, str]:
        return dict(self._hashes.get(name, {}))

    def expire(self, name: str, seconds: int) -> bool:
        return name in self._store or name in self._hashes

    def keys(self, pattern: str = "*") -> List[str]:
        prefix = pattern.rstrip("*")
        return [k for k in self._hashes if k.startswith(prefix)]

    def scan_iter(self, match: str = "*", count: int = 100) -> Any:
        prefix = match.rstrip("*")
        return [k for k in self._hashes if k.startswith(prefix)]

    def delete(self, *names: str) -> int:
        removed = 0
        for name in names:
            removed += int(self._store.pop(name, None) is not None)
            removed += int(self._hashes.pop(name, None) is not None)
        return removed

    def flushdb(self) -> bool:
        self._store.clear()
        self._hashes.clear()
        return True

    def close(self) -> None:  # parity with the real client API.
        self.flushdb()


# --------------------------------------------------------------------------- #
# Similarity + filter helpers (used by the mock search implementation)
# --------------------------------------------------------------------------- #
def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Return cosine similarity without a hard NumPy dependency."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _passes_filter(payload: Dict[str, Any], query_filter: Any) -> bool:
    """Best-effort payload filter evaluation for the mock client."""
    if query_filter is None:
        return True
    conditions = getattr(query_filter, "must", None)
    if not conditions and isinstance(query_filter, dict):
        conditions = query_filter.get("must")
    if not conditions:
        return True
    for cond in conditions:
        key = getattr(cond, "key", None) or (cond.get("key") if isinstance(cond, dict) else None)
        match = getattr(cond, "match", None) or (cond.get("match") if isinstance(cond, dict) else None)
        expected = getattr(match, "value", None) or (match.get("value") if isinstance(match, dict) else None)
        if key is not None and payload.get(key) != expected:
            return False
    return True


# --------------------------------------------------------------------------- #
# Public factories
# --------------------------------------------------------------------------- #
@with_retries(max_attempts=2, base_delay=0.2)
def _connect_qdrant() -> Any:
    """Attempt a live Qdrant connection, raising on failure."""
    from qdrant_client import QdrantClient  # local import keeps mocks importable.

    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        grpc_port=settings.qdrant_grpc_port,
        prefer_grpc=settings.qdrant_prefer_grpc,
        api_key=settings.qdrant_api_key or None,
        timeout=settings.qdrant_timeout,
    )
    # Force a round-trip so we fail fast if the socket is dead.
    client.get_collections()
    return client


@with_retries(max_attempts=2, base_delay=0.2)
def _connect_redis() -> Any:
    """Attempt a live Redis connection, raising on failure."""
    import redis  # local import keeps mocks importable.

    client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password or None,
        decode_responses=True,
        socket_connect_timeout=2.0,
    )
    client.ping()
    return client


def get_qdrant_client() -> Any:
    """Return a live Qdrant client or a :class:`MockQdrantClient` fallback."""
    try:
        client = _connect_qdrant()
        logger.info("Connected to live Qdrant at %s.", settings.qdrant_url)
        return client
    except Exception as exc:  # broad: any connectivity/import error -> mock.
        logger.warning("Qdrant unreachable (%s); using MockQdrantClient.", exc)
        return MockQdrantClient()


def get_redis_client() -> Any:
    """Return a live Redis client or a :class:`MockRedisClient` fallback."""
    try:
        client = _connect_redis()
        logger.info(
            "Connected to live Redis at %s:%s.", settings.redis_host, settings.redis_port
        )
        return client
    except Exception as exc:  # broad: any connectivity/import error -> mock.
        logger.warning("Redis unreachable (%s); using MockRedisClient.", exc)
        return MockRedisClient()
