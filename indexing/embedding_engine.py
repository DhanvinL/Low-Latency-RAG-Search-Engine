"""Dense embedding service backed by SentenceTransformers / PyTorch.

Wraps a SentenceTransformers model to convert text into fixed-size dense
vectors. The model is loaded lazily on first use so that importing this module
is cheap. When the heavy ML dependencies are unavailable, a deterministic
hashing-based fallback encoder produces vectors of the correct dimensionality,
keeping the whole pipeline runnable for tests and static analysis.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from typing import List, Optional, Sequence, Union

from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Thread-safe, lazily-initialised text embedding engine."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model_name
        self.device = device or settings.embedding_device
        self.dimension = dimension or settings.vector_dimensionality
        self.batch_size = batch_size or settings.embedding_batch_size
        self._model = None
        self._backend = "uninitialised"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #
    def _ensure_model(self) -> None:
        """Load the SentenceTransformers model exactly once."""
        if self._model is not None or self._backend == "hash":
            return
        with self._lock:
            if self._model is not None or self._backend == "hash":
                return
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(
                    "Loading embedding model '%s' on device '%s'.",
                    self.model_name,
                    self.device,
                )
                self._model = SentenceTransformer(self.model_name, device=self.device)
                # Trust the model's true dimensionality if it exposes it.
                model_dim = self._model.get_sentence_embedding_dimension()
                if model_dim:
                    self.dimension = model_dim
                self._backend = "sentence-transformers"
            except Exception as exc:  # missing torch / model download failure.
                logger.warning(
                    "SentenceTransformers unavailable (%s); using hash fallback.", exc
                )
                self._backend = "hash"

    @property
    def backend(self) -> str:
        """Return the active backend name (loads the model if needed)."""
        self._ensure_model()
        return self._backend

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #
    def embed_text(self, text: str) -> List[float]:
        """Embed a single string into a dense vector."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of strings into dense vectors."""
        if not texts:
            return []
        self._ensure_model()
        if self._backend == "sentence-transformers":
            vectors = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [vector.tolist() for vector in vectors]
        return [self._hash_embed(text) for text in texts]

    def embed_query(self, query: str) -> List[float]:
        """Embed a user query (kept distinct for API symmetry / future asymmetry)."""
        return self.embed_text(query)

    # ------------------------------------------------------------------ #
    # Deterministic fallback encoder
    # ------------------------------------------------------------------ #
    def _hash_embed(self, text: str) -> List[float]:
        """Produce a deterministic L2-normalised vector via feature hashing.

        This is *not* a semantic embedding, but it is stable and correctly
        shaped, which lets similarity math and vector stores behave sensibly
        without the ML stack installed.
        """
        vector = [0.0] * self.dimension
        tokens = text.lower().split() or [text.lower()]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            # Spread each token across a few dimensions.
            for i in range(0, len(digest), 4):
                idx = int.from_bytes(digest[i : i + 4], "little") % self.dimension
                sign = 1.0 if digest[i] % 2 == 0 else -1.0
                vector[idx] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    @staticmethod
    def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        """Cosine similarity between two equal-length vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


# Process-wide singleton to avoid reloading the model per request.
_engine: Optional[EmbeddingEngine] = None
_engine_lock = threading.Lock()


def get_embedding_engine() -> EmbeddingEngine:
    """Return a cached, process-wide :class:`EmbeddingEngine`."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = EmbeddingEngine()
    return _engine
