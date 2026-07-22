"""End-to-end RAG orchestration.

Unifies the retrieval-augmented generation flow into a single callable pipeline:

    query -> semantic cache lookup -> (miss) hybrid retrieval -> context
    formatting -> vLLM generation -> cache write -> structured response.

The chain is framework-agnostic but mirrors the LangChain/LlamaIndex
"retriever + prompt + llm" composition so it can be swapped in behind those
abstractions with minimal glue.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cache.semantic_cache import SemanticCache
from config.settings import settings
from indexing.embedding_engine import EmbeddingEngine, get_embedding_engine
from indexing.qdrant_store import QdrantStore, SearchResult
from inference.vllm_server import SamplingParams, VLLMClient, get_vllm_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. Answer the user's question using "
    "ONLY the provided context. If the context is insufficient, say so plainly. "
    "Be concise, factual, and cite source files when relevant."
)

PROMPT_TEMPLATE = """{system}

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RetrievedContext:
    """A retrieved chunk rendered for the prompt."""

    text: str
    source_file: str
    page_number: Optional[int]
    score: float


@dataclass
class RAGResponse:
    """Structured output of a RAG query."""

    query: str
    answer: str
    cached: bool
    contexts: List[RetrievedContext] = field(default_factory=list)
    cache_score: float = 0.0
    retrieval_count: int = 0
    latency_ms: float = 0.0
    backend: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "cached": self.cached,
            "cache_score": round(self.cache_score, 4),
            "retrieval_count": self.retrieval_count,
            "latency_ms": round(self.latency_ms, 2),
            "backend": self.backend,
            "contexts": [
                {
                    "source_file": c.source_file,
                    "page_number": c.page_number,
                    "score": round(c.score, 4),
                    "text": c.text,
                }
                for c in self.contexts
            ],
        }


class RAGChain:
    """Composable retrieval-augmented generation pipeline."""

    def __init__(
        self,
        store: Optional[QdrantStore] = None,
        cache: Optional[SemanticCache] = None,
        llm: Optional[VLLMClient] = None,
        embedder: Optional[EmbeddingEngine] = None,
        top_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> None:
        self.embedder = embedder or get_embedding_engine()
        self.store = store or QdrantStore()
        self.cache = cache or SemanticCache(embedding_engine=self.embedder)
        self.llm = llm or get_vllm_client()
        self.top_k = top_k or settings.retrieval_top_k
        self.use_cache = use_cache

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        query: str,
        source_file: Optional[str] = None,
        params: Optional[SamplingParams] = None,
    ) -> RAGResponse:
        """Execute the full RAG pipeline for a single query."""
        start = time.perf_counter()
        query = (query or "").strip()
        if not query:
            raise ValueError("Query must be a non-empty string.")

        # 1. Semantic cache verification.
        if self.use_cache:
            lookup = self.cache.lookup(query)
            if lookup.hit and lookup.response is not None:
                latency_ms = (time.perf_counter() - start) * 1000.0
                logger.info("Served query from semantic cache in %.2fms.", latency_ms)
                return RAGResponse(
                    query=query,
                    answer=lookup.response,
                    cached=True,
                    cache_score=lookup.score,
                    latency_ms=latency_ms,
                    backend="cache",
                )

        # 2. Hybrid retrieval.
        query_vector = self.embedder.embed_query(query)
        hits = self.store.hybrid_search(
            query_vector=query_vector,
            query_text=query,
            top_k=self.top_k,
            source_file=source_file,
        )
        contexts = self._to_contexts(hits)

        # 3. Prompt construction + generation.
        prompt = self._build_prompt(query, contexts)
        generation = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, params=params)
        answer = generation.text

        # 4. Cache write-back.
        if self.use_cache:
            self.cache.store(
                query=query,
                response=answer,
                query_vector=query_vector,
                metadata={"source_file": source_file, "backend": generation.backend},
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        return RAGResponse(
            query=query,
            answer=answer,
            cached=False,
            contexts=contexts,
            retrieval_count=len(contexts),
            latency_ms=latency_ms,
            backend=generation.backend,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_contexts(hits: List[SearchResult]) -> List[RetrievedContext]:
        contexts: List[RetrievedContext] = []
        for hit in hits:
            payload = hit.payload or {}
            contexts.append(
                RetrievedContext(
                    text=hit.text or payload.get("text", ""),
                    source_file=payload.get("source_file", "unknown"),
                    page_number=payload.get("page_number"),
                    score=hit.score,
                )
            )
        return contexts

    @staticmethod
    def _build_prompt(query: str, contexts: List[RetrievedContext]) -> str:
        if contexts:
            blocks = []
            for i, ctx in enumerate(contexts, start=1):
                citation = f"[{i}] {ctx.source_file}"
                if ctx.page_number is not None:
                    citation += f" (p.{ctx.page_number})"
                blocks.append(f"{citation}\n{ctx.text}")
            context_str = "\n\n".join(blocks)
        else:
            context_str = "(no relevant context retrieved)"
        return PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT, context=context_str, question=query
        )
