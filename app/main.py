"""FastAPI application exposing the RAG & semantic search microservice.

Endpoints:
    POST /v1/ingest  - validate, chunk, embed, and upsert documents to Qdrant.
    POST /v1/query   - cache-aware RAG retrieval + LLM generation.
    GET  /health     - system + component health.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    ComponentHealth,
    DocumentIngestRequest,
    DocumentIngestResponse,
    HealthCheckResponse,
    IngestStats,
    QueryRequest,
    QueryResponse,
    RetrievedContextModel,
)
from config.settings import settings
from data_pipeline.pdf_loader import PDFLoader
from data_pipeline.text_chunker import RecursiveCharacterTextChunker
from data_pipeline.validator import DocumentValidator
from indexing.embedding_engine import get_embedding_engine
from indexing.qdrant_store import QdrantStore
from inference.rag_chain import RAGChain

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("app.main")

# Shared, lazily-populated component registry.
_components: Dict[str, Any] = {}


def _get_components() -> Dict[str, Any]:
    """Lazily construct and cache heavyweight singletons."""
    if not _components:
        embedder = get_embedding_engine()
        store = QdrantStore()
        store.ensure_collection()
        _components["embedder"] = embedder
        _components["store"] = store
        _components["loader"] = PDFLoader()
        _components["chunker"] = RecursiveCharacterTextChunker()
        _components["validator"] = DocumentValidator()
        _components["rag_chain"] = RAGChain(store=store, embedder=embedder)
    return _components


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up components on startup; release them on shutdown."""
    logger.info("Starting %s v%s.", settings.app_name, settings.app_version)
    _get_components()
    yield
    logger.info("Shutting down.")
    _components.clear()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise Hybrid RAG & Semantic Search Engine",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
@app.post("/v1/ingest", response_model=DocumentIngestResponse, tags=["ingest"])
def ingest(request: DocumentIngestRequest) -> DocumentIngestResponse:
    """Validate, chunk, embed, and upsert documents into Qdrant."""
    comp = _get_components()
    loader: PDFLoader = comp["loader"]
    chunker: RecursiveCharacterTextChunker = comp["chunker"]
    validator: DocumentValidator = comp["validator"]
    embedder = comp["embedder"]
    store: QdrantStore = comp["store"]

    if request.recreate_collection:
        store.ensure_collection(recreate=True)

    documents = loader.load_many(request.paths)
    if not documents:
        raise HTTPException(status_code=400, detail="No documents could be loaded.")

    raw_records: List[Dict[str, Any]] = []
    for doc in documents:
        for page in doc.pages:
            for chunk in chunker.split_document(
                page.text, metadata=page.metadata, id_prefix=doc.source_file
            ):
                raw_records.append({**chunk.metadata, "text": chunk.text})

    validation = validator.validate_batch(raw_records)
    valid = validation.valid_records

    points_upserted = 0
    if valid:
        vectors = embedder.embed_texts([r["text"] for r in valid])
        points_upserted = store.upsert_chunks(vectors=vectors, payloads=valid)

    stats = IngestStats(
        documents_loaded=len(documents),
        chunks_created=len(raw_records),
        chunks_valid=len(valid),
        chunks_invalid=len(validation.errors),
        points_upserted=points_upserted,
    )
    logger.info("Ingest complete: %s", stats.model_dump())
    return DocumentIngestResponse(
        collection=store.collection_name,
        stats=stats,
        errors=validation.errors,
    )


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #
@app.post("/v1/query", response_model=QueryResponse, tags=["query"])
def query(request: QueryRequest) -> QueryResponse:
    """Run a cache-aware RAG query and return a grounded answer."""
    comp = _get_components()
    chain: RAGChain = comp["rag_chain"]
    chain.use_cache = request.use_cache
    if request.top_k:
        chain.top_k = request.top_k

    start = time.perf_counter()
    try:
        result = chain.run(request.query, source_file=request.source_file)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # unexpected downstream failure.
        logger.exception("Query pipeline failed.")
        raise HTTPException(status_code=500, detail="Internal query failure.") from exc

    logger.info(
        "Query served (cached=%s) in %.2fms.",
        result.cached,
        (time.perf_counter() - start) * 1000.0,
    )
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        cached=result.cached,
        cache_score=result.cache_score,
        retrieval_count=result.retrieval_count,
        latency_ms=result.latency_ms,
        backend=result.backend,
        contexts=[
            RetrievedContextModel(
                source_file=c.source_file,
                page_number=c.page_number,
                score=c.score,
                text=c.text,
            )
            for c in result.contexts
        ],
    )


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthCheckResponse, tags=["health"])
def health() -> HealthCheckResponse:
    """Report service and backing-component health."""
    comp = _get_components()
    store: QdrantStore = comp["store"]
    rag_chain: RAGChain = comp["rag_chain"]

    components: List[ComponentHealth] = []

    # Qdrant.
    qdrant_stats = store.stats()
    components.append(
        ComponentHealth(
            name="qdrant",
            status="mock" if store.is_mock else str(qdrant_stats.get("status", "unknown")),
            detail=f"points={qdrant_stats.get('points_count')}",
        )
    )

    # Redis / cache.
    cache = rag_chain.cache
    cache_backend = cache.redis.__class__.__name__
    components.append(
        ComponentHealth(
            name="redis",
            status="mock" if cache_backend == "MockRedisClient" else "up",
            detail=f"entries={cache.size()}",
        )
    )

    # Embedding engine.
    components.append(
        ComponentHealth(name="embedding_engine", status="up", detail=comp["embedder"].backend)
    )

    # vLLM.
    components.append(
        ComponentHealth(name="vllm", status="up", detail=rag_chain.llm.backend)
    )

    overall = "ok" if all(c.status not in {"error"} for c in components) else "degraded"
    return HealthCheckResponse(
        status=overall,
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        components=components,
    )


@app.get("/", tags=["root"])
def root() -> Dict[str, str]:
    """Service banner."""
    return {"service": settings.app_name, "version": settings.app_version, "docs": "/docs"}


if __name__ == "__main__":  # pragma: no cover - manual launch convenience.
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
    )
