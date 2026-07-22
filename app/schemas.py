"""Pydantic request/response models for the FastAPI service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentIngestRequest(BaseModel):
    """Request body for the ingestion endpoint."""

    paths: List[str] = Field(
        ..., min_length=1, description="Filesystem paths of documents to ingest."
    )
    recreate_collection: bool = Field(
        default=False,
        description="Drop and recreate the Qdrant collection before ingesting.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"paths": ["./data/handbook.pdf"], "recreate_collection": False}
        }
    }


class IngestStats(BaseModel):
    """Per-run ingestion statistics."""

    documents_loaded: int
    chunks_created: int
    chunks_valid: int
    chunks_invalid: int
    points_upserted: int


class DocumentIngestResponse(BaseModel):
    """Response body for the ingestion endpoint."""

    status: str = Field(default="ok")
    collection: str
    stats: IngestStats
    errors: List[Dict[str, Any]] = Field(default_factory=list)


class QueryRequest(BaseModel):
    """Request body for the query endpoint."""

    query: str = Field(..., min_length=1, description="Natural-language user query.")
    top_k: Optional[int] = Field(
        default=None, ge=1, le=20, description="Override for retrieval fan-out."
    )
    source_file: Optional[str] = Field(
        default=None, description="Restrict retrieval to a single source file."
    )
    use_cache: bool = Field(default=True, description="Enable the semantic cache.")

    model_config = {
        "json_schema_extra": {
            "example": {"query": "What is the PTO policy?", "top_k": 3, "use_cache": True}
        }
    }


class RetrievedContextModel(BaseModel):
    """A retrieved context chunk returned to the client."""

    source_file: str
    page_number: Optional[int] = None
    score: float
    text: str


class QueryResponse(BaseModel):
    """Response body for the query endpoint."""

    query: str
    answer: str
    cached: bool
    cache_score: float = 0.0
    retrieval_count: int = 0
    latency_ms: float = 0.0
    backend: str = ""
    contexts: List[RetrievedContextModel] = Field(default_factory=list)


class ComponentHealth(BaseModel):
    """Health status of a single backing component."""

    name: str
    status: str
    detail: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Response body for the health endpoint."""

    status: str
    app_name: str
    version: str
    environment: str
    components: List[ComponentHealth] = Field(default_factory=list)
