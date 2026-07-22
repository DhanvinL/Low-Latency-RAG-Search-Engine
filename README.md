# Enterprise Hybrid RAG & Semantic Search Engine

A production-grade, modular Python service for low-latency document search and
retrieval-augmented generation (RAG). It combines **dense vector retrieval**
(Qdrant + SentenceTransformers) with **sparse keyword filtering**, a
**Redis-backed semantic query cache**, **vLLM** inference, **Pydantic / Great
Expectations** data validation, and **Ragas** automated evaluation — exposed
through a **FastAPI** microservice.

Every network-dependent component (Qdrant, Redis, vLLM, the embedding model,
Ragas) has a transparent in-process fallback, so the entire system is runnable
end-to-end with **zero external infrastructure** for development and testing.

---

## Table of Contents

- [Architecture](#architecture)
- [Project Layout](#project-layout)
- [Request Lifecycle](#request-lifecycle)
- [Local Installation](#local-installation)
- [Running with Docker](#running-with-docker)
- [Environment Variables](#environment-variables)
- [API Documentation](#api-documentation)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Design Notes](#design-notes)

---

## Architecture

```
                         ┌──────────────────────────────────────────┐
                         │             FastAPI (app/main.py)          │
                         │   POST /v1/ingest   POST /v1/query  /health │
                         └───────┬───────────────────────┬────────────┘
                                 │                        │
                    ┌────────────▼─────────┐   ┌──────────▼──────────────┐
                    │   Data Pipeline       │   │      RAG Chain          │
                    │  loader → chunker →   │   │  cache → retrieve →     │
                    │  validator            │   │  format → generate      │
                    └────────────┬──────────┘   └───┬───────┬───────┬─────┘
                                 │                   │       │       │
                    ┌────────────▼──────────┐  ┌─────▼──┐ ┌──▼───┐ ┌─▼──────┐
                    │  Embedding Engine      │  │ Redis  │ │Qdrant│ │ vLLM   │
                    │ (SentenceTransformers) │  │ Cache  │ │Hybrid│ │ Server │
                    └────────────────────────┘  └────────┘ └──────┘ └────────┘
                                 │
                    ┌────────────▼──────────┐
                    │  Evaluation (Ragas)    │
                    └────────────────────────┘
```

**Core capabilities**

| Concern            | Implementation                                                     |
|--------------------|--------------------------------------------------------------------|
| Ingestion          | `pdfplumber`/`PyPDF2` parsing with page + provenance metadata       |
| Chunking           | Recursive character/token splitter — 512-token chunks, 64 overlap   |
| Validation         | Pydantic schema + control-char stripping (+ optional Great Expectations) |
| Embeddings         | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine)          |
| Vector store       | Qdrant collection with dense cosine vectors + payload indexes       |
| Hybrid search      | Dense similarity fused with sparse keyword/payload filtering        |
| Semantic cache     | Redis hash store, cosine lookup, threshold 0.88, target <50ms       |
| Inference          | vLLM OpenAI-compatible client, `temp=0.2, max_tokens=256, top_p=0.95` |
| Evaluation         | Ragas Faithfulness / Answer Relevance / Context Recall → JSON        |
| Serving            | FastAPI + Uvicorn, multi-stage Docker, docker-compose orchestration |

---

## Project Layout

```
.
├── config/
│   ├── settings.py            # Pydantic BaseSettings — all tunables
│   └── database_config.py     # Qdrant/Redis clients + mocks + retry logic
├── data_pipeline/
│   ├── pdf_loader.py          # PDF/text ingestion + metadata extraction
│   ├── text_chunker.py        # Recursive 512/64-token chunking
│   └── validator.py           # Schema validation + sanitisation
├── indexing/
│   ├── embedding_engine.py    # SentenceTransformers dense embeddings
│   └── qdrant_store.py        # Collection setup, upsert, hybrid search
├── cache/
│   └── semantic_cache.py      # Redis semantic query cache
├── inference/
│   ├── vllm_server.py         # vLLM OpenAI-compatible client + SamplingParams
│   └── rag_chain.py           # Unified cache→retrieve→generate pipeline
├── evaluation/
│   └── evaluator.py           # Ragas evaluation runner + JSON export
├── app/
│   ├── schemas.py             # Request/response Pydantic models
│   └── main.py                # FastAPI application + endpoints
├── tests/
│   └── test_pipeline.py       # End-to-end tests against mock backends
├── Dockerfile                 # Multi-stage Python 3.11 build
├── docker-compose.yml         # api_server + qdrant_db + redis_cache (+ vllm)
├── requirements.txt
└── README.md
```

---

## Request Lifecycle

**Ingestion (`POST /v1/ingest`)**

1. `PDFLoader` parses each path into pages with `source_file`, `page_number`, `creation_date`.
2. `RecursiveCharacterTextChunker` splits page text into 512-token / 64-overlap chunks.
3. `DocumentValidator` strips corrupted characters and enforces the schema.
4. `EmbeddingEngine` encodes valid chunks into 384-dim dense vectors.
5. `QdrantStore` bulk-upserts points (vector + payload + extracted keywords).

**Query (`POST /v1/query`)**

1. `SemanticCache` embeds the query and does a cosine lookup; a hit ≥ **0.88** returns immediately (`cached: true`).
2. On a miss, `QdrantStore.hybrid_search` fetches top-k=3 chunks (dense + sparse fusion).
3. `RAGChain` formats a grounded prompt and calls the `VLLMClient`.
4. The response is written back into the semantic cache for future queries.

---

## Local Installation

Requires **Python 3.11+**.

```bash
# 1. Clone and enter the project
cd "Accelerated Semantic Search Engine"

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) create a .env from the template
cp .env.example .env

# 5. Run the API
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for interactive Swagger UI.

> **No infrastructure? No problem.** If Qdrant/Redis/vLLM are unreachable or the
> ML libraries are not installed, the service automatically uses in-process
> mock clients and a deterministic embedding/generation fallback. Functionality
> is preserved; only answer quality changes.

---

## Running with Docker

```bash
# Build and start api_server + Qdrant + Redis
docker compose up --build

# Include the GPU vLLM inference server (needs the NVIDIA container runtime)
docker compose --profile gpu up --build
```

Services:

| Service       | Port(s)        | Purpose                          |
|---------------|----------------|----------------------------------|
| `api_server`  | 8000           | FastAPI application              |
| `qdrant_db`   | 6333 / 6334    | Vector database (HTTP / gRPC)    |
| `redis_cache` | 6379           | Semantic cache backend           |
| `vllm`        | 8001 (profile) | OpenAI-compatible LLM server     |

For a **CUDA** API image, build with:

```bash
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 -t rag-api .
```

---

## Environment Variables

All settings are defined in `config/settings.py` and can be overridden via
environment variables or a `.env` file (case-insensitive).

| Variable                   | Default                                          | Description                          |
|----------------------------|--------------------------------------------------|--------------------------------------|
| `ENVIRONMENT`              | `development`                                    | `development` / `staging` / `production` |
| `API_HOST` / `API_PORT`    | `0.0.0.0` / `8000`                               | Uvicorn bind address                 |
| `LOG_LEVEL`                | `INFO`                                           | Logging level                        |
| `EMBEDDING_MODEL_NAME`     | `sentence-transformers/all-MiniLM-L6-v2`         | Dense embedding model                |
| `VECTOR_DIMENSIONALITY`    | `384`                                            | Embedding vector size                |
| `EMBEDDING_DEVICE`         | `cpu`                                            | `cpu` or `cuda`                      |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333`                          | Qdrant connection                    |
| `QDRANT_COLLECTION_NAME`   | `enterprise_documents`                           | Collection name                      |
| `REDIS_HOST` / `REDIS_PORT`| `localhost` / `6379`                             | Redis connection                     |
| `REDIS_TTL_SECONDS`        | `3600`                                           | Cache entry TTL                      |
| `SIMILARITY_THRESHOLD`     | `0.88`                                           | Semantic cache hit threshold         |
| `RETRIEVAL_TOP_K`          | `3`                                              | Chunks fetched per query             |
| `CHUNK_SIZE_TOKENS`        | `512`                                            | Chunk size                           |
| `CHUNK_OVERLAP_TOKENS`     | `64`                                             | Chunk overlap                        |
| `VLLM_ENDPOINT`            | `http://localhost:8001/v1`                       | OpenAI-compatible vLLM base URL      |
| `VLLM_MODEL`               | `meta-llama/Meta-Llama-3-8B-Instruct`            | Served model                         |
| `LLM_TEMPERATURE`          | `0.2`                                            | Sampling temperature                 |
| `LLM_MAX_TOKENS`           | `256`                                            | Max generated tokens                 |
| `LLM_TOP_P`                | `0.95`                                           | Nucleus sampling                     |

---

## API Documentation

### `POST /v1/ingest`

```bash
curl -X POST http://localhost:8000/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"paths": ["./data/handbook.pdf"], "recreate_collection": false}'
```

```json
{
  "status": "ok",
  "collection": "enterprise_documents",
  "stats": {
    "documents_loaded": 1,
    "chunks_created": 42,
    "chunks_valid": 42,
    "chunks_invalid": 0,
    "points_upserted": 42
  },
  "errors": []
}
```

### `POST /v1/query`

```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the vacation policy?", "top_k": 3, "use_cache": true}'
```

```json
{
  "query": "What is the vacation policy?",
  "answer": "Based on the retrieved context, employees receive 20 days ...",
  "cached": false,
  "cache_score": 0.0,
  "retrieval_count": 3,
  "latency_ms": 128.4,
  "backend": "mock",
  "contexts": [
    {"source_file": "handbook.pdf", "page_number": 12, "score": 0.91, "text": "..."}
  ]
}
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "app_name": "Enterprise Hybrid RAG & Semantic Search Engine",
  "version": "1.0.0",
  "environment": "development",
  "components": [
    {"name": "qdrant", "status": "mock", "detail": "points=42"},
    {"name": "redis", "status": "mock", "detail": "entries=1"},
    {"name": "embedding_engine", "status": "up", "detail": "hash"},
    {"name": "vllm", "status": "up", "detail": "mock"}
  ]
}
```

---

## Evaluation

Run automated Ragas metrics (Faithfulness, Answer Relevance, Context Recall)
over generated samples and export a JSON report:

```python
from evaluation.evaluator import RagasEvaluator, EvaluationSample

evaluator = RagasEvaluator(output_dir="./artifacts/evaluation")
samples = [
    EvaluationSample(
        question="What is the capital of France?",
        answer="The capital of France is Paris.",
        contexts=["Paris is the capital and largest city of France."],
        ground_truth="Paris",
    ),
]
path = evaluator.run_and_export(samples)
print("Metrics written to", path)
```

If `ragas` is installed the real metrics are computed; otherwise a transparent
lexical-overlap approximation is used so the runner always emits scores.

---

## Testing

```bash
pytest -v
```

The suite in `tests/test_pipeline.py` covers chunking, validation, embeddings,
Qdrant upsert + hybrid search, semantic cache hit/miss, the full RAG chain,
vLLM mock generation, evaluation, connector fallbacks/retry, and the FastAPI
endpoints — all against in-process mocks, so **no external services are
required**.

---

## Design Notes

- **Graceful degradation over hard failure.** Every external dependency has a
  mock/fallback path (`MockQdrantClient`, `MockRedisClient`, hash embeddings,
  mock generation, lexical evaluation). The system is always runnable.
- **Single source of configuration.** `config/settings.py` centralises every
  tunable; the rest of the code imports the validated `settings` singleton.
- **Composable RAG.** `RAGChain` mirrors the retriever → prompt → LLM pattern so
  it can be dropped behind LangChain/LlamaIndex abstractions with minimal glue.
- **Deterministic mocks.** Fallback embeddings and generation are deterministic,
  keeping tests stable and static analysis meaningful.
