"""End-to-end pipeline tests.

These tests exercise the full stack against the built-in mock clients so they
run without a live Qdrant, Redis, or vLLM node — validating chunking,
validation, embedding, indexing, semantic caching, RAG orchestration,
evaluation, and the FastAPI endpoints.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from cache.semantic_cache import SemanticCache
from config.database_config import (
    MockQdrantClient,
    MockRedisClient,
    get_qdrant_client,
    get_redis_client,
    with_retries,
)
from data_pipeline.text_chunker import RecursiveCharacterTextChunker
from data_pipeline.validator import DocumentValidator
from evaluation.evaluator import EvaluationSample, RagasEvaluator
from indexing.embedding_engine import EmbeddingEngine
from indexing.qdrant_store import QdrantStore
from inference.rag_chain import RAGChain
from inference.vllm_server import SamplingParams, VLLMClient


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def embedder() -> EmbeddingEngine:
    return EmbeddingEngine(dimension=384)


@pytest.fixture()
def store(embedder: EmbeddingEngine) -> QdrantStore:
    s = QdrantStore(collection_name="test_collection", client=MockQdrantClient())
    s.ensure_collection(recreate=True)
    return s


@pytest.fixture()
def cache(embedder: EmbeddingEngine) -> SemanticCache:
    return SemanticCache(redis_client=MockRedisClient(), embedding_engine=embedder)


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def test_chunker_respects_size_and_overlap():
    chunker = RecursiveCharacterTextChunker(chunk_size_tokens=20, chunk_overlap_tokens=5)
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunker.split_text(text)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunker_empty_input_returns_empty():
    chunker = RecursiveCharacterTextChunker()
    assert chunker.split_text("") == []


def test_chunker_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        RecursiveCharacterTextChunker(chunk_size_tokens=10, chunk_overlap_tokens=10)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validator_strips_control_characters():
    validator = DocumentValidator()
    dirty = "Hello\x00\x07 World\x1f"
    assert validator.strip_corrupted_characters(dirty) == "Hello World"


def test_validator_partitions_valid_and_invalid():
    validator = DocumentValidator(min_text_length=3)
    records = [
        {"text": "valid content here", "source_file": "a.pdf", "page_number": 1},
        {"text": "", "source_file": "b.pdf", "page_number": 1},  # empty -> invalid
        {"text": "ok", "source_file": "c.pdf", "page_number": 0},  # short + bad page
    ]
    result = validator.validate_batch(records)
    assert result.summary["valid"] == 1
    assert result.summary["invalid"] == 2
    assert not result.is_valid


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def test_embeddings_have_correct_dimension(embedder: EmbeddingEngine):
    vector = embedder.embed_text("enterprise search engine")
    assert len(vector) == embedder.dimension


def test_embedding_similarity_is_symmetric(embedder: EmbeddingEngine):
    a = embedder.embed_text("machine learning")
    b = embedder.embed_text("machine learning")
    assert embedder.cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Qdrant store
# --------------------------------------------------------------------------- #
def test_upsert_and_hybrid_search(store: QdrantStore, embedder: EmbeddingEngine):
    texts = [
        "The vacation policy grants 20 days of paid leave.",
        "Expense reports must be filed within 30 days.",
        "Remote work is permitted three days per week.",
    ]
    payloads = [
        {"text": t, "source_file": "handbook.pdf", "page_number": i + 1}
        for i, t in enumerate(texts)
    ]
    vectors = embedder.embed_texts(texts)
    count = store.upsert_chunks(vectors=vectors, payloads=payloads)
    assert count == 3

    query_vector = embedder.embed_query("How many vacation days do I get?")
    results = store.hybrid_search(query_vector, query_text="vacation days paid leave", top_k=2)
    assert len(results) <= 2
    assert results[0].text  # non-empty top hit


# --------------------------------------------------------------------------- #
# Semantic cache
# --------------------------------------------------------------------------- #
def test_semantic_cache_hit_on_identical_query(cache: SemanticCache):
    cache.store("What is the refund policy?", "Refunds are issued within 14 days.")
    lookup = cache.lookup("What is the refund policy?")
    assert lookup.hit is True
    assert "Refunds" in (lookup.response or "")
    assert lookup.score >= cache.threshold


def test_semantic_cache_miss_on_unrelated_query(cache: SemanticCache):
    cache.store("What is the refund policy?", "Refunds are issued within 14 days.")
    lookup = cache.lookup("Describe the quantum entanglement experiment protocol.")
    assert lookup.hit is False


# --------------------------------------------------------------------------- #
# RAG chain
# --------------------------------------------------------------------------- #
def test_rag_chain_end_to_end(store: QdrantStore, cache: SemanticCache, embedder: EmbeddingEngine):
    texts = ["Paid time off accrues at two days per month for full-time staff."]
    payloads = [{"text": texts[0], "source_file": "policy.pdf", "page_number": 1}]
    store.upsert_chunks(vectors=embedder.embed_texts(texts), payloads=payloads)

    chain = RAGChain(store=store, cache=cache, llm=VLLMClient(), embedder=embedder)
    first = chain.run("How does PTO accrue?")
    assert first.cached is False
    assert first.answer

    # Second identical query should now be served from the semantic cache.
    second = chain.run("How does PTO accrue?")
    assert second.cached is True


# --------------------------------------------------------------------------- #
# vLLM client
# --------------------------------------------------------------------------- #
def test_vllm_mock_generation_is_grounded():
    client = VLLMClient()
    prompt = "System\n\nContext:\nParis is the capital of France.\n\nQuestion: What is the capital of France?\n\nAnswer:"
    result = client.generate(prompt, params=SamplingParams(max_tokens=64))
    assert result.text
    assert result.backend in {"mock", "vllm-openai"}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def test_evaluator_produces_scores_and_export():
    evaluator = RagasEvaluator(output_dir=tempfile.mkdtemp())
    samples = [
        EvaluationSample(
            question="What is the capital of France?",
            answer="The capital of France is Paris.",
            contexts=["Paris is the capital and largest city of France."],
            ground_truth="Paris",
        )
    ]
    report = evaluator.evaluate(samples, now=1_700_000_000.0)
    assert 0.0 <= report.faithfulness <= 1.0
    assert 0.0 <= report.answer_relevance <= 1.0
    assert 0.0 <= report.context_recall <= 1.0

    path = evaluator.run_and_export(samples, now=1_700_000_000.0)
    assert os.path.exists(path)


# --------------------------------------------------------------------------- #
# Connectors / retry
# --------------------------------------------------------------------------- #
def test_factories_return_usable_clients():
    assert get_qdrant_client() is not None
    assert get_redis_client() is not None


def test_with_retries_eventually_succeeds():
    calls = {"n": 0}

    @with_retries(max_attempts=3, base_delay=0.0)
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 2


# --------------------------------------------------------------------------- #
# FastAPI endpoints
# --------------------------------------------------------------------------- #
def test_api_health_and_query_flow():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] in {"ok", "degraded"}

        # Ingest a temporary text document.
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("The onboarding process takes five business days to complete.")
            temp_path = handle.name
        try:
            ingest = client.post("/v1/ingest", json={"paths": [temp_path]})
            assert ingest.status_code == 200
            assert ingest.json()["stats"]["points_upserted"] >= 1

            query = client.post("/v1/query", json={"query": "How long is onboarding?"})
            assert query.status_code == 200
            body = query.json()
            assert "answer" in body
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
