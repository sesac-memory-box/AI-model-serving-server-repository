import asyncio
from types import SimpleNamespace

from services import retrieval


class RaisingQueryClient:
    def __init__(self, error: Exception):
        self.error = error

    async def query_points(self, **kwargs):
        raise self.error


def _patch_retrieval_dependencies(monkeypatch, client):
    monkeypatch.setenv("QDRANT_URL", "http://localhost:9999")
    monkeypatch.setenv("QDRANT_COLLECTION", "missing_collection")
    monkeypatch.setattr(retrieval, "_embed_query", lambda query: [0.1, 0.2, 0.3])
    monkeypatch.setattr(retrieval, "_qdrant_client", lambda: client)


def test_retrieve_returns_empty_list_on_qdrant_connection_error(monkeypatch):
    _patch_retrieval_dependencies(
        monkeypatch,
        RaisingQueryClient(ConnectionError("All connection attempts failed")),
    )

    documents = asyncio.run(retrieval.retrieve("서울역"))

    assert documents == []


def test_retrieve_returns_empty_list_on_qdrant_timeout(monkeypatch):
    _patch_retrieval_dependencies(
        monkeypatch,
        RaisingQueryClient(TimeoutError("Qdrant request timed out")),
    )

    documents = asyncio.run(retrieval.retrieve("서울역"))

    assert documents == []


def test_retrieve_returns_empty_list_when_collection_is_missing(monkeypatch):
    _patch_retrieval_dependencies(
        monkeypatch,
        RaisingQueryClient(RuntimeError("Collection not found")),
    )

    documents = asyncio.run(retrieval.retrieve("서울역"))

    assert documents == []


def test_retrieve_returns_empty_list_when_qdrant_url_is_missing(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)

    def fail_embedding(query):
        raise AssertionError("Embedding must not run without QDRANT_URL")

    monkeypatch.setattr(retrieval, "_embed_query", fail_embedding)

    documents = asyncio.run(retrieval.retrieve("서울역"))

    assert documents == []


def test_retrieve_skips_payload_parsing_failures(monkeypatch):
    class PayloadErrorPoint:
        @property
        def payload(self):
            raise ValueError("payload parsing failed")

    class MixedPayloadClient:
        async def query_points(self, **kwargs):
            return SimpleNamespace(
                points=[
                    PayloadErrorPoint(),
                    SimpleNamespace(payload={"content": "정상 자료"}, score=0.9),
                    SimpleNamespace(payload="not a dict", score=0.95),
                ]
            )

    _patch_retrieval_dependencies(monkeypatch, MixedPayloadClient())

    documents = asyncio.run(retrieval.retrieve("서울역"))

    assert len(documents) == 1
    assert documents[0].payload["content"] == "정상 자료"


def test_retrieve_keeps_score_066_when_score_threshold_is_unset(monkeypatch):
    class LowScoreClient:
        async def query_points(self, **kwargs):
            return SimpleNamespace(
                points=[
                    SimpleNamespace(payload={"content": "남대문 시장 자료"}, score=0.66),
                ]
            )

    _patch_retrieval_dependencies(monkeypatch, LowScoreClient())
    monkeypatch.delenv("RAG_SCORE_THRESHOLD", raising=False)

    documents = asyncio.run(retrieval.retrieve("남대문 시장"))

    assert len(documents) == 1
    assert documents[0].payload["content"] == "남대문 시장 자료"
