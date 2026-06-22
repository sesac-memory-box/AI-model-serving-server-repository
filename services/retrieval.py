import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

CONTENT_KEYS = ("content", "content_text", "text", "chunk", "body")
logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    payload: dict[str, Any]
    score: float | None = None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return float(value)


@lru_cache(maxsize=1)
def _embedding_model():
    from fastembed import TextEmbedding

    model_name = os.getenv(
        "FASTEMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    return TextEmbedding(model_name=model_name)


@lru_cache(maxsize=1)
def _qdrant_client():
    from qdrant_client import AsyncQdrantClient

    api_key = os.getenv("QDRANT_API_KEY") or None
    return AsyncQdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=api_key,
    )


def _embed_query(query: str) -> list[float]:
    embeddings = list(_embedding_model().embed([query]))
    if not embeddings:
        return []
    return embeddings[0].tolist()


async def retrieve(query: str, top_k: int | None = None) -> list[RetrievedDocument]:
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        logger.warning("RAG fallback reason: qdrant_url_missing")
        return []

    try:
        logger.info("Retrieval query length=%s preview=%s", len(query), query[:200])
        vector = _embed_query(query)
        if not vector:
            logger.warning("RAG fallback reason: empty_embedding")
            return []

        limit = top_k or _env_int("RAG_TOP_K", 5)
        score_threshold = _env_optional_float("RAG_SCORE_THRESHOLD")
        collection = os.getenv("QDRANT_COLLECTION", "memory_box_contents")
        client = _qdrant_client()

        if hasattr(client, "query_points"):
            result = await client.query_points(
                collection_name=collection,
                query=vector,
                limit=limit,
                with_payload=True,
            )
            points = result.points
        else:
            points = await client.search(
                collection_name=collection,
                query_vector=vector,
                limit=limit,
                with_payload=True,
            )

        logger.info("Qdrant raw results: %s", len(points))
        if score_threshold is None:
            logger.info("RAG score threshold: disabled")
        else:
            logger.info("RAG score threshold: %s", score_threshold)

        return points_to_documents(points, score_threshold)
    except Exception as error:
        logger.warning(
            "RAG fallback reason: qdrant_unavailable error=%s: %s",
            type(error).__name__,
            error,
        )
        return []


def points_to_documents(
    points: list[Any],
    score_threshold: float | None = None,
) -> list[RetrievedDocument]:
    documents = []
    for point in points:
        try:
            payload = getattr(point, "payload", None) or {}
            score = getattr(point, "score", None)
            if not isinstance(payload, dict):
                continue
            if not payload:
                continue
            if score_threshold is not None and score is not None and score < score_threshold:
                continue
            documents.append(RetrievedDocument(payload=payload, score=score))
        except Exception as error:
            logger.warning(
                "Failed to parse Qdrant payload. Skipping point. error=%s: %s",
                type(error).__name__,
                error,
            )

    return documents


def extract_content(payload: dict[str, Any]) -> Any:
    for key in CONTENT_KEYS:
        content = payload.get(key)
        if content:
            return content
    return None


def build_context(documents: list[RetrievedDocument]) -> str:
    sections = []
    context_index = 1
    for document in documents:
        payload = document.payload
        content = extract_content(payload)
        if not content:
            continue

        lines = [f"[자료 {context_index}]"]

        title = payload.get("title")
        source = payload.get("source")
        url = payload.get("url")
        year = payload.get("year")
        era = payload.get("era")
        category = payload.get("category")

        if title:
            lines.append(f"제목: {title}")
        if source:
            lines.append(f"출처: {source}")
        if url:
            lines.append(f"URL: {url}")
        if year:
            lines.append(f"연도: {year}")
        if era:
            lines.append(f"시대: {era}")
        if category:
            lines.append(f"분류: {category}")
        if content:
            lines.append(f"내용: {content}")

        sections.append("\n".join(lines))
        context_index += 1

    return "\n\n".join(sections)
