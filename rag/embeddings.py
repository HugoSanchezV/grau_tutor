from __future__ import annotations
from openai import OpenAI
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    client = _get_openai_client()
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        # OpenAI recomienda reemplazar saltos de línea para embeddings
        batch_clean = [t.replace("\n", " ") for t in batch]
        response = client.embeddings.create(
            input=batch_clean,
            model=settings.embedding_model,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        logger.info(f"Embeddings generados: {start + len(batch)}/{len(texts)}")

    return all_embeddings


def embed_query(query: str) -> list[float]:
    client = _get_openai_client()
    response = client.embeddings.create(
        input=query.replace("\n", " "),
        model=settings.embedding_model,
    )
    return response.data[0].embedding
