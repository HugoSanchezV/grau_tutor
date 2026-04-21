from __future__ import annotations
import chromadb
from chromadb import HttpClient, Collection
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


def get_chroma_client() -> HttpClient:
    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )
    logger.info(f"ChromaDB conectado en {settings.chroma_host}:{settings.chroma_port}")
    return client


def get_or_create_collection(client: HttpClient | None = None) -> Collection:
    if client is None:
        client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"Colección '{settings.chroma_collection}': {collection.count()} documentos")
    return collection


def add_documents(
    collection: Collection,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    batch_size: int = 100,
) -> None:
    total = len(ids)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
        logger.info(f"Batch cargado: {end}/{total}")
    logger.info(f"Total documentos en colección: {collection.count()}")


def query_collection(
    collection: Collection,
    query_embeddings: list[list[float]],
    n_results: int = 5,
    where: dict | None = None,
) -> dict:
    kwargs: dict = {
        "query_embeddings": query_embeddings,
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


def collection_is_empty(collection: Collection) -> bool:
    return collection.count() == 0
