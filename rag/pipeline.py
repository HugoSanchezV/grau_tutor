"""Script de ingesta completa: PGN → embeddings → ChromaDB."""
from __future__ import annotations
import sys
import os

# Permite ejecutar desde la raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import settings
from core.logging import setup_logging, get_logger
from rag.ingest import ingest_all
from rag.embeddings import embed_texts
from rag.store import get_chroma_client, get_or_create_collection, add_documents, collection_is_empty
from rag.bm25 import build_bm25_index

setup_logging()
logger = get_logger(__name__)


def run_ingestion(force: bool = False) -> None:
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    if not collection_is_empty(collection) and not force:
        logger.info(f"La colección ya tiene {collection.count()} documentos. Usa force=True para reingestar.")
        return

    if force and not collection_is_empty(collection):
        logger.warning("Eliminando colección existente para reingestar...")
        client.delete_collection(settings.chroma_collection)
        collection = get_or_create_collection(client)

    logger.info("Iniciando ingesta de partidas de Grau...")
    chunks = ingest_all(settings.data_dir)

    if not chunks:
        logger.error("No se encontraron chunks. Verifica la ruta de los PGN.")
        return

    ids = [c.partida_id for c in chunks]
    documents = [c.texto_completo for c in chunks]
    metadatas = [c.to_chroma_document()["metadata"] for c in chunks]

    logger.info(f"Generando embeddings para {len(documents)} chunks...")
    embeddings = embed_texts(documents)

    logger.info("Cargando en ChromaDB...")
    add_documents(collection, ids, documents, embeddings, metadatas)

    logger.info("Construyendo índice BM25...")
    build_bm25_index(chunks, settings.bm25_index_path)

    logger.info(f"Ingesta completa. Total en ChromaDB: {collection.count()} documentos")


def rebuild_bm25_from_chroma() -> None:
    """Rebuild the BM25 index from what's currently in ChromaDB, without re-embedding."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    total = collection.count()
    if total == 0:
        logger.error("ChromaDB está vacío; ejecuta la ingesta primero.")
        return

    logger.info(f"Reconstruyendo BM25 desde {total} docs en ChromaDB...")
    batch_size = 500
    all_ids: list[str] = []
    all_texts: list[str] = []

    offset = 0
    while offset < total:
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            parts = [
                meta.get("tema", ""),
                meta.get("eco", ""),
                meta.get("white", ""),
                meta.get("black", ""),
                doc,
            ]
            all_ids.append(id_)
            all_texts.append(" ".join(p for p in parts if p))
        offset += batch_size

    from rank_bm25 import BM25Okapi
    from rag.bm25 import _tokenize
    import pickle, os

    tokenized = [_tokenize(t) for t in all_texts]
    bm25 = BM25Okapi(tokenized)
    parent = os.path.dirname(settings.bm25_index_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(settings.bm25_index_path, "wb") as f:
        pickle.dump({"ids": all_ids, "bm25": bm25}, f)
    logger.info(f"Índice BM25 reconstruido: {len(all_ids)} docs → {settings.bm25_index_path}")


if __name__ == "__main__":
    if "--rebuild-bm25" in sys.argv:
        rebuild_bm25_from_chroma()
    else:
        force = "--force" in sys.argv
        run_ingestion(force=force)
