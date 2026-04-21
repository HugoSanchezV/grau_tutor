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

    logger.info(f"Ingesta completa. Total en ChromaDB: {collection.count()} documentos")


if __name__ == "__main__":
    force = "--force" in sys.argv
    run_ingestion(force=force)
