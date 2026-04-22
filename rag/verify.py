"""Verificación rápida del stack RAG: ChromaDB → embeddings → retrieval."""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import settings
from core.logging import setup_logging, get_logger
from rag.embeddings import embed_query
from rag.store import get_chroma_client, get_or_create_collection, query_collection

setup_logging()
logger = get_logger(__name__)

QUERY = input("¿Qué quieres saber? ")


def check_chroma() -> bool:
    print("\n=== 1. Conexión ChromaDB ===")
    try:
        client = get_chroma_client()
        client.heartbeat()
        print(f"  OK  Conectado a {settings.chroma_host}:{settings.chroma_port}")
        return True
    except Exception as e:
        print(f"  FAIL  {e}")
        return False


def check_collection() -> int:
    print("\n=== 2. Colección y documentos ===")
    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        count = collection.count()
        print(f"  OK  Colección '{settings.chroma_collection}': {count} documentos")
        if count == 0:
            print("  WARN  La colección está vacía — corre: python rag/pipeline.py")
        return count
    except Exception as e:
        print(f"  FAIL  {e}")
        return 0


def check_embeddings() -> list[float] | None:
    print("\n=== 3. Embeddings (OpenAI) ===")
    try:
        emb = embed_query(QUERY)
        print(f"  OK  Embedding generado — dimensiones: {len(emb)}")
        print(f"       Primeros 5 valores: {[round(v, 4) for v in emb[:5]]}")
        return emb
    except Exception as e:
        print(f"  FAIL  {e}")
        return None


def check_retrieval(emb: list[float]) -> None:
    print("\n=== 4. Retrieval (búsqueda semántica) ===")
    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        results = query_collection(collection, query_embeddings=[emb], n_results=3)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        print(f"  OK  {len(docs)} chunks recuperados para: \"{QUERY}\"\n")
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            sim = 1 - dist
            tomo = meta.get("tomo", "?")
            tema = meta.get("tema", "?")
            print(f"  [{i}] Similitud: {sim:.3f} | Tomo {tomo} — {tema}")
            print(f"       {doc.strip()}")
            print()
    except Exception as e:
        print(f"  FAIL  {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("  RAG Verification — ProjectGrauTutor")
    print("=" * 50)

    if not check_chroma():
        print("\nDeteniéndose: ChromaDB no disponible.")
        sys.exit(1)

    count = check_collection()

    emb = check_embeddings()
    if emb is None:
        print("\nDeteniéndose: fallo en embeddings.")
        sys.exit(1)

    if count > 0:
        check_retrieval(emb)
    else:
        print("\n=== 4. Retrieval ===")
        print("  SKIP  Colección vacía, no hay nada que recuperar.")

    print("\n" + "=" * 50)
    print("  Verificación completa.")
    print("=" * 50)
