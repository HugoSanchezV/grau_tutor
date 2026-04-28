from __future__ import annotations
from chromadb import Collection
from core.config import settings
from core.logging import get_logger
from rag.embeddings import embed_query
from rag.store import query_collection
from rag.bm25 import bm25_search, load_bm25_index, rrf_fuse

logger = get_logger(__name__)


class GrauRetriever:
    def __init__(
        self,
        collection: Collection,
        n_results: int = 5,
        n_candidates: int = 20,
        bm25_index_path: str | None = None,
    ):
        self.collection = collection
        self.n_results = n_results
        self.n_candidates = n_candidates
        self.bm25_index = load_bm25_index(bm25_index_path or settings.bm25_index_path)

        # Startup consistency check: verificar que BM25 e ChromaDB estén sincronizados
        if self.bm25_index:
            self._check_bm25_chroma_sync()

    def _check_bm25_chroma_sync(self) -> None:
        """Verifica que el índice BM25 está sincronizado con ChromaDB al arrancar."""
        if not self.bm25_index:
            return

        sample_size = min(10, len(self.bm25_index.get("ids", [])))
        if sample_size == 0:
            return

        sample_ids = self.bm25_index["ids"][:sample_size]
        existing = self.collection.get(ids=sample_ids, include=[])
        existing_ids = set(existing.get("ids", []))
        missing = [id_ for id_ in sample_ids if id_ not in existing_ids]

        if missing:
            logger.warning(
                f"BM25/ChromaDB desincronización detectada al arrancar: {len(missing)}/{sample_size} "
                f"IDs del sample no existen en ChromaDB. Ejecuta: python rag/pipeline.py --rebuild-bm25"
            )

    def _hybrid_retrieve(self, question: str, top_n: int, where: dict | None = None) -> list[dict]:
        query_emb = embed_query(question)
        dense_res = query_collection(
            self.collection,
            query_embeddings=[query_emb],
            n_results=self.n_candidates,
            where=where,
        )
        dense_ids = dense_res["ids"][0]
        by_id: dict[str, dict] = {
            id_: {"doc": doc, "meta": meta, "distance": dist}
            for id_, doc, meta, dist in zip(
                dense_ids,
                dense_res["documents"][0],
                dense_res["metadatas"][0],
                dense_res["distances"][0],
            )
        }

        if self.bm25_index:
            sparse_hits = bm25_search(self.bm25_index, question, n=self.n_candidates)
            sparse_ids = [id_ for id_, _ in sparse_hits]

            # Filtrar IDs de BM25 que no existen en ChromaDB ANTES de RRF
            # para evitar asignarles ranks altos que luego se descartan
            valid_sparse_ids = [id_ for id_ in sparse_ids if id_ in by_id]
            stale_count = len(sparse_ids) - len(valid_sparse_ids)
            if stale_count > 0:
                stale_ratio = stale_count / len(sparse_ids) if sparse_ids else 0
                if stale_ratio > 0.05:  # > 5% de stale IDs = error
                    logger.error(
                        f"BM25 índice desincronizado: {stale_count}/{len(sparse_ids)} IDs "
                        f"({stale_ratio*100:.1f}%) no existen en ChromaDB. "
                        f"Ejecuta: python rag/pipeline.py --rebuild-bm25"
                    )
                else:
                    logger.warning(
                        f"BM25 tiene {stale_count} IDs stale (ratio={stale_ratio*100:.1f}%)"
                    )

            fused_ids = rrf_fuse([dense_ids, valid_sparse_ids], top_n=top_n)
            logger.info(
                f"Híbrido — dense: {len(dense_ids)} | bm25 válidos: {len(valid_sparse_ids)} "
                f"({stale_count} stale) | fusionados (top {top_n}): {len(fused_ids)}"
            )
        else:
            fused_ids = dense_ids[:top_n]
            logger.info("BM25 no disponible; usando solo dense.")

        missing = [id_ for id_ in fused_ids if id_ not in by_id]
        if missing:
            get_kwargs: dict = {"ids": missing, "include": ["documents", "metadatas"]}
            if where:
                # Filter BM25-only candidates by the same metadata constraint
                get_kwargs["where"] = where
            extra = self.collection.get(**get_kwargs)
            for id_, doc, meta in zip(extra["ids"], extra["documents"], extra["metadatas"]):
                by_id[id_] = {"doc": doc, "meta": meta, "distance": None}
            if not where:
                # Only warn when no filter is active; filtered-out IDs are intentionally absent
                still_missing = [id_ for id_ in missing if id_ not in by_id]
                if still_missing:
                    logger.warning(
                        f"{len(still_missing)} IDs en BM25 no encontrados en ChromaDB (índice desincronizado): "
                        f"{still_missing[:5]}"
                    )

        items = []
        for id_ in fused_ids:
            data = by_id.get(id_)
            if data is None:
                continue
            sim = 1 - data["distance"] if data["distance"] is not None else None
            items.append({
                "id": id_,
                "doc": data["doc"],
                "meta": data["meta"],
                "similarity": sim,
            })
        return items

    def retrieve_raw(self, question: str, n_results: int | None = None, where: dict | None = None) -> list[dict]:
        """Devuelve los chunks fusionados (id, doc, meta, similarity) sin pasar por el LLM."""
        top_n = n_results or self.n_results
        logger.info(f"RAG query: {question[:80]}... (top_n={top_n})")
        items = self._hybrid_retrieve(question, top_n=top_n, where=where)
        logger.info(f"Contexto recuperado: {len(items)} chunks")
        return items
