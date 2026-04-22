from __future__ import annotations
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from chromadb import Collection
from core.config import settings
from core.llm import get_llm
from core.logging import get_logger
from rag.embeddings import embed_query
from rag.store import query_collection
from rag.bm25 import bm25_search, load_bm25_index, rrf_fuse

logger = get_logger(__name__)

SYSTEM_PROMPT = """Eres Chess Tutor Grau, un tutor experto en ajedrez basado en los libros
"Tratado General de Ajedrez" de Roberto Grau. Responde en español de forma clara y pedagógica.

Usa ÚNICAMENTE el contenido de Grau proporcionado como contexto para responder.
Si el contexto no contiene información suficiente, indícalo honestamente.
Cita los conceptos y explicaciones de Grau cuando sea relevante.

Contexto de las partidas de Grau:
{context}
"""

HUMAN_PROMPT = "{question}"


def _format_items(items: list[dict]) -> str:
    parts = []
    for i, item in enumerate(items, 1):
        meta = item["meta"]
        doc = item["doc"]
        sim = item.get("similarity")
        tomo = meta.get("tomo", "?")
        tema = meta.get("tema", "?")
        white = meta.get("white", "?")
        black = meta.get("black", "?")
        eco = meta.get("eco", "")
        jugadas = meta.get("jugadas", "")

        header = f"[Fuente {i} — Tomo {tomo} ({tema}), {white} vs {black}"
        if eco:
            header += f", ECO {eco}"
        if sim is not None:
            header += f", similitud: {sim:.2f}"
        else:
            header += ", BM25"
        header += "]"

        block = [header, doc]
        if jugadas:
            block.append(f"Jugadas: {jugadas}")
        parts.append("\n".join(block))

    return "\n\n---\n\n".join(parts)


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
        self.llm = get_llm()
        self._build_chain()

    def _build_chain(self) -> None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self.chain = (
            {
                "context": RunnablePassthrough() | self._retrieve,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _hybrid_retrieve(self, question: str, top_n: int) -> list[dict]:
        query_emb = embed_query(question)
        dense_res = query_collection(
            self.collection,
            query_embeddings=[query_emb],
            n_results=self.n_candidates,
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
            fused_ids = rrf_fuse([dense_ids, sparse_ids], top_n=top_n)
            logger.info(
                f"Híbrido — dense: {len(dense_ids)} | bm25: {len(sparse_ids)} | "
                f"fusionados (top {top_n}): {len(fused_ids)}"
            )
        else:
            fused_ids = dense_ids[:top_n]
            logger.info("BM25 no disponible; usando solo dense.")

        missing = [id_ for id_ in fused_ids if id_ not in by_id]
        if missing:
            extra = self.collection.get(
                ids=missing,
                include=["documents", "metadatas"],
            )
            for id_, doc, meta in zip(extra["ids"], extra["documents"], extra["metadatas"]):
                by_id[id_] = {"doc": doc, "meta": meta, "distance": None}

        items = []
        for id_ in fused_ids:
            data = by_id[id_]
            sim = 1 - data["distance"] if data["distance"] is not None else None
            items.append({
                "id": id_,
                "doc": data["doc"],
                "meta": data["meta"],
                "similarity": sim,
            })
        return items

    def _retrieve(self, question: str) -> str:
        logger.info(f"RAG query: {question[:80]}...")
        items = self._hybrid_retrieve(question, top_n=self.n_results)
        logger.info(f"Contexto recuperado: {len(items)} chunks")
        return _format_items(items)

    def ask(self, question: str) -> str:
        logger.info(f"Pregunta al RAG: {question}")
        answer = self.chain.invoke(question)
        logger.info("Respuesta generada")
        return answer

    def retrieve_raw(self, question: str, n_results: int | None = None) -> list[dict]:
        """Devuelve los chunks fusionados (id, doc, meta, similarity) sin pasar por el LLM."""
        return self._hybrid_retrieve(question, top_n=n_results or self.n_results)
