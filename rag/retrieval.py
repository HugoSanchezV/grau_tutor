from __future__ import annotations
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from chromadb import Collection
from core.llm import get_llm
from core.logging import get_logger
from rag.embeddings import embed_query
from rag.store import query_collection

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


def _format_docs(results: dict) -> str:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        tomo = meta.get("tomo", "?")
        tema = meta.get("tema", "?")
        white = meta.get("white", "?")
        black = meta.get("black", "?")
        similarity = 1 - dist  # distancia coseno → similitud
        parts.append(
            f"[Fuente {i} — Tomo {tomo} ({tema}), {white} vs {black}, similitud: {similarity:.2f}]\n{doc}"
        )

    return "\n\n---\n\n".join(parts)


class GrauRetriever:
    def __init__(self, collection: Collection, n_results: int = 5):
        self.collection = collection
        self.n_results = n_results
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

    def _retrieve(self, question: str) -> str:
        logger.info(f"RAG query: {question[:80]}...")
        query_emb = embed_query(question)
        results = query_collection(
            self.collection,
            query_embeddings=[query_emb],
            n_results=self.n_results,
        )
        context = _format_docs(results)
        logger.info(f"Contexto recuperado: {len(results.get('documents', [[]])[0])} chunks")
        return context

    def ask(self, question: str) -> str:
        logger.info(f"Pregunta al RAG: {question}")
        answer = self.chain.invoke(question)
        logger.info("Respuesta generada")
        return answer

    def retrieve_raw(self, question: str, n_results: int | None = None) -> dict:
        """Devuelve los chunks crudos sin pasar por el LLM (útil para herramientas del agente)."""
        n = n_results or self.n_results
        query_emb = embed_query(question)
        return query_collection(self.collection, query_embeddings=[query_emb], n_results=n)
