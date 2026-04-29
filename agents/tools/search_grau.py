from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import StructuredTool
from core.logging import get_logger
from rag.retrieval import GrauRetriever

logger = get_logger(__name__)


@dataclass
class SearchKConfig:
    """Holder mutable del k por defecto. Se sincroniza desde la UI en tiempo de ejecución."""
    default_k: int = 3


class ChunkResult(BaseModel):
    partida_id: str
    tomo: int
    tema: str
    white: str
    black: str
    result: str
    eco: str
    fen: str
    jugadas: str
    texto: str
    resumen_simple: str = ""
    similarity: Optional[float] = None

    @classmethod
    def from_retrieval_item(cls, item: dict) -> "ChunkResult":
        meta = item["meta"]
        return cls(
            partida_id=meta.get("partida_id", item.get("id", "")),
            tomo=int(meta.get("tomo", 0)),
            tema=meta.get("tema", "?"),
            white=meta.get("white", "?"),
            black=meta.get("black", "?"),
            result=meta.get("result", "*"),
            eco=meta.get("eco", ""),
            fen=meta.get("fen", ""),
            jugadas=meta.get("jugadas", ""),
            resumen_simple=meta.get("resumen_simple", ""),
            texto=item["doc"],
            similarity=item.get("similarity"),
        )


class SearchGrauInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Pregunta o concepto a buscar en los libros de Grau "
            "(ej. 'clavada', 'peón pasado', 'ataque al rey')."
        ),
    )
    k: Optional[int] = Field(
        default=None,
        description=(
            "Número de pasajes a devolver (1–15). Si se omite, usa el k "
            "por defecto configurado para la sesión."
        ),
    )

    @field_validator("k", mode="before")
    @classmethod
    def coerce_k(cls, v: object) -> Optional[int]:
        if v is None or v == "" or str(v).lower() in ("null", "none"):
            return None
        try:
            val = int(v)
        except (ValueError, TypeError):
            raise ValueError(f"k inválido: {v!r}")
        if not (1 <= val <= 15):
            raise ValueError(f"k debe estar entre 1 y 15; se recibió {val}")
        return val
    tomo: Optional[int] = Field(
        default=None,
        description=(
            "Filtro opcional por tomo: pasa el entero 1, 2, 3 o 4 "
            "(1=Rudimentos, 2=Estrategia, 3=Medio juego, 4=Estrategia Superior), "
            "o null si no quieres filtrar. NUNCA pases el número como string."
        ),
    )

    @field_validator("tomo", mode="before")
    @classmethod
    def coerce_tomo(cls, v: object) -> Optional[int]:
        if v is None or v == "" or str(v).lower() in ("null", "none"):
            return None
        try:
            val = int(v)
        except (ValueError, TypeError):
            raise ValueError(f"tomo inválido: {v!r}")
        if not (1 <= val <= 4):
            raise ValueError(f"tomo debe ser 1, 2, 3 o 4; se recibió {val}")
        return val

    tema: Optional[str] = Field(
        default=None,
        description="Filtro opcional por tema exacto (coincide con el campo 'tema' del chunk).",
    )


def _filter_chunks(
    chunks: list[ChunkResult],
    tema: Optional[str],
) -> list[ChunkResult]:
    if tema is None:
        return chunks
    return [c for c in chunks if c.tema.lower() == tema.lower()]


def search_grau(
    retriever: GrauRetriever,
    query: str,
    k: int = 5,
    tomo: Optional[int] = None,
    tema: Optional[str] = None,
) -> list[ChunkResult]:
    """Búsqueda programática (devuelve objetos tipados).

    El filtro por tomo se empuja a ChromaDB (where clause); evita recuperar
    candidatos del tomo equivocado antes de filtrar.
    El filtro por tema (nombre de capítulo) sigue siendo post-filtro en Python,
    ya que no discrimina bien en ChromaDB para consultas semánticas.
    """
    where: Optional[dict] = {"tomo": tomo} if tomo is not None else None
    # Solo inflar el pool cuando hay post-filtro por tema; tomo lo maneja ChromaDB
    pool = k * 4 if tema is not None else k
    raw_items = retriever.retrieve_raw(query, n_results=pool, where=where)
    chunks = [ChunkResult.from_retrieval_item(it) for it in raw_items]
    filtered = _filter_chunks(chunks, tema)  # tomo ya filtrado por ChromaDB where clause
    logger.info(
        f"search_grau — query='{query[:60]}...' pool={len(chunks)} "
        f"post_filter={len(filtered)} k={k}"
    )
    return filtered[:k]


def format_chunks_for_llm(chunks: list[ChunkResult]) -> str:
    if not chunks:
        return "Sin resultados en el corpus de Grau para esa consulta."
    parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[Fuente {i}] Tomo {c.tomo} ({c.tema}): {c.white} vs {c.black} {c.result}"
        if c.eco:
            header += f" | ECO {c.eco}"
        header += f"\n        ID: {c.partida_id}"

        block = [header]

        # Resumen educativo
        if c.resumen_simple:
            block.append(f"\n📚 CLAVE: {c.resumen_simple}")

        # Análisis pedagógico completo
        block.append(f"\nAnálisis:\n{c.texto}")

        # Movimientos
        if c.jugadas:
            block.append(f"\nMovimientos: {c.jugadas}")

        # FEN si existe
        if c.fen:
            block.append(f"\nFEN: {c.fen}")

        parts.append("\n".join(block))
    return "\n\n" + "="*60 + "\n\n".join(parts)


TOOL_DESCRIPTION = (
    "Busca en los libros 'Tratado General de Ajedrez' de Roberto Grau. "
    "Úsala para responder dudas conceptuales de ajedrez (táctica, estrategia, "
    "aperturas, finales) o para encontrar partidas anotadas por tomo/tema. "
    "Devuelve pasajes pedagógicos con metadatos (tomo, jugadores, ECO, FEN, jugadas). "
    "Filtros opcionales: tomo (1–4) y tema."
)


def build_search_grau_tool(
    retriever: GrauRetriever,
    k_config: Optional[SearchKConfig] = None,
) -> StructuredTool:
    """Construye la LangChain Tool a partir de un retriever ya inicializado.

    `k_config` es un holder mutable: el valor `default_k` puede actualizarse
    en caliente desde la UI sin reconstruir el agente.
    """
    k_config = k_config or SearchKConfig()

    def _run(
        query: str,
        k: Optional[int] = None,
        tomo: Optional[int] = None,
        tema: Optional[str] = None,
    ) -> str:
        effective_k = k if k is not None else k_config.default_k
        chunks = search_grau(retriever, query=query, k=effective_k, tomo=tomo, tema=tema)
        return format_chunks_for_llm(chunks)

    return StructuredTool.from_function(
        func=_run,
        name="search_grau",
        description=TOOL_DESCRIPTION,
        args_schema=SearchGrauInput,
    )
