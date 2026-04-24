from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import StructuredTool
from core.logging import get_logger
from rag.retrieval import GrauRetriever

logger = get_logger(__name__)


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
    k: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Número de pasajes a devolver (1–15).",
    )
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
    tomo: Optional[int],
    tema: Optional[str],
) -> list[ChunkResult]:
    if tomo is None and tema is None:
        return chunks
    out = []
    for c in chunks:
        if tomo is not None and c.tomo != tomo:
            continue
        if tema is not None and c.tema.lower() != tema.lower():
            continue
        out.append(c)
    return out


def search_grau(
    retriever: GrauRetriever,
    query: str,
    k: int = 5,
    tomo: Optional[int] = None,
    tema: Optional[str] = None,
) -> list[ChunkResult]:
    """Búsqueda programática (devuelve objetos tipados).

    Si hay filtros, se recupera un pool más grande y se filtra después,
    para no quedarnos cortos de resultados.
    """
    pool = k * 4 if (tomo is not None or tema is not None) else k
    raw_items = retriever.retrieve_raw(query, n_results=pool)
    chunks = [ChunkResult.from_retrieval_item(it) for it in raw_items]
    filtered = _filter_chunks(chunks, tomo, tema)
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
        header = f"[Fuente {i} — Tomo {c.tomo} ({c.tema}), {c.white} vs {c.black}"
        if c.eco:
            header += f", ECO {c.eco}"
        if c.similarity is not None:
            header += f", sim={c.similarity:.2f}"
        header += f", id={c.partida_id}]"

        block = [header, c.texto]
        if c.jugadas:
            block.append(f"Jugadas: {c.jugadas}")
        if c.fen:
            block.append(f"FEN: {c.fen}")
        parts.append("\n".join(block))
    return "\n\n---\n\n".join(parts)


TOOL_DESCRIPTION = (
    "Busca en los libros 'Tratado General de Ajedrez' de Roberto Grau. "
    "Úsala para responder dudas conceptuales de ajedrez (táctica, estrategia, "
    "aperturas, finales) o para encontrar partidas anotadas por tomo/tema. "
    "Devuelve pasajes pedagógicos con metadatos (tomo, jugadores, ECO, FEN, jugadas). "
    "Filtros opcionales: tomo (1–4) y tema."
)


def build_search_grau_tool(retriever: GrauRetriever) -> StructuredTool:
    """Construye la LangChain Tool a partir de un retriever ya inicializado."""

    def _run(
        query: str,
        k: int = 5,
        tomo: Optional[int] = None,
        tema: Optional[str] = None,
    ) -> str:
        chunks = search_grau(retriever, query=query, k=k, tomo=tomo, tema=tema)
        return format_chunks_for_llm(chunks)

    return StructuredTool.from_function(
        func=_run,
        name="search_grau",
        description=TOOL_DESCRIPTION,
        args_schema=SearchGrauInput,
    )
