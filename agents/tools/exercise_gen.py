"""Generador de ejercicios: posición del corpus + validación de la respuesta del alumno.

Se apoya en:
- search_grau: para encontrar chunks con FEN que ilustren un tema.
- chess_engine: para validar la jugada del alumno sobre la posición.
"""
from __future__ import annotations
from typing import Optional, Literal

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from core.logging import get_logger
from rag.retrieval import GrauRetriever
from agents.tools.search_grau import search_grau
from agents.tools.chess_engine import validate_move, apply_move, analyze_position

logger = get_logger(__name__)


# ---------- salidas tipadas ----------


class Ejercicio(BaseModel):
    fen: str
    pregunta: str
    jugada_correcta: Optional[str] = None
    comentario_grau: str
    tomo: int
    partida_id: str
    turno: Literal["blancas", "negras"]


class Evaluacion(BaseModel):
    legal: bool
    correcta: Optional[bool] = None  # None si no hay ground truth
    feedback: str
    jugada_esperada: Optional[str] = None
    jugada_alumno_san: Optional[str] = None


# ---------- funciones puras ----------


def generate_exercise(
    retriever: GrauRetriever,
    tema: str,
    tomo: Optional[int] = None,
) -> Optional[Ejercicio]:
    """Busca una posición con FEN que ilustre `tema` y arma el ejercicio.

    Devuelve None si no encuentra ninguna posición válida para ese tema.
    No filtra por `tema` exacto: el campo meta.tema del corpus es muy grueso
    (Rudimentos / Estrategia / ...), así que confiamos en la búsqueda semántica.
    """
    chunks = search_grau(retriever, query=tema, k=10, tomo=tomo)
    with_fen = [c for c in chunks if c.fen]
    if not with_fen:
        # Segundo intento: pool más grande sin filtro de tomo
        chunks = search_grau(retriever, query=tema, k=30)
        with_fen = [c for c in chunks if c.fen]
    if not with_fen:
        logger.info(f"generate_exercise — sin FEN disponibles para tema='{tema}'")
        return None

    for chunk in with_fen:
        try:
            a = analyze_position(chunk.fen)
        except ValueError:
            continue
        pregunta = (
            f"Juegan {a.turno}. Observa la posición y encuentra la mejor jugada. "
            "Explica brevemente tu razonamiento."
        )
        return Ejercicio(
            fen=chunk.fen,
            pregunta=pregunta,
            jugada_correcta=None,
            comentario_grau=chunk.texto,
            tomo=chunk.tomo,
            partida_id=chunk.partida_id,
            turno=a.turno,
        )

    logger.info(f"generate_exercise — todos los FEN candidatos inválidos para tema='{tema}'")
    return None


def evaluate_answer(
    fen: str,
    jugada_alumno: str,
    jugada_esperada: Optional[str] = None,
) -> Evaluacion:
    """Evalúa la jugada del alumno sobre `fen`.

    - Si la jugada es ilegal: legal=False, correcta=False.
    - Si hay ground truth (`jugada_esperada`): compara SAN canónico.
    - Si no hay ground truth: correcta=None y feedback técnico con flags del motor.
    """
    validation = validate_move(fen, jugada_alumno)
    if not validation.legal:
        return Evaluacion(
            legal=False,
            correcta=False,
            feedback=f"La jugada '{jugada_alumno}' no es legal: {validation.razon}",
            jugada_esperada=jugada_esperada,
        )

    try:
        result = apply_move(fen, jugada_alumno)
    except ValueError as e:
        return Evaluacion(
            legal=False,
            correcta=False,
            feedback=f"No se pudo aplicar la jugada: {e}",
            jugada_esperada=jugada_esperada,
        )

    san_alumno = result.san

    if jugada_esperada:
        # Normalizamos comparando sobre SAN canónico del motor.
        esperada_validation = validate_move(fen, jugada_esperada)
        esperada_san = esperada_validation.move_san or jugada_esperada
        correcta = san_alumno == esperada_san
        if correcta:
            feedback = f"¡Correcto! {san_alumno} es la jugada esperada."
        else:
            feedback = (
                f"Tu jugada {san_alumno} es legal, pero la jugada esperada era "
                f"{esperada_san}. Revisa el comentario pedagógico."
            )
        return Evaluacion(
            legal=True,
            correcta=correcta,
            feedback=feedback,
            jugada_esperada=esperada_san,
            jugada_alumno_san=san_alumno,
        )

    flags = []
    if result.es_mate:
        flags.append("da jaque mate")
    elif result.es_jaque:
        flags.append("da jaque")
    if result.es_tablas:
        flags.append("lleva a tablas")

    extra = f" Observación: la jugada {', '.join(flags)}." if flags else ""
    feedback = (
        f"Tu jugada {san_alumno} es legal.{extra} "
        "Compara tu razonamiento con el comentario de Grau para la posición."
    )
    return Evaluacion(
        legal=True,
        correcta=None,
        feedback=feedback,
        jugada_alumno_san=san_alumno,
    )


# ---------- schemas de entrada del agente ----------


class GenerateExerciseInput(BaseModel):
    tema: str = Field(
        ...,
        description=(
            "Tema del ejercicio (ej. 'clavada', 'peón pasado', 'ataque al rey'). "
            "Se usa como query semántica sobre el corpus de Grau."
        ),
    )
    tomo: Optional[int] = Field(
        default=None,
        ge=1,
        le=4,
        description="Filtro opcional por tomo (1–4).",
    )


class EvaluateAnswerInput(BaseModel):
    fen: str = Field(..., description="Posición FEN del ejercicio.")
    jugada_alumno: str = Field(
        ...,
        description="Respuesta del alumno en SAN ('Nf3') o UCI ('g1f3').",
    )
    jugada_esperada: Optional[str] = Field(
        default=None,
        description="Jugada esperada si se conoce. Si se omite, el feedback es abierto.",
    )


# ---------- wrappers (string → LLM) ----------


def _build_generate_tool(retriever: GrauRetriever):
    def _run(tema: str, tomo: Optional[int] = None) -> str:
        ej = generate_exercise(retriever, tema=tema, tomo=tomo)
        if ej is None:
            return f"No encontré una posición con FEN para el tema '{tema}'."
        lines = [
            f"Ejercicio (Tomo {ej.tomo}, partida {ej.partida_id}):",
            f"FEN: {ej.fen}",
            f"Turno: {ej.turno}",
            f"Pregunta: {ej.pregunta}",
            "",
            f"Contexto de Grau:\n{ej.comentario_grau}",
        ]
        return "\n".join(lines)

    return _run


def _evaluate_tool(
    fen: str,
    jugada_alumno: str,
    jugada_esperada: Optional[str] = None,
) -> str:
    ev = evaluate_answer(fen, jugada_alumno, jugada_esperada)
    if ev.correcta is True:
        status = "OK"
    elif ev.correcta is False:
        status = "ERROR"
    else:
        status = "ABIERTA"
    parts = [f"[{status}] {ev.feedback}"]
    if ev.jugada_alumno_san:
        parts.append(f"SAN alumno: {ev.jugada_alumno_san}")
    if ev.jugada_esperada:
        parts.append(f"SAN esperada: {ev.jugada_esperada}")
    return " | ".join(parts)


def build_exercise_gen_tools(retriever: GrauRetriever) -> list[StructuredTool]:
    """Devuelve las 2 herramientas del generador de ejercicios."""
    return [
        StructuredTool.from_function(
            func=_build_generate_tool(retriever),
            name="generate_exercise",
            description=(
                "Genera un ejercicio a partir de un tema usando posiciones del corpus de Grau. "
                "Devuelve FEN, pregunta y contexto pedagógico. "
                "Guarda el FEN devuelto para pasárselo después a evaluate_answer."
            ),
            args_schema=GenerateExerciseInput,
        ),
        StructuredTool.from_function(
            func=_evaluate_tool,
            name="evaluate_answer",
            description=(
                "Evalúa la respuesta del alumno sobre un FEN de ejercicio. "
                "Verifica legalidad con el motor y compara con la jugada esperada si se conoce."
            ),
            args_schema=EvaluateAnswerInput,
        ),
    ]
