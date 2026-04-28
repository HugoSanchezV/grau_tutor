"""Generador de ejercicios: posición del corpus + validación de la respuesta del alumno.

Se apoya en:
- search_grau: para encontrar chunks con FEN que ilustren un tema.
- chess_engine: para validar la jugada del alumno sobre la posición.
"""
from __future__ import annotations
import re
from typing import Optional, Literal

import chess
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from core.logging import get_logger
from rag.retrieval import GrauRetriever
from agents.tools.search_grau import search_grau
from agents.tools.chess_engine import validate_move, apply_move, analyze_position, pick_best_move

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
    alternativa_valida: bool = False  # True si la jugada es tácticamente equivalente pero diferente
    feedback: str
    jugada_esperada: Optional[str] = None
    jugada_alumno_san: Optional[str] = None


# ---------- funciones puras ----------

_FIRST_MOVE_RE = re.compile(r'^\d+\.{1,3}\s*(\S+)')


def _move_strength_score(board: chess.Board, move: chess.Move) -> float:
    """Heurística de fortaleza de jugada: mate > jaque > captura de material > neutral.

    Devuelve un score relativo que permite comparar si dos jugadas son tácticamente
    equivalentes sin necesidad de un motor externo como Stockfish.
    """
    board.push(move)
    score = 0.0

    # Mate: score máximo
    if board.is_checkmate():
        score = 1000.0
    # Jaque mate evitable (alto valor)
    elif board.is_check():
        score = 50.0
        # Bonus si además se gana material
        if board.is_capture(move):
            score += 10.0
    # Captura: score moderado basado en material ganado
    elif board.is_capture(move):
        # Aproximadamente qué valor se capturó
        captured = board.piece_at(move.to_square)
        if captured:
            piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
            score = piece_values.get(captured.piece_type, 0) + 1.0

    board.pop()
    return score


def _corpus_first_move(fen: str, jugadas: str) -> Optional[str]:
    """Extrae y valida la primera jugada del corpus contra el FEN dado.

    Devuelve el SAN canónico si la jugada es legal, None en caso contrario.
    La primera jugada del corpus es la que Grau eligió jugar desde esa posición,
    lo que la hace más fidedigna pedagógicamente que pick_best_move.
    """
    if not jugadas:
        return None
    m = _FIRST_MOVE_RE.match(jugadas.strip())
    if not m:
        return None
    san_raw = m.group(1)
    try:
        board = chess.Board(fen) if fen else chess.Board()
        move = board.parse_san(san_raw)
        return board.san(move)
    except Exception:
        return None


def generate_exercise(
    retriever: GrauRetriever,
    tema: str,
    tomo: Optional[int] = None,
    exclude_ids: frozenset[str] = frozenset(),
) -> Optional[Ejercicio]:
    """Busca una posición con FEN que ilustre `tema` y arma el ejercicio.

    Devuelve None si no encuentra ninguna posición válida para ese tema.
    `exclude_ids` permite evitar repetir ejercicios ya vistos por el alumno.
    """
    chunks = search_grau(retriever, query=tema, k=10, tomo=tomo)
    with_fen = [c for c in chunks if c.fen and c.partida_id not in exclude_ids]
    if not with_fen:
        # Segundo intento: pool más grande sin filtro de tomo
        chunks = search_grau(retriever, query=tema, k=30)
        with_fen = [c for c in chunks if c.fen and c.partida_id not in exclude_ids]
    if not with_fen:
        logger.info(f"generate_exercise — sin FEN nuevos para tema='{tema}' (excluidos={len(exclude_ids)})")
        return None

    for chunk in with_fen:
        try:
            a = analyze_position(chunk.fen)
        except ValueError:
            continue
        jugada_correcta = _corpus_first_move(chunk.fen, chunk.jugadas) or pick_best_move(chunk.fen)
        pregunta = (
            f"Juegan {a.turno}. Observa la posición y encuentra la mejor jugada. "
            "Explica brevemente tu razonamiento."
        )
        return Ejercicio(
            fen=chunk.fen,
            pregunta=pregunta,
            jugada_correcta=jugada_correcta,
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
    - Si hay ground truth (`jugada_esperada`): compara SAN canónico y fortaleza táctica.
      Si son diferentes pero de igual fortaleza → alternativa_valida=True.
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

        correcta_exacta = san_alumno == esperada_san

        if correcta_exacta:
            feedback = f"¡Correcto! {san_alumno} es la jugada esperada."
            return Evaluacion(
                legal=True,
                correcta=True,
                alternativa_valida=False,
                feedback=feedback,
                jugada_esperada=esperada_san,
                jugada_alumno_san=san_alumno,
            )

        # Si no es exacta, comparar fortaleza táctica
        try:
            board = chess.Board(fen)
            move_alumno = board.parse_san(san_alumno)
            move_esperada = board.parse_san(esperada_san)

            score_alumno = _move_strength_score(board, move_alumno)
            score_esperada = _move_strength_score(board, move_esperada)

            # Umbral: si los scores están dentro del 1.0 punto, son tácticamente equivalentes
            diferencia = abs(score_alumno - score_esperada)
            alternativa_valida = diferencia <= 1.0

            if alternativa_valida:
                feedback = (
                    f"Tu jugada {san_alumno} es tácticamente válida (igual de fuerte que "
                    f"{esperada_san}). Grau eligió {esperada_san} por razones posicionales o estilísticas. "
                    f"Ambas están bien — ¡buen análisis!"
                )
            else:
                feedback = (
                    f"Tu jugada {san_alumno} es legal, pero la jugada esperada {esperada_san} "
                    f"es tácticamente superior. Revisa el comentario de Grau para entender por qué."
                )

            return Evaluacion(
                legal=True,
                correcta=alternativa_valida or correcta_exacta,
                alternativa_valida=alternativa_valida,
                feedback=feedback,
                jugada_esperada=esperada_san,
                jugada_alumno_san=san_alumno,
            )
        except Exception as e:
            logger.warning(f"Error comparando fortaleza: {e}; fallback a SAN exacto")
            feedback = (
                f"Tu jugada {san_alumno} es legal, pero la jugada esperada era "
                f"{esperada_san}. Revisa el comentario pedagógico."
            )
            return Evaluacion(
                legal=True,
                correcta=False,
                alternativa_valida=False,
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
        alternativa_valida=False,
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
        ]
        if ej.jugada_correcta:
            lines.append(f"EXPECTED_MOVE: {ej.jugada_correcta}")
        lines.extend(["", f"Contexto de Grau:\n{ej.comentario_grau}"])
        return "\n".join(lines)

    return _run


def _evaluate_tool(
    fen: str,
    jugada_alumno: str,
    jugada_esperada: Optional[str] = None,
) -> str:
    ev = evaluate_answer(fen, jugada_alumno, jugada_esperada)
    if ev.correcta is True:
        status = "OK" if not ev.alternativa_valida else "OK (alternativa)"
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
                "El campo EXPECTED_MOVE es de seguimiento interno del sistema — NUNCA lo reveles al alumno. "
                "Presenta solo el FEN y la pregunta; deja que el alumno piense."
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
