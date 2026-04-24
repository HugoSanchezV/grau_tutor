"""Motor de tablero: funciones puras sobre python-chess + wrappers como LangChain Tools.

Expone 4 herramientas al agente: validate_move, apply_move, list_legal_moves,
analyze_position. `render_board` queda como utilidad de UI (devuelve SVG) y NO
se expone al agente para no contaminar el contexto del LLM con strings SVG grandes.
"""
from __future__ import annotations
from typing import Literal, Optional

import chess
import chess.svg
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from core.logging import get_logger

logger = get_logger(__name__)


# ---------- salidas tipadas ----------


class MoveValidation(BaseModel):
    legal: bool
    move_san: Optional[str] = None
    move_uci: Optional[str] = None
    razon: str = ""


class MoveResult(BaseModel):
    fen_resultante: str
    san: str
    uci: str
    es_jaque: bool
    es_mate: bool
    es_tablas: bool


class PositionAnalysis(BaseModel):
    fen: str
    turno: Literal["blancas", "negras"]
    en_jaque: bool
    mate: bool
    ahogado: bool
    tablas_material_insuficiente: bool
    puede_reclamar_tablas: bool
    material_blancas: int
    material_negras: int
    halfmove_clock: int
    fullmove_number: int


# ---------- helpers ----------


def _parse_fen(fen: str) -> chess.Board:
    try:
        return chess.Board(fen)
    except ValueError as e:
        raise ValueError(f"FEN inválido: {e}")


def _parse_move(board: chess.Board, move: str) -> chess.Move:
    """Acepta SAN ('Nf3', 'O-O') o UCI ('g1f3', 'e7e8q')."""
    move = move.strip()
    try:
        return board.parse_san(move)
    except ValueError:
        pass
    try:
        m = chess.Move.from_uci(move.lower())
    except ValueError:
        raise ValueError(f"Jugada inválida (ni SAN ni UCI reconocido): {move}")
    if m not in board.legal_moves:
        raise ValueError(f"Jugada ilegal en esta posición: {move}")
    return m


_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def _material(board: chess.Board, color: chess.Color) -> int:
    return sum(
        len(board.pieces(piece, color)) * value
        for piece, value in _PIECE_VALUES.items()
    )


# ---------- funciones puras ----------


def validate_move(fen: str, move: str) -> MoveValidation:
    """Valida si `move` es legal en `fen`. No modifica la posición."""
    try:
        board = _parse_fen(fen)
    except ValueError as e:
        return MoveValidation(legal=False, razon=str(e))
    try:
        parsed = _parse_move(board, move)
        return MoveValidation(
            legal=True,
            move_san=board.san(parsed),
            move_uci=parsed.uci(),
        )
    except ValueError as e:
        return MoveValidation(legal=False, razon=str(e))


def apply_move(fen: str, move: str) -> MoveResult:
    """Aplica `move` sobre `fen` y devuelve el FEN resultante + flags."""
    board = _parse_fen(fen)
    parsed = _parse_move(board, move)
    san = board.san(parsed)
    uci = parsed.uci()
    board.push(parsed)
    return MoveResult(
        fen_resultante=board.fen(),
        san=san,
        uci=uci,
        es_jaque=board.is_check(),
        es_mate=board.is_checkmate(),
        es_tablas=(
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.can_claim_draw()
        ),
    )


def list_legal_moves(fen: str) -> list[str]:
    board = _parse_fen(fen)
    return [board.san(m) for m in board.legal_moves]


def analyze_position(fen: str) -> PositionAnalysis:
    board = _parse_fen(fen)
    return PositionAnalysis(
        fen=board.fen(),
        turno="blancas" if board.turn == chess.WHITE else "negras",
        en_jaque=board.is_check(),
        mate=board.is_checkmate(),
        ahogado=board.is_stalemate(),
        tablas_material_insuficiente=board.is_insufficient_material(),
        puede_reclamar_tablas=board.can_claim_draw(),
        material_blancas=_material(board, chess.WHITE),
        material_negras=_material(board, chess.BLACK),
        halfmove_clock=board.halfmove_clock,
        fullmove_number=board.fullmove_number,
    )


def render_board(
    fen: str,
    last_move_uci: Optional[str] = None,
    size: int = 360,
    flipped: bool = False,
) -> str:
    """Genera el SVG del tablero para la UI. No se expone como tool del agente."""
    board = _parse_fen(fen)
    lastmove = None
    if last_move_uci:
        try:
            candidate = chess.Move.from_uci(last_move_uci.lower())
            lastmove = candidate
        except ValueError:
            lastmove = None
    orientation = chess.BLACK if flipped else chess.WHITE
    check_square = board.king(board.turn) if board.is_check() else None
    return chess.svg.board(
        board=board,
        lastmove=lastmove,
        orientation=orientation,
        check=check_square,
        size=size,
    )


# ---------- schemas de entrada del agente ----------


class FenInput(BaseModel):
    fen: str = Field(..., description="Posición en notación FEN.")


class FenMoveInput(BaseModel):
    fen: str = Field(..., description="Posición en notación FEN.")
    move: str = Field(
        ...,
        description="Jugada en SAN ('Nf3', 'O-O', 'exd5') o UCI ('g1f3', 'e7e8q').",
    )


# ---------- wrappers (string → LLM) ----------


def _validate_tool(fen: str, move: str) -> str:
    r = validate_move(fen, move)
    if r.legal:
        return f"Legal. SAN={r.move_san} UCI={r.move_uci}"
    return f"Ilegal: {r.razon}"


def _apply_tool(fen: str, move: str) -> str:
    try:
        r = apply_move(fen, move)
    except ValueError as e:
        return f"Error: {e}"
    flags = []
    if r.es_mate:
        flags.append("jaque mate")
    elif r.es_jaque:
        flags.append("jaque")
    if r.es_tablas:
        flags.append("tablas")
    suffix = f" ({', '.join(flags)})" if flags else ""
    return f"Jugada {r.san} aplicada. FEN: {r.fen_resultante}{suffix}"


def _list_tool(fen: str) -> str:
    try:
        moves = list_legal_moves(fen)
    except ValueError as e:
        return f"Error: {e}"
    if not moves:
        return "Sin jugadas legales (mate o ahogado)."
    return ", ".join(moves)


def _analyze_tool(fen: str) -> str:
    try:
        a = analyze_position(fen)
    except ValueError as e:
        return f"Error: {e}"
    parts = [
        f"Turno: {a.turno}",
        f"Material: blancas={a.material_blancas} negras={a.material_negras}",
        f"Jugada #{a.fullmove_number}",
    ]
    if a.mate:
        parts.append("JAQUE MATE")
    elif a.en_jaque:
        parts.append("en jaque")
    if a.ahogado:
        parts.append("ahogado (tablas)")
    if a.tablas_material_insuficiente:
        parts.append("tablas por material insuficiente")
    if a.puede_reclamar_tablas and not a.ahogado:
        parts.append("se pueden reclamar tablas (50 mov / triple repetición)")
    return " | ".join(parts)


def build_chess_engine_tools() -> list[StructuredTool]:
    """Devuelve las 4 herramientas de motor de tablero listas para el agente ReAct."""
    return [
        StructuredTool.from_function(
            func=_validate_tool,
            name="validate_move",
            description=(
                "Verifica si una jugada es LEGAL en una posición FEN. "
                "Acepta SAN ('Nf3', 'O-O') o UCI ('g1f3'). "
                "Úsala antes de afirmar que una jugada del alumno es correcta."
            ),
            args_schema=FenMoveInput,
        ),
        StructuredTool.from_function(
            func=_apply_tool,
            name="apply_move",
            description=(
                "Aplica una jugada a una posición FEN y devuelve el FEN resultante "
                "más indicadores (jaque, mate, tablas). No la uses para validar — "
                "para eso está validate_move."
            ),
            args_schema=FenMoveInput,
        ),
        StructuredTool.from_function(
            func=_list_tool,
            name="list_legal_moves",
            description=(
                "Lista todas las jugadas legales en una posición FEN (notación SAN). "
                "Útil cuando el alumno pide opciones o para verificar jugadas candidatas."
            ),
            args_schema=FenInput,
        ),
        StructuredTool.from_function(
            func=_analyze_tool,
            name="analyze_position",
            description=(
                "Analiza una posición FEN: turno, material de cada bando, "
                "jaque/mate/ahogado/tablas reclamables."
            ),
            args_schema=FenInput,
        ),
    ]
