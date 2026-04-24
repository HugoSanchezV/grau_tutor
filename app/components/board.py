"""Componente de tablero SVG para la UI de Streamlit.

Extrae FENs del texto del agente y renderiza el tablero con chess.svg.board().
No se expone como tool del agente — es pura utilidad de presentación.
"""
from __future__ import annotations
import re
from typing import Optional

import chess
import streamlit as st

from agents.tools.chess_engine import render_board

# Regex permisivo que captura cualquier cadena con aspecto de FEN
_FEN_RE = re.compile(
    r"(?:FEN[:\s]+)?"            # prefijo opcional "FEN: "
    r"([1-8pPnNbBrRqQkK/]{10,}"  # parte de piezas (mínimo 10 chars)
    r"\s+[wb]"                    # turno
    r"(?:\s+[-KQkq]{1,4})?"      # enroque (opcional)
    r"(?:\s+[-a-h1-8]{1,2})?"    # al paso (opcional)
    r"(?:\s+\d+\s+\d+)?)"        # contadores (opcional)
)


def extract_fen(text: str) -> Optional[str]:
    """Devuelve el primer FEN válido encontrado en `text`, o None."""
    for match in _FEN_RE.finditer(text):
        candidate = match.group(1).strip()
        try:
            chess.Board(candidate)  # valida sintaxis
            return candidate
        except ValueError:
            continue
    return None


def render_board_panel(
    fen: str,
    last_move_uci: Optional[str] = None,
    flipped: bool = False,
    size: int = 370,
) -> None:
    """Renderiza el tablero SVG en el contenedor activo de Streamlit."""
    try:
        svg = render_board(fen, last_move_uci=last_move_uci, size=size, flipped=flipped)
        # Streamlit no acepta SVG directamente en st.image → usamos HTML
        st.markdown(
            f'<div style="display:flex;justify-content:center;">{svg}</div>',
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.error(f"No se pudo renderizar el tablero: {exc}")
