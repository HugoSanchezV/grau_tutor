"""Tests del motor de tablero (funciones puras + LangChain tools)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import chess

from agents.tools.chess_engine import (
    validate_move,
    apply_move,
    list_legal_moves,
    analyze_position,
    render_board,
    build_chess_engine_tools,
)


START_FEN = chess.STARTING_FEN
# Mate en 1 para blancas: 1.Qxf7#  (Scholar's mate listo para rematar)
SCHOLARS_READY = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
# Posición con blancas a punto de ahogar a negras (FEN clásico de ahogado)
STALEMATE_BLACK_TO_MOVE = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
# Rey vs rey (material insuficiente)
KK_INSUFFICIENT = "8/8/4k3/8/8/4K3/8/8 w - - 0 1"


# ---------- validate_move ----------


def test_validate_san_legal_desde_inicio():
    r = validate_move(START_FEN, "e4")
    assert r.legal is True
    assert r.move_san == "e4"
    assert r.move_uci == "e2e4"


def test_validate_uci_legal_desde_inicio():
    r = validate_move(START_FEN, "g1f3")
    assert r.legal is True
    assert r.move_san == "Nf3"


def test_validate_san_ilegal():
    r = validate_move(START_FEN, "e5")  # las blancas no pueden jugar e5 de entrada
    assert r.legal is False
    assert r.razon


def test_validate_basura():
    r = validate_move(START_FEN, "zzz")
    assert r.legal is False


def test_validate_fen_invalido():
    r = validate_move("esto-no-es-fen", "e4")
    assert r.legal is False
    assert "FEN" in r.razon


def test_validate_enroque():
    # Blancas enroquables corto
    fen = "r3k2r/pppqbppp/2n2n2/3pp3/3PP3/2N2N2/PPPQBPPP/R3K2R w KQkq - 0 1"
    assert validate_move(fen, "O-O").legal is True
    assert validate_move(fen, "O-O-O").legal is True


def test_validate_promocion_uci():
    # Reyes alejados para que la promoción NO dé jaque (simplifica el SAN esperado)
    fen = "8/P7/8/8/8/8/8/4k2K w - - 0 1"
    r = validate_move(fen, "a7a8q")
    assert r.legal is True
    assert r.move_san == "a8=Q"


# ---------- apply_move ----------


def test_apply_move_basico():
    r = apply_move(START_FEN, "e4")
    assert r.san == "e4"
    assert r.uci == "e2e4"
    assert r.es_jaque is False
    assert r.es_mate is False
    assert " b " in r.fen_resultante  # siguiente a mover: negras


def test_apply_move_mate():
    # Dama + Rey vs Rey: mate en 1 con Qg7
    fen = "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1"
    r = apply_move(fen, "Qg7")
    assert r.es_mate is True
    assert r.es_jaque is True


def test_apply_move_jaque_sin_mate():
    # Rey blanco h1, torre en a1, rey negro en e8. Ra8+ da jaque por fila 8 sin mate.
    fen = "4k3/8/8/8/8/8/8/R6K w - - 0 1"
    r = apply_move(fen, "Ra8+")
    assert r.es_jaque is True
    assert r.es_mate is False


def test_apply_move_ilegal_lanza():
    with pytest.raises(ValueError):
        apply_move(START_FEN, "e5")


def test_apply_move_fen_invalido_lanza():
    with pytest.raises(ValueError):
        apply_move("fen-basura", "e4")


# ---------- list_legal_moves ----------


def test_list_legal_moves_inicio_tiene_20():
    moves = list_legal_moves(START_FEN)
    assert len(moves) == 20  # 16 peones + 4 caballos


def test_list_legal_moves_mate_vacio():
    # Mate del pastor terminado: negras en mate, sin jugadas
    mate_fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    moves = list_legal_moves(mate_fen)
    assert moves == []


# ---------- analyze_position ----------


def test_analyze_posicion_inicial():
    a = analyze_position(START_FEN)
    assert a.turno == "blancas"
    assert a.en_jaque is False
    assert a.mate is False
    assert a.ahogado is False
    assert a.material_blancas == a.material_negras == 1 * 8 + 3 * 2 + 3 * 2 + 5 * 2 + 9
    assert a.fullmove_number == 1


def test_analyze_mate():
    a = analyze_position("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    assert a.mate is True
    assert a.en_jaque is True


def test_analyze_ahogado():
    a = analyze_position(STALEMATE_BLACK_TO_MOVE)
    assert a.ahogado is True
    assert a.mate is False


def test_analyze_material_insuficiente():
    a = analyze_position(KK_INSUFFICIENT)
    assert a.tablas_material_insuficiente is True


def test_analyze_fen_invalido_lanza():
    with pytest.raises(ValueError):
        analyze_position("xxx")


# ---------- render_board ----------


def test_render_board_devuelve_svg():
    svg = render_board(START_FEN)
    assert svg.startswith("<svg") or "<svg" in svg[:200]
    assert "</svg>" in svg


def test_render_board_resalta_ultima_jugada():
    svg = render_board(START_FEN, last_move_uci="e2e4")
    # python-chess marca la última jugada con clase específica en el SVG
    assert "lastmove" in svg or "e2" in svg


def test_render_board_flipped():
    normal = render_board(START_FEN)
    flipped = render_board(START_FEN, flipped=True)
    assert normal != flipped


# ---------- tools de LangChain ----------


def test_build_tools_cuatro():
    tools = build_chess_engine_tools()
    names = [t.name for t in tools]
    assert set(names) == {"validate_move", "apply_move", "list_legal_moves", "analyze_position"}


def test_tool_validate_ok():
    tools = {t.name: t for t in build_chess_engine_tools()}
    out = tools["validate_move"].invoke({"fen": START_FEN, "move": "e4"})
    assert "Legal" in out
    assert "SAN=e4" in out


def test_tool_validate_ilegal():
    tools = {t.name: t for t in build_chess_engine_tools()}
    out = tools["validate_move"].invoke({"fen": START_FEN, "move": "e5"})
    assert out.startswith("Ilegal")


def test_tool_apply_reporta_mate():
    tools = {t.name: t for t in build_chess_engine_tools()}
    out = tools["apply_move"].invoke(
        {"fen": "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1", "move": "Qg7"}
    )
    assert "jaque mate" in out


def test_tool_list_devuelve_csv():
    tools = {t.name: t for t in build_chess_engine_tools()}
    out = tools["list_legal_moves"].invoke({"fen": START_FEN})
    assert "e4" in out and "Nf3" in out
    assert out.count(",") >= 10  # hay 20 jugadas, al menos 10 comas


def test_tool_analyze_muestra_turno():
    tools = {t.name: t for t in build_chess_engine_tools()}
    out = tools["analyze_position"].invoke({"fen": START_FEN})
    assert "Turno: blancas" in out
    assert "Material" in out


def test_tool_captura_fen_invalido_sin_levantar():
    """Los wrappers deben devolver string de error, no crashear."""
    tools = {t.name: t for t in build_chess_engine_tools()}
    out = tools["apply_move"].invoke({"fen": "basura", "move": "e4"})
    assert out.startswith("Error")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
