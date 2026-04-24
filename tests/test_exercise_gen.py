"""Tests del generador de ejercicios (mockea el retriever)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock
import pytest
import chess

from agents.tools.exercise_gen import (
    Ejercicio,
    Evaluacion,
    GenerateExerciseInput,
    EvaluateAnswerInput,
    generate_exercise,
    evaluate_answer,
    build_exercise_gen_tools,
)


START_FEN = chess.STARTING_FEN
# Dama + Rey vs Rey con mate en 1 jugando Qg7 (blancas a mover)
MATE_EN_1_FEN = "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1"


def _make_item(
    partida_id: str,
    tomo: int = 2,
    tema: str = "Estrategia",
    fen: str = START_FEN,
    doc: str = "Comentario pedagógico de Grau sobre la posición.",
    similarity: float = 0.82,
) -> dict:
    return {
        "id": partida_id,
        "doc": doc,
        "meta": {
            "partida_id": partida_id,
            "tomo": tomo,
            "tema": tema,
            "white": "Capablanca",
            "black": "Lasker",
            "result": "1-0",
            "eco": "C60",
            "fen": fen,
            "jugadas": "1.e4 e5 2.Nf3 Nc6",
        },
        "similarity": similarity,
    }


def _retriever_with(items: list[dict]) -> MagicMock:
    mock = MagicMock()
    mock.retrieve_raw.return_value = items
    return mock


# ---------- generate_exercise ----------


def test_generate_exercise_devuelve_ejercicio_con_fen():
    retriever = _retriever_with([_make_item("tomo2-1", fen=START_FEN)])
    ej = generate_exercise(retriever, tema="aperturas")
    assert ej is not None
    assert isinstance(ej, Ejercicio)
    assert ej.fen == START_FEN
    assert ej.tomo == 2
    assert ej.partida_id == "tomo2-1"
    assert ej.turno == "blancas"
    assert "blancas" in ej.pregunta.lower()
    assert ej.jugada_correcta is None
    assert "Grau" in ej.comentario_grau or "pedagógico" in ej.comentario_grau


def test_generate_exercise_sin_fen_devuelve_none():
    items = [_make_item("tomo2-1", fen=""), _make_item("tomo2-2", fen="")]
    retriever = _retriever_with(items)
    assert generate_exercise(retriever, tema="x") is None


def test_generate_exercise_salta_chunks_sin_fen():
    items = [
        _make_item("tomo2-1", fen=""),
        _make_item("tomo2-2", fen=START_FEN),
    ]
    retriever = _retriever_with(items)
    ej = generate_exercise(retriever, tema="x")
    assert ej is not None
    assert ej.partida_id == "tomo2-2"


def test_generate_exercise_salta_fen_invalido():
    items = [
        _make_item("tomo2-1", fen="esto-no-es-fen"),
        _make_item("tomo2-2", fen=START_FEN),
    ]
    retriever = _retriever_with(items)
    ej = generate_exercise(retriever, tema="x")
    assert ej is not None
    assert ej.partida_id == "tomo2-2"


def test_generate_exercise_detecta_turno_negras():
    fen_negras = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    retriever = _retriever_with([_make_item("tomo2-1", fen=fen_negras)])
    ej = generate_exercise(retriever, tema="x")
    assert ej is not None
    assert ej.turno == "negras"
    assert "negras" in ej.pregunta.lower()


def test_generate_exercise_propaga_tomo_al_retriever():
    retriever = _retriever_with([_make_item("tomo3-1", tomo=3, fen=START_FEN)])
    generate_exercise(retriever, tema="final", tomo=3)
    retriever.retrieve_raw.assert_called_once()


# ---------- evaluate_answer ----------


def test_evaluate_answer_legal_sin_esperada_es_abierta():
    ev = evaluate_answer(START_FEN, "e4")
    assert ev.legal is True
    assert ev.correcta is None
    assert ev.jugada_alumno_san == "e4"
    assert "legal" in ev.feedback.lower()


def test_evaluate_answer_ilegal():
    ev = evaluate_answer(START_FEN, "e5")  # las blancas no pueden
    assert ev.legal is False
    assert ev.correcta is False
    assert "no es legal" in ev.feedback.lower()


def test_evaluate_answer_basura_es_ilegal():
    ev = evaluate_answer(START_FEN, "zzz")
    assert ev.legal is False
    assert ev.correcta is False


def test_evaluate_answer_fen_invalido():
    ev = evaluate_answer("fen-basura", "e4")
    assert ev.legal is False
    assert ev.correcta is False


def test_evaluate_answer_correcta_con_esperada():
    ev = evaluate_answer(START_FEN, "e4", jugada_esperada="e4")
    assert ev.legal is True
    assert ev.correcta is True
    assert ev.jugada_esperada == "e4"
    assert ev.jugada_alumno_san == "e4"
    assert "correcto" in ev.feedback.lower()


def test_evaluate_answer_correcta_normaliza_uci_a_san():
    # Alumno manda UCI, esperada en SAN: debe normalizar y comparar
    ev = evaluate_answer(START_FEN, "e2e4", jugada_esperada="e4")
    assert ev.correcta is True
    assert ev.jugada_alumno_san == "e4"


def test_evaluate_answer_incorrecta_con_esperada():
    ev = evaluate_answer(START_FEN, "e4", jugada_esperada="d4")
    assert ev.legal is True
    assert ev.correcta is False
    assert "esperada" in ev.feedback.lower()


def test_evaluate_answer_reporta_mate():
    ev = evaluate_answer(MATE_EN_1_FEN, "Qg7")
    assert ev.legal is True
    assert "mate" in ev.feedback.lower()


# ---------- schemas de entrada ----------


def test_generate_input_valida_rangos():
    GenerateExerciseInput(tema="clavada", tomo=2)  # ok
    with pytest.raises(Exception):
        GenerateExerciseInput(tema="x", tomo=5)
    with pytest.raises(Exception):
        GenerateExerciseInput(tema="x", tomo=0)


def test_evaluate_input_requiere_fen_y_jugada():
    EvaluateAnswerInput(fen=START_FEN, jugada_alumno="e4")  # ok
    with pytest.raises(Exception):
        EvaluateAnswerInput(jugada_alumno="e4")  # falta fen
    with pytest.raises(Exception):
        EvaluateAnswerInput(fen=START_FEN)  # falta jugada


# ---------- build_exercise_gen_tools ----------


def test_build_tools_devuelve_dos():
    retriever = _retriever_with([])
    tools = build_exercise_gen_tools(retriever)
    names = [t.name for t in tools]
    assert set(names) == {"generate_exercise", "evaluate_answer"}


def test_tool_generate_formato_string():
    retriever = _retriever_with([_make_item("tomo2-1", fen=START_FEN)])
    tools = {t.name: t for t in build_exercise_gen_tools(retriever)}
    out = tools["generate_exercise"].invoke({"tema": "aperturas"})
    assert isinstance(out, str)
    assert "FEN:" in out
    assert "Tomo 2" in out
    assert "tomo2-1" in out
    assert "Contexto de Grau" in out


def test_tool_generate_sin_resultados():
    retriever = _retriever_with([_make_item("tomo2-1", fen="")])
    tools = {t.name: t for t in build_exercise_gen_tools(retriever)}
    out = tools["generate_exercise"].invoke({"tema": "x"})
    assert "No encontré" in out


def test_tool_evaluate_ok():
    retriever = _retriever_with([])
    tools = {t.name: t for t in build_exercise_gen_tools(retriever)}
    out = tools["evaluate_answer"].invoke(
        {"fen": START_FEN, "jugada_alumno": "e4", "jugada_esperada": "e4"}
    )
    assert out.startswith("[OK]")
    assert "e4" in out


def test_tool_evaluate_error():
    retriever = _retriever_with([])
    tools = {t.name: t for t in build_exercise_gen_tools(retriever)}
    out = tools["evaluate_answer"].invoke(
        {"fen": START_FEN, "jugada_alumno": "e5"}
    )
    assert out.startswith("[ERROR]")


def test_tool_evaluate_abierta_sin_esperada():
    retriever = _retriever_with([])
    tools = {t.name: t for t in build_exercise_gen_tools(retriever)}
    out = tools["evaluate_answer"].invoke(
        {"fen": START_FEN, "jugada_alumno": "e4"}
    )
    assert out.startswith("[ABIERTA]")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
