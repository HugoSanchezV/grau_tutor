"""Tests del grafo LangGraph multiagente (sin llamadas reales al LLM ni a ChromaDB)."""
from __future__ import annotations
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from graph.state import TutorState
from graph.nodes import _looks_like_move, router_node, hitl_review_node


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> dict:
    base: dict = {
        "messages": [],
        "student_id": "test_student",
        "mode": "tutor",
        "current_fen": None,
        "expected_move": None,
        "evaluation_reasoning": None,
        "hitl_pending": False,
        "hitl_decision": None,
        "reasoning_trace": [],
        "progress_summary": "",
    }
    base.update(overrides)
    return base


def _human_msg(content: str):
    from langchain_core.messages import HumanMessage
    return HumanMessage(content=content)


# ---------------------------------------------------------------------------
# _looks_like_move
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("move,expected", [
    ("e4", True),
    ("Nf3", True),
    ("Bxc6", True),
    ("O-O", True),
    ("O-O-O", True),
    ("e8=Q", True),
    ("g1f3", True),        # UCI
    ("e2e4", True),        # UCI
    ("Explícame la clavada", False),
    ("Dame un ejercicio", False),
    ("¿Qué es el gambito de rey?", False),
    ("Rf8+", True),
])
def test_looks_like_move(move: str, expected: bool) -> None:
    assert _looks_like_move(move) == expected


# ---------------------------------------------------------------------------
# router_node — fast path (ejercicio activo + jugada)
# ---------------------------------------------------------------------------

def test_router_fast_path_move_with_fen() -> None:
    state = _make_state(
        messages=[_human_msg("Nf3")],
        current_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    )
    result = router_node(state)
    assert result["mode"] == "evaluador"


def test_router_fast_path_no_fen_calls_llm() -> None:
    """Sin FEN activo, el router llama al LLM aunque parezca jugada."""
    state = _make_state(
        messages=[_human_msg("Nf3")],
        current_fen=None,
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="tutor")

    with patch("graph.nodes.get_llm", return_value=mock_llm):
        result = router_node(state)

    mock_llm.invoke.assert_called_once()
    assert result["mode"] in ("tutor", "evaluador")


def test_router_llm_classifies_tutor() -> None:
    state = _make_state(messages=[_human_msg("¿Qué es la clavada absoluta?")])
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="tutor")

    with patch("graph.nodes.get_llm", return_value=mock_llm):
        result = router_node(state)

    assert result["mode"] == "tutor"


def test_router_llm_classifies_evaluador() -> None:
    state = _make_state(messages=[_human_msg("Dame un ejercicio de táctica")])
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="evaluador")

    with patch("graph.nodes.get_llm", return_value=mock_llm):
        result = router_node(state)

    assert result["mode"] == "evaluador"


# ---------------------------------------------------------------------------
# hitl_review_node
# ---------------------------------------------------------------------------

def test_hitl_review_acepto() -> None:
    state = _make_state(
        hitl_decision="acepto",
        evaluation_reasoning="La jugada correcta era Nf3, no Nc3.",
        student_id="test_student",
    )
    with patch("graph.nodes.get_progress_summary", return_value="sin progreso"):
        result = hitl_review_node(state)

    assert result["hitl_pending"] is False
    assert result["hitl_decision"] is None
    assert result["current_fen"] is None
    # El reply contiene la evaluación confirmada
    reply = result["messages"][0].content
    assert "confirmada" in reply.lower() or "nf3" in reply.lower()


def test_hitl_review_disputo() -> None:
    state = _make_state(
        hitl_decision="disputo",
        evaluation_reasoning="La jugada correcta era Nf3, no Nc3.",
        student_id="test_student",
    )
    with patch("graph.nodes.get_progress_summary", return_value="sin progreso"):
        result = hitl_review_node(state)

    assert result["hitl_pending"] is False
    reply = result["messages"][0].content
    assert "beneficio" in reply.lower() or "disputa" in reply.lower() or "legal" in reply.lower()


def test_hitl_review_default_acepto() -> None:
    """Sin hitl_decision explícito, se trata como acepto."""
    state = _make_state(hitl_decision=None, student_id="test_student")
    with patch("graph.nodes.get_progress_summary", return_value="sin progreso"):
        result = hitl_review_node(state)
    assert result["hitl_pending"] is False


# ---------------------------------------------------------------------------
# memory/database — init_db crea tablas
# ---------------------------------------------------------------------------

def test_init_db_creates_tables(tmp_path) -> None:
    import sqlite3
    from memory.database import init_db

    db_path = str(tmp_path / "test.db")
    init_db(db_path=db_path)

    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()

    assert "progreso_alumno" in tables
    assert "historial_conversacion" in tables


# ---------------------------------------------------------------------------
# memory/progress — upsert y get
# ---------------------------------------------------------------------------

def test_upsert_and_get_progress(tmp_path) -> None:
    from memory.database import init_db
    from memory.progress import get_progress, upsert_progress

    db_path = str(tmp_path / "test.db")
    init_db(db_path=db_path)

    with patch("memory.progress.get_connection") as mock_conn, \
         patch("memory.progress.init_db"):
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        mock_conn.return_value.__enter__ = lambda s: conn
        mock_conn.return_value.__exit__ = lambda s, *a: conn.commit() or False

        upsert_progress("alumno1", "táctica", delta_consultas=3)
        upsert_progress("alumno1", "táctica", delta_consultas=2, delta_ejercicios_intentados=1)

        rows = get_progress("alumno1")

    assert len(rows) == 1
    assert rows[0].tema == "táctica"
    assert rows[0].consultas == 5
    assert rows[0].ejercicios_intentados == 1


# ---------------------------------------------------------------------------
# memory/history — add y get
# ---------------------------------------------------------------------------

def test_add_and_get_history(tmp_path) -> None:
    import sqlite3
    from memory.database import init_db
    from memory.history import add_message, get_history

    db_path = str(tmp_path / "test.db")
    init_db(db_path=db_path)

    with patch("memory.history.get_connection") as mock_conn, \
         patch("memory.history.init_db"):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        mock_conn.return_value.__enter__ = lambda s: conn
        mock_conn.return_value.__exit__ = lambda s, *a: conn.commit() or False

        add_message("alumno1", "thread1", "user", "¿Qué es la clavada?")
        add_message("alumno1", "thread1", "assistant", "La clavada es…")

        history = get_history("alumno1", "thread1")

    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


# ---------------------------------------------------------------------------
# graph/graph — TutorGraph se puede instanciar con mock
# ---------------------------------------------------------------------------

def test_tutor_graph_builds() -> None:
    from graph.graph import TutorGraph

    mock_retriever = MagicMock()
    mock_agent = MagicMock()

    with patch("graph.graph.GrauAgent", return_value=mock_agent):
        tg = TutorGraph(retriever=mock_retriever)

    assert tg._graph is not None
    assert tg.checkpointer is not None


def test_tutor_graph_is_interrupted_false_on_new_thread() -> None:
    from graph.graph import TutorGraph

    mock_retriever = MagicMock()
    with patch("graph.graph.GrauAgent"):
        tg = TutorGraph(retriever=mock_retriever)

    assert tg.is_interrupted("nonexistent-thread") is False
