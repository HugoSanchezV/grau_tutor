"""Tests del agente ReAct (mockea el grafo de LangGraph y el retriever)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.react_agent import (
    AgentResponse,
    GrauAgent,
    SYSTEM_PROMPT,
    _extract_final_reply,
    _extract_reasoning,
    build_tools,
)


def _retriever() -> MagicMock:
    mock = MagicMock()
    mock.retrieve_raw.return_value = []
    return mock


# ---------- build_tools ----------


def test_build_tools_devuelve_las_siete():
    tools = build_tools(_retriever())
    names = [t.name for t in tools]
    assert set(names) == {
        "search_grau",
        "validate_move",
        "apply_move",
        "list_legal_moves",
        "analyze_position",
        "generate_exercise",
        "evaluate_answer",
    }
    assert len(tools) == 7


def test_build_tools_todas_tienen_descripcion_y_schema():
    tools = build_tools(_retriever())
    for t in tools:
        assert t.description and len(t.description) > 10
        assert t.args_schema is not None


# ---------- _extract_reasoning ----------


def test_extract_reasoning_vacio_sin_tool_calls():
    msgs = [
        HumanMessage(content="hola"),
        AIMessage(content="hola alumno"),
    ]
    assert _extract_reasoning(msgs) == []


def test_extract_reasoning_captura_tool_call_y_resultado():
    ai_with_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_grau", "args": {"query": "clavada", "k": 3}, "id": "call_1"}
        ],
    )
    tool_result = ToolMessage(content="Fuente 1 — Tomo 2...", tool_call_id="call_1", name="search_grau")
    final = AIMessage(content="Según Grau...")
    trace = _extract_reasoning([HumanMessage(content="qué es clavada"), ai_with_call, tool_result, final])
    assert len(trace) == 2
    assert trace[0] == {"type": "tool_call", "name": "search_grau", "args": {"query": "clavada", "k": 3}}
    assert trace[1]["type"] == "tool_result"
    assert trace[1]["name"] == "search_grau"
    assert "Fuente 1" in trace[1]["content"]


def test_extract_reasoning_preserva_contenido_completo():
    long_content = "x" * 1000
    tool_result = ToolMessage(content=long_content, tool_call_id="c", name="search_grau")
    trace = _extract_reasoning([tool_result])
    assert len(trace[0]["content"]) == 1000


def test_extract_reasoning_varios_tool_calls_en_un_mensaje():
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "validate_move", "args": {"fen": "x", "move": "e4"}, "id": "1"},
            {"name": "analyze_position", "args": {"fen": "x"}, "id": "2"},
        ],
    )
    trace = _extract_reasoning([ai])
    assert len(trace) == 2
    assert trace[0]["name"] == "validate_move"
    assert trace[1]["name"] == "analyze_position"


# ---------- _extract_final_reply ----------


def test_extract_final_reply_string_simple():
    msgs = [HumanMessage(content="hi"), AIMessage(content="hola alumno")]
    assert _extract_final_reply(msgs) == "hola alumno"


def test_extract_final_reply_ignora_ai_con_tool_calls():
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}]),
        ToolMessage(content="res", tool_call_id="1", name="x"),
        AIMessage(content="respuesta final"),
    ]
    assert _extract_final_reply(msgs) == "respuesta final"


def test_extract_final_reply_bloques_anthropic():
    msgs = [
        AIMessage(content=[{"type": "text", "text": "parte 1"}, {"type": "text", "text": "parte 2"}])
    ]
    out = _extract_final_reply(msgs)
    assert "parte 1" in out and "parte 2" in out


def test_extract_final_reply_sin_ai_devuelve_vacio():
    assert _extract_final_reply([HumanMessage(content="hola")]) == ""


# ---------- GrauAgent (grafo mockeado) ----------


@patch("agents.react_agent.create_react_agent")
def test_grau_agent_construye_con_llm_inyectado(mock_create):
    mock_create.return_value = MagicMock()
    agent = GrauAgent(retriever=_retriever(), llm=MagicMock())
    assert len(agent.tools) == 7
    assert agent.graph is mock_create.return_value
    mock_create.assert_called_once()
    kwargs = mock_create.call_args.kwargs
    assert kwargs["prompt"] == SYSTEM_PROMPT
    assert kwargs["checkpointer"] is agent.checkpointer


@patch("agents.react_agent.create_react_agent")
def test_grau_agent_chat_devuelve_reply_y_reasoning(mock_create):
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {
        "messages": [
            HumanMessage(content="qué es clavada"),
            AIMessage(content="", tool_calls=[{"name": "search_grau", "args": {"query": "clavada"}, "id": "1"}]),
            ToolMessage(content="Fuente 1...", tool_call_id="1", name="search_grau"),
            AIMessage(content="Una clavada es..."),
        ]
    }
    mock_create.return_value = fake_graph

    agent = GrauAgent(retriever=_retriever(), llm=MagicMock())
    out = agent.chat("qué es clavada")

    assert isinstance(out, AgentResponse)
    assert out.reply == "Una clavada es..."
    assert len(out.reasoning) == 2
    assert out.reasoning[0]["type"] == "tool_call"
    assert out.reasoning[1]["type"] == "tool_result"
    fake_graph.invoke.assert_called_once()
    invoke_kwargs = fake_graph.invoke.call_args
    # thread_id por defecto
    assert invoke_kwargs.kwargs["config"] == {"configurable": {"thread_id": "default"}}


@patch("agents.react_agent.create_react_agent")
def test_grau_agent_chat_respeta_thread_id(mock_create):
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"messages": [AIMessage(content="ok")]}
    mock_create.return_value = fake_graph

    agent = GrauAgent(retriever=_retriever(), llm=MagicMock())
    agent.chat("hola", thread_id="alumno-42")

    cfg = fake_graph.invoke.call_args.kwargs["config"]
    assert cfg == {"configurable": {"thread_id": "alumno-42"}}


@patch("agents.react_agent.create_react_agent")
def test_grau_agent_chat_sin_messages_devuelve_reply_vacio(mock_create):
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {}
    mock_create.return_value = fake_graph

    agent = GrauAgent(retriever=_retriever(), llm=MagicMock())
    out = agent.chat("hola")
    assert out.reply == ""
    assert out.reasoning == []


@patch("agents.react_agent.create_react_agent")
def test_grau_agent_stream_yield_chunks(mock_create):
    fake_graph = MagicMock()
    fake_graph.stream.return_value = iter([{"messages": [AIMessage(content="paso1")]}, {"messages": [AIMessage(content="paso2")]}])
    mock_create.return_value = fake_graph

    agent = GrauAgent(retriever=_retriever(), llm=MagicMock())
    chunks = list(agent.stream("hola", thread_id="t1"))
    assert len(chunks) == 2
    stream_kwargs = fake_graph.stream.call_args.kwargs
    assert stream_kwargs["config"] == {"configurable": {"thread_id": "t1"}}
    assert stream_kwargs["stream_mode"] == "values"


@patch("agents.react_agent.create_react_agent")
def test_grau_agent_memoria_entre_turnos_mismo_thread(mock_create):
    """El agente pasa SIEMPRE el mismo checkpointer al grafo: LangGraph se ocupa del resto."""
    mock_create.return_value = MagicMock()
    agent = GrauAgent(retriever=_retriever(), llm=MagicMock())
    assert mock_create.call_args.kwargs["checkpointer"] is agent.checkpointer


# ---------- SYSTEM_PROMPT sanity ----------


def test_system_prompt_menciona_tools_clave():
    p = SYSTEM_PROMPT.lower()
    for name in ["search_grau", "generate_exercise", "evaluate_answer", "validate_move"]:
        assert name in p, f"el prompt debería mencionar {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
