"""Construcción y compilación del grafo LangGraph multiagente."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from agents.react_agent import GrauAgent
from core.checkpointer import make_checkpointer
from core.logging import get_logger
from graph.nodes import evaluador_node, hitl_review_node, router_node, tutor_node
from graph.state import TutorState
from rag.retrieval import GrauRetriever

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Respuesta pública
# ---------------------------------------------------------------------------

@dataclass
class GraphResponse:
    reply: str
    reasoning_trace: list[dict] = field(default_factory=list)
    current_fen: Optional[str] = None
    hitl_pending: bool = False
    progress_summary: str = ""
    evaluation_reasoning: Optional[str] = None


# ---------------------------------------------------------------------------
# Routing condicional
# ---------------------------------------------------------------------------

def _route_from_router(state: TutorState) -> str:
    return state.get("mode", "tutor")


def _route_from_evaluador(state: TutorState) -> str:
    if state.get("hitl_pending"):
        return "hitl_review"
    return END


# ---------------------------------------------------------------------------
# TutorGraph
# ---------------------------------------------------------------------------

class TutorGraph:
    """Grafo LangGraph que orquesta Tutor ↔ Evaluador con HITL y memoria SQLite."""

    def __init__(self, retriever: GrauRetriever) -> None:
        self.retriever = retriever
        # Crear agente en modo stateless: todo el estado lo maneja el grafo externo
        self.agent = GrauAgent(retriever=retriever, stateless=True)
        self.checkpointer = make_checkpointer("graph_checkpoints.db")
        self._graph = self._build()

    @property
    def k_config(self):
        """Acceso directo al holder mutable del k de search_grau."""
        return self.agent.k_config

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self):
        agent = self.agent
        retriever = self.retriever

        def _tutor(state: TutorState) -> dict:
            return tutor_node(state, agent)

        def _evaluador(state: TutorState) -> dict:
            return evaluador_node(state, retriever)

        builder = StateGraph(TutorState)
        builder.add_node("router", router_node)
        builder.add_node("tutor", _tutor)
        builder.add_node("evaluador", _evaluador)
        builder.add_node("hitl_review", hitl_review_node)

        builder.add_edge(START, "router")
        builder.add_conditional_edges(
            "router",
            _route_from_router,
            {"tutor": "tutor", "evaluador": "evaluador"},
        )
        builder.add_edge("tutor", END)
        builder.add_conditional_edges(
            "evaluador",
            _route_from_evaluador,
            {"hitl_review": "hitl_review", END: END},
        )
        builder.add_edge("hitl_review", END)

        return builder.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["hitl_review"],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _extract_reply(self, result: dict) -> str:
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                content = msg.content
                if isinstance(content, list):
                    return "\n".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ).strip()
                return str(content).strip()
        return ""

    def _to_response(self, result: dict, thread_id: str) -> GraphResponse:
        return GraphResponse(
            reply=self._extract_reply(result),
            reasoning_trace=result.get("reasoning_trace", []),
            current_fen=result.get("current_fen"),
            hitl_pending=self.is_interrupted(thread_id),
            progress_summary=result.get("progress_summary", ""),
            evaluation_reasoning=result.get("evaluation_reasoning"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        thread_id: str = "default",
        student_id: str = "alumno",
    ) -> GraphResponse:
        config = self._config(thread_id)
        existing = self._graph.get_state(config)

        if existing.values:
            input_data: dict = {
                "messages": [HumanMessage(content=message)],
                "student_id": student_id,
                "thread_id": thread_id,  # Bug #2: siempre actualizar para sincronizar agente
            }
        else:
            input_data = {
                "messages": [HumanMessage(content=message)],
                "student_id": student_id,
                "thread_id": thread_id,  # Bug #2: inicializar thread_id en el estado
                "mode": "tutor",
                "current_fen": None,
                "expected_move": None,
                "evaluation_reasoning": None,
                "hitl_pending": False,
                "hitl_decision": None,
                "reasoning_trace": [],
                "progress_summary": "",
            }

        result = self._graph.invoke(input_data, config=config)
        return self._to_response(result, thread_id)

    def is_interrupted(self, thread_id: str) -> bool:
        state = self._graph.get_state(self._config(thread_id))
        return bool(state.next)

    def resume_hitl(self, thread_id: str, decision: str) -> GraphResponse:
        """Reanuda el grafo tras la decisión HITL del alumno."""
        config = self._config(thread_id)
        self._graph.update_state(config, {"hitl_decision": decision})
        result = self._graph.invoke(None, config=config)
        return self._to_response(result, thread_id)

    def reset(self, thread_id: str) -> None:
        """Limpia el estado del hilo en el grafo (único checkpointer)."""
        if hasattr(self.checkpointer, "delete_thread"):
            self.checkpointer.delete_thread(thread_id)
        else:
            for attr in ("storage", "writes"):
                store = getattr(self.checkpointer, attr, None)
                if store is None:
                    continue
                for key in [k for k in list(store) if isinstance(k, tuple) and k[0] == thread_id]:
                    del store[key]
        # El agente no tiene checkpointer propio en modo stateless, nada más que limpiar
        self.agent.reset(thread_id)  # No-op en modo stateless


def build_graph(retriever: GrauRetriever) -> TutorGraph:
    return TutorGraph(retriever=retriever)
