"""Agente ReAct: orquesta search_grau + chess_engine + exercise_gen con memoria de conversación.

Usa `langgraph.prebuilt.create_react_agent` como motor ReAct + un `MemorySaver`
para mantener el hilo de la conversación (incluye el FEN del último ejercicio,
que el LLM recuerda vía historial).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from agents.tools.chess_engine import build_chess_engine_tools
from agents.tools.exercise_gen import build_exercise_gen_tools
from agents.tools.search_grau import build_search_grau_tool
from core.llm import get_llm
from core.logging import get_logger
from rag.retrieval import GrauRetriever

logger = get_logger(__name__)


SYSTEM_PROMPT = """Eres el Tutor Grau, un mentor de ajedrez que enseña basándose
en el "Tratado General de Ajedrez" de Roberto Grau. Tu pedagogía es paciente,
socrática y rigurosa.

REGLAS OBLIGATORIAS — sin excepción:
1. SIEMPRE llama a search_grau antes de responder cualquier pregunta de ajedrez.
   Nunca respondas desde tu conocimiento general: solo el corpus de Grau cuenta.
2. Nunca afirmes que una jugada es correcta sin validarla con validate_move o apply_move.
3. Si el alumno pide un ejercicio o ejemplo, llama a generate_exercise.
   Si la herramienta indica que no hay posición disponible, díselo al alumno
   y ofrece buscar pasajes sobre el tema con search_grau en su lugar.
4. Si el alumno responde a un ejercicio, reutiliza el FEN del turno anterior
   que aparece en el historial — no regeneres un ejercicio nuevo.
5. Cita siempre la fuente (tomo, partida, ECO si aplica) de lo que recupera search_grau.
6. Responde en español, conciso pero pedagógico. Prefiere una pregunta que invite
   al alumno a pensar en lugar de dar la respuesta directamente.

Herramientas disponibles:
- search_grau: OBLIGATORIO para preguntas conceptuales. Cita la fuente.
- validate_move / apply_move / list_legal_moves / analyze_position: motor de ajedrez.
- generate_exercise: genera ejercicio con FEN para que el alumno practique.
- evaluate_answer: evalúa la jugada del alumno sobre el FEN del ejercicio.
"""


@dataclass
class AgentResponse:
    reply: str
    reasoning: list[dict] = field(default_factory=list)


def build_tools(retriever: GrauRetriever) -> list[BaseTool]:
    """Ensambla las 7 tools del agente (1 search + 4 motor + 2 ejercicios)."""
    tools: list[BaseTool] = [build_search_grau_tool(retriever)]
    tools.extend(build_chess_engine_tools())
    tools.extend(build_exercise_gen_tools(retriever))
    return tools


def _extract_reasoning(messages: list[Any]) -> list[dict]:
    """Extrae la traza de tool calls y resultados para mostrar como cadena de pensamiento."""
    trace: list[dict] = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                trace.append(
                    {
                        "type": "tool_call",
                        "name": tc.get("name", "?"),
                        "args": tc.get("args", {}),
                    }
                )
        elif isinstance(m, ToolMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            trace.append(
                {
                    "type": "tool_result",
                    "name": getattr(m, "name", "?"),
                    "content": content[:500],
                }
            )
    return trace


def _extract_final_reply(messages: list[Any]) -> str:
    """Devuelve el texto del último AIMessage sin tool_calls (la respuesta al alumno)."""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            content = m.content
            if isinstance(content, list):
                parts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                return "\n".join(p for p in parts if p).strip()
            return str(content).strip()
    return ""


class GrauAgent:
    """Envuelve el grafo ReAct de LangGraph con memoria por `thread_id`."""

    def __init__(
        self,
        retriever: GrauRetriever,
        llm: Optional[BaseChatModel] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm or get_llm()
        self.tools = build_tools(retriever)
        self.checkpointer = MemorySaver()
        self.graph = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=system_prompt,
            checkpointer=self.checkpointer,
        )

    def chat(self, message: str, thread_id: str = "default") -> AgentResponse:
        config = {"configurable": {"thread_id": thread_id}}
        state = self.graph.get_state(config)
        prev_count = len(state.values.get("messages", [])) if state.values else 0
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )
        messages = result.get("messages", [])
        return AgentResponse(
            reply=_extract_final_reply(messages),
            reasoning=_extract_reasoning(messages[prev_count:]),
        )

    def stream(self, message: str, thread_id: str = "default") -> Iterable[dict]:
        """Stream de estados del grafo (para pintar la cadena de pensamiento en vivo)."""
        config = {"configurable": {"thread_id": thread_id}}
        yield from self.graph.stream(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            stream_mode="values",
        )

    def reset(self, thread_id: str = "default") -> None:
        """Descarta la memoria del hilo (útil al pulsar 'nueva conversación' en la UI)."""
        try:
            self.checkpointer.delete_thread(thread_id)
        except AttributeError:
            self.checkpointer = MemorySaver()
            self.graph = create_react_agent(
                model=self.llm,
                tools=self.tools,
                prompt=SYSTEM_PROMPT,
                checkpointer=self.checkpointer,
            )
