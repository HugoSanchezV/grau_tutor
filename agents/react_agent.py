"""Agente ReAct: orquesta search_grau + chess_engine + exercise_gen con memoria de conversación.

Usa `langgraph.prebuilt.create_react_agent` como motor ReAct + un SqliteSaver
para mantener el hilo de la conversación de forma persistente entre reinicios.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from agents.tools.chess_engine import build_chess_engine_tools
from agents.tools.exercise_gen import build_exercise_gen_tools
from agents.tools.search_grau import SearchKConfig, build_search_grau_tool
from core.checkpointer import make_checkpointer
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


def build_tools(
    retriever: GrauRetriever,
    k_config: Optional[SearchKConfig] = None,
) -> list[BaseTool]:
    """Ensambla las 7 tools del agente (1 search + 4 motor + 2 ejercicios)."""
    tools: list[BaseTool] = [build_search_grau_tool(retriever, k_config=k_config)]
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
                    "content": content,
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
    """Envuelve el grafo ReAct de LangGraph como agente stateless.

    El estado del agente es administrado por el grafo LangGraph externo;
    este agente simplemente ejecuta el ReAct sin persistencia propia.
    """

    def __init__(
        self,
        retriever: GrauRetriever,
        llm: Optional[BaseChatModel] = None,
        system_prompt: str = SYSTEM_PROMPT,
        stateless: bool = True,
        k_config: Optional[SearchKConfig] = None,
    ) -> None:
        self.llm = llm or get_llm()
        self.k_config = k_config or SearchKConfig()
        self.tools = build_tools(retriever, k_config=self.k_config)
        self.system_prompt = system_prompt

        # Solo crear checkpointer si no es stateless (para uso standalone del agente)
        checkpointer = None if stateless else make_checkpointer("agent_checkpoints.db")

        self.graph = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=system_prompt,
            checkpointer=checkpointer,
        )

    def chat(
        self,
        message: str,
        thread_id: str = "default",
        history: Optional[list[Any]] = None,
    ) -> AgentResponse:
        """Invoca el agente ReAct. Si history se proporciona, la usa como contexto previo.

        Cuando se usa en el contexto de un grafo externo (como TutorGraph),
        pasar history = state["messages"] para que el agente tenga contexto del flujo.
        """
        # Si no hay historial explícito y tenemos checkpointer, usar el flujo con thread_id
        if history is None and self.graph.checkpointer is not None:
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

        # Agente stateless: usar historia proporcionada explícitamente
        if history is None:
            history = []
        input_messages = list(history) + [HumanMessage(content=message)]
        result = self.graph.invoke(
            {"messages": input_messages},
            config=None,
        )
        messages = result.get("messages", [])
        # Devolver solo los nuevos mensajes (los que se añadieron en esta invocación)
        new_messages = messages[len(history) :]
        return AgentResponse(
            reply=_extract_final_reply(new_messages),
            reasoning=_extract_reasoning(new_messages),
        )

    def stream(self, message: str, thread_id: str = "default") -> Iterable[dict]:
        """Stream de estados del grafo (para pintar la cadena de pensamiento en vivo).

        Solo funciona si el agente tiene checkpointer (modo stateful).
        """
        if self.graph.checkpointer is None:
            logger.warning("stream() llamado en agente stateless; necesita checkpointer")
            return

        config = {"configurable": {"thread_id": thread_id}}
        yield from self.graph.stream(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            stream_mode="values",
        )

    def reset(self, thread_id: str = "default") -> None:
        """Descarta la memoria del hilo si el agente tiene checkpointer.

        En modo stateless, esto es un no-op.
        """
        if self.graph.checkpointer is None:
            return  # Agente stateless, sin memoria para limpiar

        if hasattr(self.graph.checkpointer, "delete_thread"):
            self.graph.checkpointer.delete_thread(thread_id)
            return
        # Fallback para versiones de LangGraph sin delete_thread: borrado quirúrgico
        for attr in ("storage", "writes"):
            store = getattr(self.graph.checkpointer, attr, None)
            if store is None:
                continue
            for key in [k for k in list(store) if isinstance(k, tuple) and k[0] == thread_id]:
                del store[key]
