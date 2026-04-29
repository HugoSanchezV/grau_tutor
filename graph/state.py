from __future__ import annotations
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class TutorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str                      # ID del hilo externo (sincroniza memoria del agente)
    student_id: str
    mode: str                          # "tutor" | "evaluador"
    current_fen: Optional[str]         # FEN del ejercicio activo
    expected_move: Optional[str]       # Jugada correcta esperada (si existe)
    evaluation_reasoning: Optional[str]  # Texto del evaluador (para HITL)
    hitl_pending: bool                 # True cuando se necesita confirmación humana
    hitl_decision: Optional[str]       # "acepto" | "disputo" (post-HITL)
    reasoning_trace: list[dict]        # Cadena de pensamiento para la UI
    progress_summary: str              # Resumen de progreso del alumno
