from __future__ import annotations
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ProgresoAlumno(BaseModel):
    student_id: str
    tema: str
    consultas: int = 0
    ejercicios_intentados: int = 0
    ejercicios_correctos: int = 0
    ultima_actividad: datetime = Field(default_factory=datetime.utcnow)


class HistorialConversacion(BaseModel):
    id: Optional[int] = None
    student_id: str
    thread_id: str
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
