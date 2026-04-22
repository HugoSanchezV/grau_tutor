from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class ChunkMetadata(BaseModel):
    tomo: int = Field(..., description="Número de tomo (1-4)")
    tema: str = Field(..., description="Tema del tomo (Rudimentos, Estrategia, etc.)")
    event: str = Field(default="?")
    white: str = Field(default="?")
    black: str = Field(default="?")
    result: str = Field(default="*")
    eco: str = Field(default="")
    annotator: str = Field(default="")
    fen: str = Field(default="", description="FEN inicial si existe")
    ply_count: int = Field(default=0)
    chunk_index: int = Field(default=0, description="Índice del chunk dentro de la partida")
    partida_id: str = Field(..., description="ID único: tomo-índice")


class PartidaGrau(BaseModel):
    partida_id: str = Field(..., description="ID único: tomo-{tomo}-{idx}")
    texto_completo: str = Field(..., description="Texto del chunk para embedding")
    jugadas: str = Field(default="", description="Jugadas en notación algebraica")
    comentarios: str = Field(default="", description="Comentarios pedagógicos de Grau")
    metadata: ChunkMetadata

    def to_chroma_document(self) -> dict:
        return {
            "id": self.partida_id,
            "document": self.texto_completo,
            "metadata": {
                "tomo": self.metadata.tomo,
                "tema": self.metadata.tema,
                "event": self.metadata.event,
                "white": self.metadata.white,
                "black": self.metadata.black,
                "result": self.metadata.result,
                "eco": self.metadata.eco,
                "annotator": self.metadata.annotator,
                "fen": self.metadata.fen,
                "ply_count": self.metadata.ply_count,
                "partida_id": self.partida_id,
                "jugadas": self.jugadas,
            },
        }
