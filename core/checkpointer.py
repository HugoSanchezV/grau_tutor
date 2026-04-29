from __future__ import annotations
import sqlite3
from pathlib import Path

from core.config import settings


def make_checkpointer(db_name: str):
    """Crea un SqliteSaver persistente; cae back a MemorySaver si no está disponible.

    Usar un archivo SQLite distinto al de la app evita conflictos de schema con
    las tablas de progreso e historial.
    """
    db_path = str(Path(settings.sqlite_db_path).parent / db_name)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
