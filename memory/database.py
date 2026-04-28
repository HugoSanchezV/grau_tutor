from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from core.config import settings


_SCHEMA = """
CREATE TABLE IF NOT EXISTS progreso_alumno (
    student_id TEXT NOT NULL,
    tema       TEXT NOT NULL,
    consultas  INTEGER NOT NULL DEFAULT 0,
    ejercicios_intentados INTEGER NOT NULL DEFAULT 0,
    ejercicios_correctos  INTEGER NOT NULL DEFAULT 0,
    ultima_actividad TEXT NOT NULL,
    PRIMARY KEY (student_id, tema)
);

CREATE TABLE IF NOT EXISTS historial_conversacion (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    thread_id  TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    timestamp  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ejercicios_usados (
    student_id  TEXT NOT NULL,
    partida_id  TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    PRIMARY KEY (student_id, partida_id)
);

CREATE TABLE IF NOT EXISTS ejercicios_disputados (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  TEXT NOT NULL,
    partida_id  TEXT NOT NULL,
    fen         TEXT NOT NULL,
    jugada_alumno TEXT NOT NULL,
    jugada_esperada TEXT NOT NULL,
    razonamiento TEXT,
    timestamp_disputa TEXT NOT NULL,
    estado      TEXT DEFAULT 'en_disputa',
    resultado_arbitro TEXT
);

CREATE INDEX IF NOT EXISTS idx_historial_student_thread
    ON historial_conversacion (student_id, thread_id);
CREATE INDEX IF NOT EXISTS idx_disputas_student
    ON ejercicios_disputados (student_id);
"""


def _db_path() -> str:
    return settings.sqlite_db_path


def init_db(db_path: str | None = None) -> None:
    path = db_path or _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)


@contextmanager
def get_connection(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
