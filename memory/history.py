from __future__ import annotations
from datetime import datetime, timezone

from contracts.progreso import HistorialConversacion
from memory.database import get_connection, init_db


def add_message(student_id: str, thread_id: str, role: str, content: str) -> None:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO historial_conversacion
                (student_id, thread_id, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (student_id, thread_id, role, content, now),
        )


def get_history(
    student_id: str,
    thread_id: str,
    limit: int = 20,
) -> list[HistorialConversacion]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM historial_conversacion
            WHERE student_id = ? AND thread_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (student_id, thread_id, limit),
        ).fetchall()
    return [
        HistorialConversacion(
            id=r["id"],
            student_id=r["student_id"],
            thread_id=r["thread_id"],
            role=r["role"],
            content=r["content"],
            timestamp=datetime.fromisoformat(r["timestamp"]),
        )
        for r in reversed(rows)
    ]


def clear_history(student_id: str, thread_id: str) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM historial_conversacion WHERE student_id = ? AND thread_id = ?",
            (student_id, thread_id),
        )
