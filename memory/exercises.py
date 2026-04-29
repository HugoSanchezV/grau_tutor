from __future__ import annotations
from datetime import datetime, timezone

from memory.database import get_connection, init_db


def get_used_exercise_ids(student_id: str) -> set[str]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT partida_id FROM ejercicios_usados WHERE student_id = ?",
            (student_id,),
        ).fetchall()
    return {r["partida_id"] for r in rows}


def mark_exercise_used(student_id: str, partida_id: str) -> None:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO ejercicios_usados (student_id, partida_id, timestamp)
            VALUES (?, ?, ?)
            """,
            (student_id, partida_id, now),
        )
