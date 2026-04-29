from __future__ import annotations
from datetime import datetime, timezone

from contracts.progreso import ProgresoAlumno
from memory.database import get_connection, init_db


def upsert_progress(
    student_id: str,
    tema: str,
    delta_consultas: int = 0,
    delta_ejercicios_intentados: int = 0,
    delta_ejercicios_correctos: int = 0,
) -> None:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO progreso_alumno
                (student_id, tema, consultas, ejercicios_intentados,
                 ejercicios_correctos, ultima_actividad)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(student_id, tema) DO UPDATE SET
                consultas             = consultas + excluded.consultas,
                ejercicios_intentados = ejercicios_intentados + excluded.ejercicios_intentados,
                ejercicios_correctos  = ejercicios_correctos  + excluded.ejercicios_correctos,
                ultima_actividad      = excluded.ultima_actividad
            """,
            (
                student_id, tema,
                delta_consultas,
                delta_ejercicios_intentados,
                delta_ejercicios_correctos,
                now,
            ),
        )


def get_progress(student_id: str) -> list[ProgresoAlumno]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM progreso_alumno
            WHERE student_id = ?
            ORDER BY ultima_actividad DESC
            """,
            (student_id,),
        ).fetchall()
    return [
        ProgresoAlumno(
            student_id=r["student_id"],
            tema=r["tema"],
            consultas=r["consultas"],
            ejercicios_intentados=r["ejercicios_intentados"],
            ejercicios_correctos=r["ejercicios_correctos"],
            ultima_actividad=datetime.fromisoformat(r["ultima_actividad"]),
        )
        for r in rows
    ]


def get_progress_summary(student_id: str) -> str:
    rows = get_progress(student_id)
    if not rows:
        return "Sin progreso registrado todavía."
    lines = [f"Progreso de '{student_id}':"]
    for p in rows[:10]:
        if p.ejercicios_intentados > 0:
            tasa = f"{p.ejercicios_correctos}/{p.ejercicios_intentados} ejercicios"
        else:
            tasa = "sin ejercicios"
        lines.append(f"  {p.tema}: {p.consultas} consultas, {tasa}")
    return "\n".join(lines)
