"""Panel de progreso del alumno para la UI de Streamlit."""
from __future__ import annotations

import streamlit as st

from memory.progress import get_progress


def render_progress_panel(student_id: str) -> None:
    """Muestra el progreso del alumno en el contenedor Streamlit activo."""
    rows = get_progress(student_id)

    if not rows:
        st.caption("Aún no hay actividad registrada.")
        return

    total_consultas = sum(r.consultas for r in rows)
    total_intentados = sum(r.ejercicios_intentados for r in rows)
    total_correctos = sum(r.ejercicios_correctos for r in rows)

    col1, col2, col3 = st.columns(3)
    col1.metric("Consultas", total_consultas)
    col2.metric("Ejercicios", total_intentados)
    if total_intentados > 0:
        pct = round(100 * total_correctos / total_intentados)
        col3.metric("Aciertos", f"{pct}%")
    else:
        col3.metric("Aciertos", "—")

    if len(rows) > 1:
        with st.expander("Desglose por tema"):
            for p in rows:
                if p.ejercicios_intentados > 0:
                    tasa = f"{p.ejercicios_correctos}/{p.ejercicios_intentados}"
                    label = f"**{p.tema}** — {p.consultas} consultas · {tasa} ejercicios"
                else:
                    label = f"**{p.tema}** — {p.consultas} consultas"
                st.markdown(label)
