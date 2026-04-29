import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from app.components.board import extract_fen, render_board_panel
from app.components.progress import render_progress_panel
from core.logging import get_logger
from graph.graph import GraphResponse, TutorGraph, build_graph
from rag.retrieval import GrauRetriever
from rag.store import get_chroma_client, get_or_create_collection

load_dotenv()
logger = get_logger(__name__)

st.set_page_config(page_title="Chess Tutor Grau", page_icon="🎓", layout="wide")
st.title("♟️ Chess Tutor Grau")
st.subheader("Tu mentor basado en el Tratado General de Ajedrez de Roberto Grau")

_WELCOME = (
    "¡Hola! Soy tu Tutor Grau. ¿Sobre qué concepto de ajedrez tienes dudas, "
    "o prefieres que te proponga un ejercicio?"
)

# ---------------------------------------------------------------------------
# Inicialización
# ---------------------------------------------------------------------------

@st.cache_resource
def init_graph() -> TutorGraph | None:
    logger.info("Inicializando TutorGraph…")
    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        retriever = GrauRetriever(collection)
        return build_graph(retriever)
    except Exception as exc:
        logger.error(f"Error al inicializar TutorGraph: {exc}")
        return None


graph = init_graph()

if graph is None:
    st.error(
        "No se pudo arrancar el sistema. "
        "Verifica que ChromaDB esté disponible y que las API keys estén en .env."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------

st.session_state.setdefault("messages", [{"role": "assistant", "content": _WELCOME}])
st.session_state.setdefault("current_fen", None)
st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("student_id", "alumno")
st.session_state.setdefault("hitl_pending", False)
st.session_state.setdefault("hitl_reasoning", None)
st.session_state.setdefault("reasoning_trace", [])
st.session_state.setdefault("search_k", graph.k_config.default_k)

thread_id: str = st.session_state["thread_id"]
student_id: str = st.session_state["student_id"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_reasoning(trace: list[dict]) -> None:
    for step in trace:
        if step["type"] == "tool_call":
            st.code(
                f"Herramienta: {step['name']}\nArgumentos: {step['args']}",
                language="json",
            )
        elif step["type"] == "tool_result":
            with st.expander(f"Resultado: {step['name']}"):
                st.text(step["content"])


def _apply_response(response: GraphResponse, *, append_to_chat: bool = True) -> None:
    if append_to_chat and response.reply:
        st.session_state["messages"].append(
            {"role": "assistant", "content": response.reply}
        )

    # Tablero:l FEN explícito del grafo primero, uego extraer del texto
    new_fen = response.current_fen or extract_fen(response.reply or "")
    if not new_fen:
        for step in response.reasoning_trace:
            if step["type"] == "tool_result":
                new_fen = extract_fen(step["content"])
                if new_fen:
                    break
    if new_fen:
        st.session_state["current_fen"] = new_fen

    st.session_state["hitl_pending"] = response.hitl_pending
    st.session_state["hitl_reasoning"] = response.evaluation_reasoning
    st.session_state["reasoning_trace"] = response.reasoning_trace


# ---------------------------------------------------------------------------
# Layout — columnas
# ---------------------------------------------------------------------------

col_chat, col_side = st.columns([2, 1])

# ---- Panel lateral --------------------------------------------------------
with col_side:
    if st.button("Nueva conversación", use_container_width=True):
        graph.reset(thread_id)
        st.session_state.update(
            messages=[{"role": "assistant", "content": _WELCOME}],
            current_fen=None,
            thread_id=str(uuid.uuid4()),
            hitl_pending=False,
            hitl_reasoning=None,
            reasoning_trace=[],
        )
        st.rerun()

    with st.container(border=True):
        st.markdown("### 🔍 Búsqueda")
        st.slider(
            "Fuentes por consulta (k)",
            min_value=1,
            max_value=15,
            key="search_k",
            help=(
                "Cuántos pasajes del corpus de Grau recibe el tutor por consulta. "
                "Más fuentes = más contexto pero respuestas más lentas."
            ),
        )

    with st.container(border=True):
        st.markdown("### ♟️ Posición")
        board_slot = st.empty()
        if st.session_state["current_fen"]:
            with board_slot.container():
                render_board_panel(st.session_state["current_fen"])
        else:
            board_slot.info("No hay posición activa todavía.")

    with st.container(border=True):
        st.markdown("### 📈 Progreso")
        render_progress_panel(student_id)

    with st.container(border=True):
        st.markdown("### 🧠 Cadena de Pensamiento")
        st.caption(f"Modelo: {os.getenv('LLM_MODEL', 'llama-3.3-70b-versatile')}")
        if st.session_state["reasoning_trace"]:
            _render_reasoning(st.session_state["reasoning_trace"])
        else:
            st.caption("Sin razonamiento todavía.")

# ---- Panel de chat --------------------------------------------------------
with col_chat:
    chat_panel = st.container(border=True)
    with chat_panel:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # --- HITL: confirmación humana ---
    if st.session_state["hitl_pending"]:
        st.warning(
            "**Revisión pendiente** — El Evaluador marcó tu jugada como incorrecta. "
            "¿Aceptas la evaluación o la disputas?"
        )
        if st.session_state["hitl_reasoning"]:
            with st.expander("Ver razonamiento del Evaluador"):
                st.markdown(st.session_state["hitl_reasoning"])

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            acepto = st.button("✅ Acepto la corrección", use_container_width=True)
        with btn_col2:
            disputo = st.button("❌ Disputo la evaluación", use_container_width=True)

        if acepto or disputo:
            decision = "acepto" if acepto else "disputo"
            with st.spinner("Procesando decisión…"):
                hitl_resp = graph.resume_hitl(thread_id, decision)
            _apply_response(hitl_resp)
            st.rerun()

    # --- Input del usuario ---
    if prompt := st.chat_input(
        "Escribe tu duda, jugada o pide un ejercicio…",
        disabled=st.session_state["hitl_pending"],
    ):
        st.session_state["messages"].append({"role": "user", "content": prompt})

        graph.k_config.default_k = int(st.session_state["search_k"])

        with chat_panel:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Reflexionando…"):
                    response = graph.chat(prompt, thread_id=thread_id, student_id=student_id)
                if response.reply:
                    st.markdown(response.reply)

        _apply_response(response)
        st.rerun()
