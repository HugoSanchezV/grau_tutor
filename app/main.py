import streamlit as st
import os
import uuid
from dotenv import load_dotenv

from core.logging import get_logger
from rag.store import get_chroma_client, get_or_create_collection
from rag.retrieval import GrauRetriever
from agents.react_agent import GrauAgent
from app.components.board import extract_fen, render_board_panel

load_dotenv()
logger = get_logger(__name__)

st.set_page_config(page_title="Chess Tutor Grau", page_icon="🎓", layout="wide")

st.title("♟️ Chess Tutor Grau")
st.subheader("Tu mentor basado en el Tratado General de Ajedrez de Roberto Grau")

# --- Inicialización del Sistema ---
@st.cache_resource
def init_agent():
    logger.info("Inicializando cliente Chroma y Retriever...")
    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        retriever = GrauRetriever(collection)
        return GrauAgent(retriever=retriever)
    except Exception as e:
        logger.error(f"Error inicializando agente: {e}")
        return None

agent = init_agent()

if agent is None:
    st.error("Error al arrancar el agente. Verifica si ChromaDB está disponible en la red o si las credenciales en .env son correctas.")
    st.stop()

# --- Estado de la sesión ---
st.session_state.setdefault("messages", [
    {"role": "assistant", "content": "¡Hola! Soy tu Tutor Grau. ¿En qué concepto de ajedrez te puedo ayudar hoy o prefieres hacer un ejercicio?"}
])
st.session_state.setdefault("current_fen", None)
st.session_state.setdefault("thread_id", str(uuid.uuid4()))

thread_id = st.session_state["thread_id"]

# --- Layout: Chat principal vs Panel lateral ---
col_chat, col_side = st.columns([2, 1])

# --- Panel lateral (tablero + razonamiento) ---
with col_side:
    if st.button("Nueva conversación", use_container_width=True):
        agent.reset(thread_id)
        st.session_state["messages"] = [
            {"role": "assistant", "content": "¡Hola! Soy tu Tutor Grau. ¿En qué concepto de ajedrez te puedo ayudar hoy o prefieres hacer un ejercicio?"}
        ]
        st.session_state["current_fen"] = None
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.rerun()

    with st.container(border=True):
        st.markdown("### ♟️ Posición")
        board_container = st.empty()
        if st.session_state["current_fen"]:
            with board_container.container():
                render_board_panel(st.session_state["current_fen"])
        else:
            with board_container.container():
                st.info("No hay posición detectada en la conversación todavía.")

    with st.container(border=True):
        st.markdown("### 🧠 Cadena de Pensamiento")
        st.caption(f"Modelo: {os.getenv('LLM_MODEL', 'llama-3.3-70b-versatile')}")
        reasoning_container = st.empty()

# --- Historial del chat ---
with col_chat:
    chat_panel = st.container(border=True)
    with chat_panel:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

# --- Input fuera de columnas: se ancla al fondo de la página ---
if prompt := st.chat_input("Escribe tu duda, jugada o pide un ejercicio..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with chat_panel:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Reflexionando..."):
                response = agent.chat(prompt, thread_id=thread_id)
            st.markdown(response.reply)

    st.session_state["messages"].append({"role": "assistant", "content": response.reply})

    # Extraer FEN: primero en la respuesta final, si no en los tool results
    new_fen = extract_fen(response.reply)
    if not new_fen:
        for step in response.reasoning:
            if step["type"] == "tool_result":
                new_fen = extract_fen(step["content"])
                if new_fen:
                    break
    if new_fen:
        st.session_state["current_fen"] = new_fen
        board_container.empty()
        with board_container.container():
            render_board_panel(new_fen)

    # Mostrar razonamiento en el panel lateral
    if response.reasoning:
        with reasoning_container.container():
            for step in response.reasoning:
                if step["type"] == "tool_call":
                    st.code(f"Herramienta invocada: {step['name']}\nArgumentos: {step['args']}", language="json")
                elif step["type"] == "tool_result":
                    with st.expander(f"Resultado de la herramienta: {step['name']}"):
                        st.text(step["content"])