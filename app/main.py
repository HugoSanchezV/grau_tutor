import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Chess Tutor Grau", page_icon="🎓", layout="wide")

st.title("♟️ Chess Tutor Grau")
st.subheader("Tu mentor basado en el Tratado General de Ajedrez")

# Sidebar para estado
with st.sidebar:
    st.success("Docker: Conectado")
    st.info(f"Modelo: {os.getenv('LLM_MODEL', 'llama3-70b-8192')}")

st.write("¡Bienvenido, Hugo! La infraestructura está lista. El siguiente paso es la ingesta de las partidas de Grau.")

if st.button("Probar conexión con ChromaDB"):
    # Aquí irá la lógica de búsqueda después
    st.write("Buscando latido de la base de datos...")