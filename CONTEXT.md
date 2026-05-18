# 🎓 PROYECTO FINAL — BOOTCAMP AI ENGINEER (CÓDIGO FACILITO)

## CONTEXTO DEL ARCHIVO
Este documento es el **contexto completo** del proyecto final del Bootcamp AI Engineer de Código Facilito. Debe usarse como referencia principal para cualquier decisión de diseño, implementación o arquitectura. Cualquier código, estructura o decisión que se tome debe alinearse con lo descrito aquí.

---

## 1. DESCRIPCIÓN DEL PROYECTO

### Nombre
**Chess Tutor Grau** — Tutor de Ajedrez basado en los libros de Roberto Grau

### Concepto
Sistema de tutoría inteligente de ajedrez que utiliza como base de conocimiento los libros "Tratado General de Ajedrez" de Roberto Grau. El sistema responde dudas sobre táctica, estrategia, aperturas y finales; genera ejercicios de práctica; y evalúa las respuestas del alumno.

### Dominio
Educación + Ajedrez. Los libros de Grau son clásicos pedagógicos del ajedrez en español, estructurados por temas con progresión didáctica natural.

### Fecha de entrega
**Lunes 18 de mayo de 2026**

### Modalidad
Individual.

---

## 2. FUENTE DE DATOS

### Archivos disponibles
Se dispone de **4 archivos PGN** con las partidas anotadas de los 4 tomos de Grau:

| Archivo | Tomo | Tema | Partidas |
|---------|------|------|----------|
| `Grau_I.pgn` | Tomo I | Rudimentos | 164 |
| `Grau_II.pgn` | Tomo II | Estrategia | 418 |
| `Grau_III.pgn` | Tomo III | Medio juego / Conformación de peones | 264 |
| `Grau_IV.pgn` | Tomo IV | Estrategia Superior | 226 |
| **Total** | | | **1,072 partidas** |

### Estructura de cada partida en PGN
Cada entrada contiene:
- **Headers PGN**: Event, Site, Date, Round, White, Black, Result, ECO, Annotator, PlyCount
- **FEN** (en posiciones de diagrama): Posición inicial del ejercicio/diagrama
- **Jugadas en notación algebraica estándar**: 1.e4 e5 2.Nf3 Nc6...
- **Comentarios de Grau en español**: Texto pedagógico embebido entre llaves `{...}` con explicaciones de conceptos, evaluaciones posicionales, razonamientos tácticos y estratégicos
- **Variantes**: Líneas alternativas entre paréntesis `(...)` con sus propios comentarios
- **NAGs (Numeric Annotation Glyphs)**: $1 (buena jugada), $2 (error), etc.

### Importancia clave
Los comentarios de Grau dentro del PGN **SON** el contenido pedagógico del libro. No se necesitan los PDFs como fuente primaria — los PGN contienen el texto explicativo completo con variantes y análisis.

### Estrategia de ingesta
```
PGN → Parser (python-chess) → Por cada partida:
  - Metadatos (jugadores, evento, fecha, ECO, resultado, tomo)
  - Posición FEN (si existe)
  - Jugadas en notación algebraica
  - Variantes con sus comentarios
  - Comentarios de Grau (texto pedagógico)
  → Chunk = una partida completa con sus anotaciones
  → Embeddings → ChromaDB (Vector Store)
```

### Estrategia de chunking
Cada partida anotada es un **chunk natural**. Los metadatos (tomo, tema, jugadores, ECO) se almacenan como metadata filtrable en ChromaDB. Para partidas con comentarios muy extensos (>2000 tokens), se puede subdividir manteniendo los metadatos compartidos.

---

## 3. COMPONENTES OBLIGATORIOS (RÚBRICA)

El proyecto debe cumplir **7 componentes obligatorios** para aprobar:

### 3.1 RAG Funcional
- Pipeline de embeddings y retrieval sobre las partidas anotadas de Grau
- Ingesta de los 4 archivos PGN (1,072 partidas)
- Chunking por partida (chunk natural)
- Vector store con ChromaDB
- Generación de respuestas con grounding (citando contenido real de Grau)

### 3.2 Agente con Herramientas (mínimo 2, implementamos 3)
| Herramienta | Función |
|-------------|---------|
| **Buscar en Grau** (RAG) | Consulta el vector store para responder dudas conceptuales sobre ajedrez |
| **Motor de tablero** (`python-chess`) | Recibe jugadas o FEN, valida legalidad, devuelve posición, detecta jaque/mate/tablas |
| **Generador de ejercicios** | Selecciona posiciones del corpus según tema, presenta problema táctico, valida respuesta del alumno |

- Patrón ReAct con cadena de razonamiento visible (logging)
- El agente decide qué herramienta usar según la consulta

### 3.3 Orquestación Multiagente
Dos roles diferenciados con flujo de control:

| Rol | Responsabilidad |
|-----|----------------|
| **Tutor** | Recibe preguntas, consulta RAG, explica conceptos, delega al Evaluador si se pide ejercicio |
| **Evaluador** | Genera ejercicios tácticos, recibe respuesta del alumno, califica usando python-chess, da retroalimentación |

- Hand-off: Tutor → Evaluador → Tutor
- Fallback: si el RAG no encuentra contenido relevante, el Tutor lo comunica honestamente
- Implementado con LangGraph (grafos de estado)

### 3.4 Memoria y/o HITL
**Memoria (SQLite):**
- Tabla `progreso_alumno`: temas consultados, ejercicios resueltos, tasa de acierto
- Tabla `historial_conversacion`: persiste contexto entre interacciones
- El Tutor consulta el progreso para adaptar respuestas

**HITL (Human-in-the-Loop):**
- Antes de calificar un ejercicio como incorrecto, el Evaluador muestra su razonamiento y pide confirmación
- El alumno puede disputar la evaluación y el sistema re-evalúa
- Nodo de interrupción en LangGraph

### 3.5 Evaluación
- Dataset de **25-30 pares** pregunta→respuesta esperada
- Tipos: conceptuales, tácticas, históricas
- Métricas:
  - **Faithfulness**: ¿la respuesta se basa en contenido de Grau?
  - **Retrieval Precision**: ¿se encontró el chunk correcto?
  - **Precisión de ejercicios**: ¿la jugada correcta es realmente correcta? (validable con python-chess)
- Script runner de evaluación automatizado

### 3.6 Calidad de Código
- Modular, documentado, con manejo de errores
- Pydantic v2 para contratos: `PartidaGrau`, `Ejercicio`, `ProgresoAlumno`, `RespuestaEvaluacion`
- `.env` para configuración (API keys, rutas, modelo LLM)
- Logging estructurado de cada paso del agente
- Type hints en todo el código

### 3.7 Documentación y README
- Descripción del proyecto
- Diagrama de arquitectura (Mermaid o imagen)
- Instrucciones de instalación (Docker y manual)
- Instrucciones de uso
- Explicación de decisiones de ingeniería
- Ejemplos de uso con capturas

---

## 4. STACK TÉCNICO

| Capa | Tecnología | Propósito |
|------|-----------|-----------|
| Lenguaje | Python 3.11+ | Base del proyecto |
| Orquestación de agentes | LangGraph | Multiagente (Tutor ↔ Evaluador), flujo de control, hand-offs |
| Cadenas y prompts | LangChain | RAG chain, prompt templates, herramientas del agente |
| Contratos/Schemas | Pydantic v2 | Modelos de datos tipados y validados |
| Vector Store | ChromaDB (servicio Docker) | Almacenamiento y retrieval de embeddings |
| Embeddings | OpenAI `text-embedding-3-small` | Generación de embeddings del corpus |
| LLM | Claude Sonnet (Anthropic API) | Razonamiento del agente, generación de respuestas |
| Motor de ajedrez | `python-chess` | Parseo PGN, validación de jugadas, FEN, detección mate/jaque, renderizado SVG de tableros |
| Memoria/Estado | SQLite | Persistencia de progreso y conversaciones |
| UI | Streamlit | Interfaz web: chat, tablero (SVG), progreso del alumno |
| Logging | `logging` (Python stdlib) | Trazabilidad del agente |
| Config | `python-dotenv` | Variables de entorno |
| Contenedorización | Docker + Docker Compose | Reproducibilidad y despliegue |

### Nota sobre la UI
Se eligió **Streamlit** sobre FastAPI + frontend manual por las siguientes razones:
- Componentes de chat nativos (`st.chat_message`, `st.chat_input`)
- Estado de sesión integrado (`st.session_state`)
- Invocación directa de LangGraph sin necesidad de capa API intermedia
- Ahorro significativo de tiempo de desarrollo (~3-4 días)
- Renderizado de tableros SVG generados por `python-chess` vía `chess.svg.board()`

---

## 5. ARQUITECTURA

### Flujo principal
```
[Streamlit UI]
    ├── Chat (st.chat_message / st.chat_input)
    ├── Tablero SVG (chess.svg.board → st.image)
    └── Panel de progreso
        ↓ (invocación directa, sin API REST)
[LangGraph Orchestrator]
    ├── [Nodo Tutor]
    │     ├── Consulta RAG (ChromaDB)
    │     ├── Usa Motor de Tablero (python-chess)
    │     └── Delega a Evaluador si se pide ejercicio
    │
    └── [Nodo Evaluador]
          ├── Genera ejercicio desde corpus
          ├── Valida respuesta con python-chess
          └── Da retroalimentación
    ↓
[Memoria SQLite] ← progreso, historial
```

### Flujo de ejemplo completo
```
Alumno escribe en chat: "Explícame la clavada con un ejemplo"
  → Streamlit envía mensaje al grafo LangGraph
  → Nodo Tutor consulta RAG → encuentra partida con clavada en Grau
  → Tutor extrae posición, usa python-chess para generar FEN
  → python-chess genera SVG del tablero
  → Streamlit muestra: explicación de Grau + tablero SVG

Alumno escribe: "Dame un ejercicio de clavada"
  → Nodo Tutor detecta solicitud de ejercicio → hand-off al Evaluador
  → Nodo Evaluador busca posiciones con clavada en el corpus
  → Genera FEN de la posición, renderiza SVG
  → Streamlit muestra tablero + pregunta "¿Cuál es la mejor jugada?"
  → Alumno responde: "Dxf7"
  → Evaluador usa python-chess para validar la jugada
  → Da retroalimentación basada en los comentarios de Grau
  → Actualiza progreso en SQLite
```

---

## 6. ESTRUCTURA DEL REPOSITORIO

```
chess-tutor-grau/
├── core/               # Cliente LLM, configuración global, logging
│   ├── __init__.py
│   ├── config.py       # Settings con Pydantic BaseSettings + .env
│   ├── llm.py          # Cliente Anthropic/OpenAI configurable
│   └── logging.py      # Setup de logging estructurado
│
├── prompting/          # Templates de prompts
│   ├── __init__.py
│   ├── tutor.py        # System prompt y templates del Tutor
│   └── evaluator.py    # System prompt y templates del Evaluador
│
├── contracts/          # Schemas Pydantic (compartidos UI ↔ agentes)
│   ├── __init__.py
│   ├── partida.py      # PartidaGrau, ChunkMetadata
│   ├── ejercicio.py    # Ejercicio, RespuestaAlumno, Evaluacion
│   └── progreso.py     # ProgresoAlumno, HistorialConversacion
│
├── rag/                # Pipeline RAG completo
│   ├── __init__.py
│   ├── ingest.py       # Parser PGN → chunks
│   ├── embeddings.py   # Generación de embeddings
│   ├── store.py        # ChromaDB HttpClient y operaciones
│   └── retrieval.py    # Retrieval chain con LangChain
│
├── agents/             # Herramientas del agente
│   ├── __init__.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search_grau.py    # Herramienta RAG
│   │   ├── chess_engine.py   # Herramienta python-chess
│   │   └── exercise_gen.py   # Herramienta generador de ejercicios
│   └── react_agent.py        # Agente ReAct
│
├── graph/              # Orquestación LangGraph
│   ├── __init__.py
│   ├── state.py        # Definición del estado del grafo
│   ├── nodes.py        # Nodos: tutor, evaluador, router
│   └── graph.py        # Construcción y compilación del grafo
│
├── memory/             # Persistencia de estado
│   ├── __init__.py
│   ├── database.py     # SQLite setup y migraciones
│   ├── progress.py     # CRUD progreso del alumno
│   └── history.py      # CRUD historial de conversación
│
├── evals/              # Evaluación cuantitativa
│   ├── __init__.py
│   ├── dataset.json    # Dataset de evaluación (25-30 pares)
│   ├── metrics.py      # Cálculo de métricas
│   └── runner.py       # Script runner de evaluación
│
├── app/                # Streamlit UI
│   ├── main.py         # App principal (st.chat, layout, tablero)
│   └── components/     # Componentes reutilizables
│       ├── board.py    # Renderizado de tablero SVG
│       └── progress.py # Panel de progreso del alumno
│
├── data/               # Archivos fuente
│   ├── Grau_I.pgn      # Tomo I — Rudimentos (164 partidas)
│   ├── Grau_II.pgn     # Tomo II — Estrategia (418 partidas)
│   ├── Grau_III.pgn    # Tomo III — Medio juego (264 partidas)
│   └── Grau_IV.pgn     # Tomo IV — Estrategia Superior (226 partidas)
│
├── tests/              # Pruebas
│   ├── test_ingest.py
│   ├── test_tools.py
│   └── test_graph.py
│
├── deploy/             # Docker
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── .env.example        # Variables de entorno (sin claves reales)
├── .gitignore
├── README.md
├── requirements.txt
└── main.py             # Punto de entrada CLI (pruebas sin UI)
```

---

## 7. DOCKER

### Imágenes base
| Servicio | Imagen | Propósito |
|----------|--------|-----------|
| App (Streamlit) | `python:3.11-slim-bookworm` | Imagen oficial Python, ligera (~150MB), Debian Bookworm |
| ChromaDB | `chromadb/chroma:1.5.3` | Imagen oficial ChromaDB, vector store como servicio separado |

### docker-compose.yml
```yaml
services:
  app:
    build:
      context: .
      dockerfile: deploy/Dockerfile
    ports:
      - "8501:8501"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - sqlite-data:/app/db
    depends_on:
      chromadb:
        condition: service_healthy

  chromadb:
    image: chromadb/chroma:1.5.3
    ports:
      - "8000:8000"
    volumes:
      - chroma-data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v2/heartbeat"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  chroma-data:
  sqlite-data:
```

### Dockerfile
```dockerfile
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### .env.example
```env
# LLM
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-sonnet-4-20250514

# Embeddings
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small

# ChromaDB
CHROMA_HOST=chromadb
CHROMA_PORT=8000

# SQLite
SQLITE_DB_PATH=/app/db/chess_tutor.db

# Logging
LOG_LEVEL=INFO
```

---

## 8. PLAN DE EJECUCIÓN (4 SEMANAS)

### Semana 1 (Abr 19–25): RAG Funcional
- Día 1-2: Script de ingesta PGN con python-chess, modelo Pydantic `PartidaGrau`
- Día 3-4: Pipeline de embeddings, carga en ChromaDB (1,072 partidas)
- Día 5-6: RAG chain con LangChain, probar con 10 preguntas manuales
- Día 7: Estructura de carpetas, .env.example, Docker base, primer commit
- **Entregable**: Pregunta → Respuesta con grounding de Grau

### Semana 2 (Abr 26–May 2): Agente con Herramientas
- Día 1-2: Herramienta RAG como tool de LangChain
- Día 3-4: Herramienta Motor de Tablero (python-chess + SVG)
- Día 5-6: Herramienta Generador de Ejercicios
- Día 7: Agente ReAct integrado con logging de razonamiento
- **Entregable**: Agente funcional con 3 herramientas

### Semana 3 (May 3–9): Multiagente + Memoria + HITL + UI
- Día 1-3: LangGraph (Tutor ↔ Evaluador), hand-off, fallbacks
- Día 4-5: SQLite (progreso + historial), HITL con nodo de interrupción
- Día 6-7: Streamlit UI (chat + tablero SVG + panel progreso)
- **Entregable**: Sistema multiagente completo con interfaz web

### Semana 4 (May 10–17): Evaluación + Documentación + Pulido
- Día 1-3: Dataset de evaluación, métricas, runner automatizado
- Día 4-5: README completo, diagrama de arquitectura
- Día 6: Docker funcional (docker-compose up desde cero)
- Día 7: Prueba en entorno limpio, push final, entrega en plataforma
- **Entregable**: Proyecto completo entregado

---

## 9. DECISIONES DE INGENIERÍA (PARA DOCUMENTAR EN README)

1. **PGN sobre PDF**: Los archivos PGN contienen los comentarios pedagógicos de Grau embebidos, lo que elimina la necesidad de OCR o extracción de texto de PDFs. Cada partida es un chunk natural con metadatos estructurados.

2. **python-chess como herramienta del agente**: Permite validar jugadas programáticamente, evitando que el LLM "alucine" jugadas ilegales. El agente puede "ver" el tablero sin imágenes. Además genera tableros SVG para visualización.

3. **Streamlit sobre FastAPI + frontend manual**: Elimina la necesidad de una capa API intermedia. El grafo LangGraph se invoca directamente desde la UI. Componentes de chat nativos y estado de sesión integrado ahorran ~3-4 días de desarrollo.

4. **ChromaDB como servicio Docker separado**: Separa el estado del vector store del backend, facilita persistencia y reproducibilidad.

5. **SQLite para memoria**: Suficiente para un proyecto académico, sin overhead de configurar PostgreSQL. Persiste progreso y conversaciones entre sesiones.

6. **3 herramientas en lugar de 2**: La rúbrica pide mínimo 2. Con 3 (RAG, motor de tablero, generador de ejercicios) se demuestra mayor dominio sin complejidad innecesaria.

7. **Chunking por partida**: Cada partida anotada es una unidad pedagógica completa. No se fragmentan los comentarios de Grau, lo que preserva el contexto necesario para respuestas coherentes.

8. **Tableros como SVG**: `python-chess` genera SVG del tablero con `chess.svg.board()`. Streamlit los renderiza directamente. Evita dependencias de JavaScript externas (chessboard.js) manteniendo visualización clara.

---

## 10. CRITERIOS DE EVALUACIÓN (RESUMEN)

| Resultado | Requisito |
|-----------|-----------|
| **Aprobado** | 7 componentes obligatorios completos, ≥70% del peso total |
| **Aprobado con distinción** | Obligatorios con alta calidad + 2 opcionales (Docker + video cuentan) |
| **No aprobado** | Faltan componentes obligatorios o el sistema no funciona end-to-end |

### Ejes de evaluación
- **Funcionalidad**: RAG responde con grounding, agente razona y actúa, multiagente cumple su propósito
- **Ingeniería**: Código modular, Pydantic, .env, logging, pruebas y evaluaciones cuantitativas
- **Presentación**: README claro y completo, cualquier persona puede entender, instalar y usar el proyecto

---

## 11. NOTAS IMPORTANTES

- **No replicar DocOps Agent** (proyecto de clase). Este es un proyecto propio.
- **Priorizar calidad sobre cantidad**: mejor 3 herramientas bien hechas que 10 frágiles.
- **Iterar incrementalmente**: RAG → agente → multiagente → evaluación.
- **Evaluar cuantitativamente**: no basta con que "se vea que funciona".
- **Documentar decisiones**: el razonamiento importa tanto como el código.
- **Corpus total**: 1,072 partidas anotadas en 4 tomos de Grau.