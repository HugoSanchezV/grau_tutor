# PROGRESS — Chess Tutor Grau

## Estado actual: PROYECTO COMPLETO — Entrega 2026-05-18

---

## Lo que se construyó

### Estructura de carpetas creada
```
core/           — Configuración global, logging, cliente LLM
contracts/      — Schemas Pydantic v2
rag/            — Pipeline RAG completo
agents/         — (vacío, semana 2)
graph/          — (vacío, semana 3)
memory/         — (vacío, semana 3)
evals/          — (vacío, semana 4)
tests/          — Tests de ingesta
```

### Módulos implementados

#### `core/`
| Archivo | Contenido |
|---------|-----------|
| `config.py` | `Settings` con Pydantic BaseSettings + `.env`. Variables: API keys (Anthropic, OpenAI), ChromaDB host/port, SQLite path, log level, data dir. |
| `logging.py` | Setup de logging estructurado con `get_logger(name)`. Formato: `fecha | nivel | módulo | mensaje`. |
| `llm.py` | `get_llm()` → `ChatAnthropic` con modelo configurable vía `.env`. |

#### `contracts/`
| Archivo | Contenido |
|---------|-----------|
| `partida.py` | `ChunkMetadata` (tomo, tema, event, white, black, result, eco, annotator, fen, ply_count, partida_id) + `PartidaGrau` (partida_id, texto_completo, jugadas, comentarios, metadata). Método `to_chroma_document()` para serializar a ChromaDB. |

#### `rag/`
| Archivo | Contenido |
|---------|-----------|
| `ingest.py` | Parser PGN con `python-chess`. Extrae jugadas (notación algebraica) + comentarios de Grau (texto pedagógico) + variantes. Construye `texto_completo` para embedding. Divide chunks >7200 chars preservando oraciones. `ingest_all(data_dir)` procesa los 4 tomos. |
| `store.py` | `get_chroma_client()` → HttpClient. `get_or_create_collection()`. `add_documents()` en batches de 100. `query_collection()` con soporte de filtros `where`. `collection_is_empty()`. |
| `embeddings.py` | `embed_texts(texts)` → batches de 100 a OpenAI `text-embedding-3-small`. `embed_query(query)` → embedding individual. |
| `retrieval.py` | `GrauRetriever`: construye LangChain chain con `ChatPromptTemplate` + `ChatAnthropic` + `StrOutputParser`. Método `ask(question)` para respuesta completa. Método `retrieve_raw()` para chunks crudos (usado por herramientas del agente). Prompt en español con instrucción de citar a Grau. |
| `pipeline.py` | Script de ingesta completa. Detecta si la colección ya existe (evita re-ingesta). Flag `--force` para reingestar. Flujo: `ingest_all` → `embed_texts` → `add_documents`. |

---

## Resultado de la ingesta (parser)

| Tomo | Tema | Chunks generados |
|------|------|-----------------|
| I | Rudimentos | 112 |
| II | Estrategia | 256 |
| III | Medio juego / Conformación de peones | 160 |
| IV | Estrategia Superior | 158 |
| **Total** | | **686 chunks** |

> Las 1,072 partidas originales generan 686 chunks porque partidas con FEN pero sin jugadas adicionales son más cortas. Ningún chunk superó el límite de 7,200 chars (no hubo subdivisiones).

---

## Tests

Archivo: `tests/test_ingest.py` — **5/5 pasando**

| Test | Qué verifica |
|------|-------------|
| `test_parse_tomo_i` | Parser procesa Tomo I, genera `PartidaGrau` válidos |
| `test_encoding_correcta` | Acentos en español correctos (U+00F3 = ó, no bytes Latin-1) |
| `test_ingest_all_todos_los_tomos` | Los 4 tomos se procesan, >600 chunks totales |
| `test_to_chroma_document` | Serialización a formato ChromaDB correcta |
| `test_ids_unicos` | Sin IDs duplicados en todo el corpus |

---

## Decisiones tomadas

1. **Chunking por partida**: cada partida completa es un chunk. No se fragmentan los comentarios de Grau para preservar el contexto pedagógico.
2. **Subdivisión a 7,200 chars**: para partidas muy largas, se divide por oraciones. En la práctica ninguna partida del corpus lo requirió.
3. **`texto_completo` como campo de embedding**: combina jugadores, apertura ECO, resultado, jugadas y análisis de Grau en un solo texto coherente.
4. **`retrieve_raw()` en `GrauRetriever`**: método separado que devuelve chunks crudos sin pasar por el LLM, necesario para que las herramientas del agente (semana 2) accedan directamente al corpus.
5. **Encoding**: los PGN usan UTF-8 con BOM (`\xef\xbb\xbf`). Se abre con `encoding="utf-8-sig"` y se pasa como `io.StringIO` a python-chess.

---

## Pendiente para ejecutar la ingesta real

1. Tener `.env` con las API keys reales (copiar de `.env.example`)
2. Levantar ChromaDB: `docker-compose up chromadb -d`
3. Ejecutar ingesta: `py rag/pipeline.py`
4. Verificar con: `py rag/pipeline.py` (segunda vez, debe decir "ya tiene N documentos")

---

## Semana 2 — Agente con Herramientas

### Módulos implementados

#### `agents/tools/`
| Archivo | Contenido |
|---------|-----------|
| `search_grau.py` | `SearchGrauInput` (Pydantic, valida `k` 1–15 y `tomo` 1–4). `ChunkResult` tipado. `search_grau()` busca en ChromaDB con pool ampliado cuando hay filtros. `format_chunks_for_llm()` serializa a texto con fuente (tomo, jugadores, ECO, FEN). `build_search_grau_tool()` envuelve todo como `StructuredTool`. |
| `chess_engine.py` | Funciones puras: `validate_move`, `apply_move`, `list_legal_moves`, `analyze_position`. Acepta SAN o UCI. `render_board()` genera SVG para la UI (no expuesta al agente para evitar strings largos en contexto). 4 `StructuredTool` wrappers. Modelos de salida: `MoveValidation`, `MoveResult`, `PositionAnalysis`. |
| `exercise_gen.py` | `generate_exercise()`: busca chunks con FEN semántico; fallback a pool 30 sin filtro de tomo. `evaluate_answer()`: valida legalidad, compara SAN canónico si hay ground truth, feedback abierto si no. Modelos: `Ejercicio`, `Evaluacion`. 2 `StructuredTool`. |

#### `agents/react_agent.py`
| Elemento | Detalle |
|----------|---------|
| `GrauAgent` | Envuelve `create_react_agent` de LangGraph con `MemorySaver` por `thread_id`. |
| `build_tools()` | Ensambla las 7 herramientas: 1 search + 4 motor + 2 ejercicios. |
| `chat()` | Invoca el grafo, devuelve `AgentResponse(reply, reasoning)`. |
| `stream()` | Yield de estados del grafo para traza en vivo. |
| `reset()` | Descarta la memoria del hilo (nueva conversación). |
| `_extract_reasoning()` | Extrae tool calls + tool results como lista de dicts (cadena de pensamiento). |
| System prompt | 6 reglas obligatorias: siempre llamar `search_grau`, validar jugadas con el motor, citar fuente (tomo, ECO), responder en español. |

### Tests

| Archivo | Tests | Estado |
|---------|-------|--------|
| `tests/test_chess_engine.py` | Motor de tablero — validación, aplicación, listado, análisis | ✅ |
| `tests/test_exercise_gen.py` | Generación de ejercicios y evaluación de respuestas | ✅ |
| `tests/test_search_grau.py` | Búsqueda, formateo, validación de schema (k, tomo) | ✅ |
| `tests/test_react_agent.py` | Construcción del agente, chat, stream, memoria multi-turn | ✅ |
| **Total Semana 2** | **83/83** | ✅ |

### Decisiones tomadas

1. **`render_board` no expuesto al agente**: el SVG puede llegar a 10–20 KB; contaminaría el contexto del LLM. La UI lo llama directamente con el FEN que el agente devuelve.
2. **Pool ampliado con filtros**: si `tomo` o `tema` está activo, se recuperan `k×4` chunks de ChromaDB y se filtra después, para no quedarse cortos.
3. **Fallback de dos pasos en `generate_exercise`**: primero busca con filtro de tomo y `k=10`; si no hay FEN, amplía a `k=30` sin filtro.
4. **Validación de tomo estricta**: el validator de `SearchGrauInput` lanza `ValueError` para valores fuera de 1–4 (antes devolvía `None` silenciosamente).
5. **`MemorySaver` por `thread_id`**: cada sesión de alumno mantiene su propio hilo de memoria; `reset()` recrea el checkpointer si LangGraph no expone `delete_thread`.

---

---

## Semana 3 — Multiagente + Memoria + HITL + UI

### Módulos implementados

#### `contracts/`
| Archivo | Contenido |
|---------|-----------|
| `progreso.py` | `ProgresoAlumno` (student_id, tema, consultas, ejercicios_intentados/correctos, ultima_actividad) + `HistorialConversacion` (id, student_id, thread_id, role, content, timestamp). |

#### `memory/`
| Archivo | Contenido |
|---------|-----------|
| `database.py` | SQLite setup + `init_db()` (crea tablas si no existen) + `get_connection()` context manager con autocommit/rollback. |
| `progress.py` | `upsert_progress()` (INSERT … ON CONFLICT DO UPDATE para acumular deltas) + `get_progress()` + `get_progress_summary()` (texto para inyectar en contexto del agente). |
| `history.py` | `add_message()` + `get_history()` (ultimos N mensajes ordenados) + `clear_history()`. |

#### `graph/`
| Archivo | Contenido |
|---------|-----------|
| `state.py` | `TutorState` TypedDict con: messages (add_messages), student_id, mode, current_fen, expected_move, evaluation_reasoning, hitl_pending, hitl_decision, reasoning_trace, progress_summary. |
| `nodes.py` | `router_node` (fast-path por regex + clasificación LLM), `tutor_node` (delega a GrauAgent + persiste en SQLite), `evaluador_node` (genera/evalúa ejercicios + activa HITL si jugada legal-incorrecta), `hitl_review_node` (procesa "acepto"/"disputo"). |
| `graph.py` | `TutorGraph`: StateGraph compilado con `interrupt_before=["hitl_review"]` + `MemorySaver`. `GraphResponse` dataclass. Métodos: `chat()`, `is_interrupted()`, `resume_hitl()`, `reset()`. Función `build_graph()`. |

#### `app/`
| Archivo | Contenido |
|---------|-----------|
| `components/progress.py` | `render_progress_panel()`: métricas totales (consultas, ejercicios, % aciertos) + expander con desglose por tema. |
| `main.py` | UI actualizada: usa `TutorGraph` en lugar de `GrauAgent`. Panel de progreso en sidebar. Flujo HITL con dos botones ("Acepto"/"Disputo") que aparecen cuando `hitl_pending=True`. Chat input deshabilitado durante HITL. |

### Flujo del grafo

```
START → router → tutor  → END
              → evaluador → END              (respuesta correcta / ejercicio nuevo)
                         → [INTERRUPT]       (jugada legal pero incorrecta)
                         → hitl_review → END
```

### Tests

| Archivo | Tests | Estado |
|---------|-------|--------|
| `tests/test_graph.py` | Regex de jugadas, router (fast-path + LLM), HITL review, SQLite CRUD, TutorGraph build | ✅ |
| **Total Semana 3** | **24 nuevos / 112 totales** | ✅ |

### Decisiones tomadas

1. **`interrupt_before=["hitl_review"]`**: La interrupción ocurre DESPUÉS de que `evaluador_node` fija `hitl_pending=True` pero ANTES de que `hitl_review_node` ejecute. La UI de Streamlit detecta el interrupt con `get_state().next` y muestra los botones.
2. **Fast-path en el router**: Si hay `current_fen` activo y el mensaje coincide con regex SAN/UCI, se omite la llamada al LLM y se va directo al evaluador. Evita latencia y errores de clasificación en mensajes cortos tipo "Nf3".
3. **GrauAgent como subagente del tutor**: El nodo tutor invoca `GrauAgent.chat()` con un `thread_id` propio (`{student_id}_tutor`), manteniendo la cadena ReAct interna independiente del grafo exterior. El grafo exterior solo almacena el mensaje final.
4. **SQLite con `UPSERT` acumulativo**: `upsert_progress` usa `INSERT … ON CONFLICT DO UPDATE SET campo = campo + excluded.campo` para acumular contadores sin race conditions en la capa de aplicación.
5. **Chat input deshabilitado durante HITL**: Evita que el alumno envíe nuevo mensaje mientras hay una evaluación pendiente de confirmar.

---

---

## Semana 4 — Evaluación + Auditoría + Pulido

### Auditoría del sistema agéntico (commit `ef3a8d9`)

Se corrigieron 6 debilidades detectadas en el agente y el grafo:

1. **GrauAgent stateless** — eliminado `MemorySaver` propio; el estado lo gestiona el `TutorGraph`
2. **Una sola fuente de verdad** — `SqliteSaver` centralizado en `core/checkpointer.py`
3. **Router fail-closed** — respuesta `"tutor"` por defecto ante LLM no clasificable
4. **Guardrails deterministas** — `core/guardrails.py`: regex de prompt injection ejecutado ANTES del LLM
5. **LLM retry con clasificación** — `core/llm_retry.py`: distingue errores transitorios (rate limit, timeout) de errores de configuración (modelo no encontrado, key inválida); retry con backoff solo para transitorios
6. **Nodo `refusal`** — nodo terminal en el grafo para bloquear off-topic e inyecciones con mensaje específico

#### Nuevos módulos

| Archivo | Contenido |
|---------|-----------|
| `core/guardrails.py` | Regex bilingüe (ES/EN) para prompt injection. `is_prompt_injection(text)`. Mensajes de rechazo: off-topic, injection, error de agente, error de config, recursión. |
| `core/llm_retry.py` | `is_transient_error(exc)` + `is_config_error(exc)`. Agnóstico al provider (Groq, OpenAI, Anthropic). Un error de config nunca es transitorio, aunque contenga keywords solapados. |

### Módulos de evaluación

| Archivo | Contenido |
|---------|-----------|
| `evals/runner.py` | Eval de retrieval A/B (dense vs híbrido) sobre 25 queries. Hit@5 = **80%** híbrido vs 72% dense. |
| `evals/faithfulness_runner.py` | Faithfulness del agente sobre 25 pares. Groundedness = **100%**, citation = 69%. |
| `evals/exercises_runner.py` | Generador + evaluador de jugadas sobre 25 ejercicios. Precisión = **100%** (correctas / ilegales). |
| `evals/dataset.json` | 25 pares retrieval |
| `evals/faithfulness_dataset.json` | 25 pares faithfulness |
| `evals/exercises_dataset.json` | 25 pares ejercicios |

### Tests (estado final)

| Archivo | Tests | Estado |
|---------|-------|--------|
| `tests/test_ingest.py` | 5 | ✅ |
| `tests/test_search_grau.py` | 16 | ✅ |
| `tests/test_chess_engine.py` | 29 | ✅ |
| `tests/test_exercise_gen.py` | 23 | ✅ |
| `tests/test_react_agent.py` | 18 | ✅ |
| `tests/test_graph.py` | 36 | ✅ |
| **Total** | **127** | ✅ |

Los 15 tests nuevos de Semana 4 en `test_graph.py` cubren: router injection blocking, off-topic via LLM, fail-closed, fast-path con vocab de ajedrez, sanitización de input, y el nodo `refusal`.

### Documentación

- `README.md` — Completo con diagrama de arquitectura Mermaid, quickstart Docker, ejemplos con capturas, tabla de evaluaciones, decisiones de arquitectura, limitaciones y roadmap.
- `evals/EVAL_REPORT.md`, `evals/FAITHFULNESS_REPORT.md`, `evals/EXERCISES_REPORT.md` — Reportes cuantitativos de cada eval.
- `docs/prompt_documentacion_tecnica.md` — Documentación técnica de prompts.
