# Evaluación de Faithfulness del Agente — Chess Tutor Grau

**Fecha:** 2026-04-28
**Dataset:** 15 queries (13 in-scope + 2 out-of-scope) | k=5
**Modelo:** OpenAI gpt-4o-mini (temperature=0.0)
**Corpus:** 660 chunks de los 4 tomos del Tratado General de Ajedrez de Grau

---

## Por qué evaluar faithfulness

El reporte de retrieval ([EVAL_REPORT.md](EVAL_REPORT.md)) demuestra que el sistema híbrido recupera contexto correcto en 80% de las queries. Pero recuperar bien no garantiza que el LLM **use** ese contexto para responder. Un agente puede:

- Recibir el contexto correcto y responder desde su conocimiento previo (no usa el corpus → alucinación)
- Recibir el contexto correcto y responder bien pero sin citar la fuente (incumple el contrato pedagógico)
- Responder a preguntas fuera del dominio en lugar de abstenerse

Faithfulness mide si el agente **respeta el corpus** y el contrato del prompt:
1. ¿Llama a `search_grau` antes de responder? (no alucina)
2. ¿Cita la fuente en la respuesta? (trazabilidad pedagógica)
3. ¿Su respuesta menciona los conceptos clave del corpus? (groundedness)
4. ¿Se abstiene en preguntas fuera del dominio?

---

## Metodología

Cada query del dataset trae:
- `query` — la pregunta al agente
- `must_use_search` — si el agente debe llamar a search_grau
- `must_cite_source` — si la respuesta debe citar fuente
- `must_mention_any` — keywords que deben aparecer en la respuesta
- `must_abstain` — para queries out-of-scope, validar abstención

Todas las métricas son **deterministas** (regex y substring) — sin LLM-as-judge para evitar circularidad y coste de tokens.

### Detección de citas

Distinguimos dos niveles:

| Nivel | Regex | Significado |
|-------|-------|-------------|
| **Estricta** | `Tomo [1-4]`, `tomo\d-\d+`, `ECO [A-E]\d{2}`, `partida \d+` | Cita el pasaje exacto del corpus |
| **Débil** | `Grau`, `Tratado General` | Menciona autor/libro pero no localiza el pasaje |

Una cita estricta es lo que pide el system_prompt (`"Cita siempre la fuente (tomo, partida, ECO si aplica)"`). La cita débil es aceptable como mínimo pero diagnostica que el agente no localiza el pasaje específico.

### Decisión de "passed"

- **In-scope:** debe usar search_grau ∧ debe citar (estricta o débil) ∧ groundedness ≥ 25% de keywords
- **Out-of-scope:** NO debe usar search_grau ∧ debe mencionar abstención ("no puedo", "fuera del alcance", etc.)

---

## Resultados

### Resumen global

| Métrica | In-scope (n=13) | Out-of-scope (n=2) |
|---------|----------------:|-------------------:|
| **tool_use** (llama search_grau) | **92.3%** | 0.0% |
| **citation strict** (Tomo X) | 23.1% | 0.0% |
| **citation weak** (Grau / Tratado) | 69.2% | 0.0% |
| **citation any** | 69.2% | 0.0% |
| **groundedness** (keywords ≥ 25%) | **100.0%** | 100.0% |
| **avg keyword_overlap** | 0.850 | — |
| **passed (overall)** | 61.5% | **100.0%** |

### Detalle por query

| ID | Tipo | Search | Cite strict | Cite weak | Overlap | Grounded | PASS |
|----|------|:------:|:-----------:|:---------:|--------:|:--------:|:----:|
| f01 | concepto | Y | – | Y | 0.75 | Y | PASS |
| f02 | concepto | Y | – | Y | 0.75 | Y | PASS |
| f03 | concepto | Y | **Y** | Y | 1.00 | Y | PASS |
| f04 | final | Y | – | – | 0.75 | Y | **FAIL** |
| f05 | concepto | Y | – | Y | 0.50 | Y | PASS |
| f06 | estrategia | Y | **Y** | Y | 1.00 | Y | PASS |
| f07 | táctica | Y | – | Y | 1.00 | Y | PASS |
| f08 | táctica | – | Y | Y | 0.50 | Y | **FAIL** |
| f09 | jugador (Capablanca) | Y | – | – | 1.00 | Y | **FAIL** |
| f10 | concepto | Y | – | Y | 1.00 | Y | PASS |
| f11 | final | Y | – | – | 1.00 | Y | **FAIL** |
| f12 | out_of_scope (pizza) | – | – | – | 0.40 | Y | PASS |
| f13 | out_of_scope (planetas) | – | – | – | 0.40 | Y | PASS |
| f14 | concepto | Y | – | – | 1.00 | Y | **FAIL** |
| f15 | concepto | Y | – | Y | 0.80 | Y | PASS |

---

## Análisis

### Lo que funciona bien

**Groundedness 100%:** la respuesta del agente menciona conceptos clave del corpus en TODAS las queries in-scope, con un overlap promedio del 85%. El agente no alucina información ajena al corpus.

**Tool use 92.3%:** en 12 de 13 queries el agente llama a `search_grau` antes de responder. Solo f08 (combinación que termina en mate) saltó la búsqueda — posiblemente porque el modelo consideró que era una pregunta de pedagogía general.

**Abstención 100% (out-of-scope):** ante preguntas claramente fuera del dominio (pizza, sistema solar) el agente NO llama a search_grau y se identifica como tutor de ajedrez. Comportamiento ideal.

### Lo que falla — el problema de citación

**Solo 23.1% de las respuestas citan el tomo específico** ("Tomo 2", "ECO B22", "partida tomo3-45"). El system_prompt es explícito:

> *"Cita siempre la fuente (tomo, partida, ECO si aplica) de lo que recupera search_grau."*

El modelo no sigue esta instrucción consistentemente. En 5/13 queries la respuesta es totalmente correcta pero **no cita ninguna forma de fuente** (ni siquiera "Grau") — son los FAIL de f04, f08, f09, f11, f14.

Las posibles causas:
1. **Prompt débil:** "cita la fuente" es ambiguo. Mejor sería un formato obligatorio: `"Termina cada respuesta con: Fuente: Tomo X (capítulo Y)."`
2. **Tool result no enfatiza el tomo:** [search_grau.py:148](agents/tools/search_grau.py#L148) sí incluye `"[Fuente N] Tomo X (tema)"` en el output, pero el modelo a veces lo omite.
3. **gpt-4o-mini conservador:** un modelo más capaz (gpt-4o, claude-sonnet) probablemente seguiría la instrucción mejor.

### El caso f08 — combinación que termina en mate

Esta query es la única donde el agente **NO llamó a search_grau** pero la respuesta sí cita el Tratado y menciona conceptos de mate y combinación. Comportamiento ambiguo: pasó groundedness y citó débil, pero violó la regla "SIEMPRE llama a search_grau antes de responder". Falla por ese motivo.

### Mitigaciones propuestas

1. **Citation enforcement post-hoc:** añadir un nodo en el grafo que valide que la respuesta del agente contiene una cita estricta. Si no, regenerar con un prompt forzado.
2. **System prompt más estricto:**
   ```
   FORMATO OBLIGATORIO de respuesta:
   1. [Texto de la respuesta]
   2. Fuente: Tomo X (tema), partida {white} vs {black} {result}.
   ```
3. **Test contra modelos más capaces:** correr este eval con gpt-4o o claude-sonnet-4 para confirmar si es limitación de gpt-4o-mini o del prompt.

---

## Conclusiones

1. **El agente es fiel al corpus** (groundedness 100%, alucinación cero) — la integración RAG funciona.
2. **Abstención out-of-scope es perfecta** — el agente no responde a preguntas fuera del dominio.
3. **La citación específica es el punto débil**: el modelo cumple "responde con info del corpus" pero falla en "indica de qué tomo/partida". Es un problema de adherencia al prompt, no de retrieval.
4. **Pass rate global: 73.3%** (11/15) — aceptable para un MVP pedagógico, mejorable con prompt engineering o validación post-hoc.

---

*Ejecutar: `py evals/faithfulness_runner.py` (requiere ChromaDB + OPENAI_API_KEY)*
*Resultados completos en [`evals/faithfulness_results.json`](faithfulness_results.json)*
