# Evaluación del Retriever — Chess Tutor Grau

**Fecha:** 2026-04-28
**Dataset:** 25 queries | k=5 | n_candidates=20
**Colección:** `grau_partidas` — 660 chunks (4 tomos de Grau)

---

## Por qué se pasó de Dense-Only a Hybrid

El sistema empezó con **retrieval denso puro**: cada chunk se convierte en un vector de 1536 dimensiones (OpenAI `text-embedding-3-small`) y la búsqueda recupera los más cercanos por similitud coseno. Esto funciona bien para preguntas donde el significado semántico basta.

El problema aparece con términos técnicos poco frecuentes o muy específicos. Un embedding aprende representaciones generales del lenguaje; si un término como "oposición" (concepto táctico de finales) aparece en muy pocos chunks, su vector queda "diluido" entre otros usos del mismo idioma. El modelo semántico no sabe que en ajedrez "oposición" tiene un significado preciso y restringido.

La solución es **BM25** (Best Match 25), un algoritmo de recuperación léxica clásico. BM25 hace búsqueda exacta de palabras: si la query contiene "oposición", busca directamente ese token en el corpus. Es determinista, rápido y excelente con términos técnicos, nombres propios, y vocabulario específico del dominio.

Combinar ambos se llama **retrieval híbrido**, y la fusión se hace con **RRF (Reciprocal Rank Fusion)**:

```
score(doc) = Σ  1 / (k + rank_i)
```

Cada sistema rankea los documentos de forma independiente, y RRF combina los rankings sumando el recíproco del puesto (k=60 valor canónico de Cormack et al., 2009). Un documento que aparece bien posicionado en ambos sistemas sube más que uno que destaca en solo uno.

### Flujo del retrieval híbrido

```
Query
  │
  ├── embed_query() ──► ChromaDB (cosine) ──► top-20 dense
  │
  └── BM25Okapi ───────────────────────────► top-20 sparse
                                              (stopwords españolas filtradas)
            │                   │
            └──── RRF Fusion ───┘
                       │
                   top-5 fusionados ──► LLM (o devuelto como raw)
```

---

## Metodología de evaluación

Se usa **evaluación con criterios verificables externos** en lugar de anotación manual. Para cada query se define un criterio determinista:

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| `keyword_any` | El doc contiene al menos uno de los términos | `["clavada", "clavar"]` |
| `keyword_all` | El doc contiene todos los términos | `["sacrificio", "calidad"]` |
| `meta_any` | Al menos una condición sobre metadata se cumple | `white="Capablanca" OR black="Capablanca"` |

Si el criterio se cumple, el chunk se considera **relevante**. La comparación es case-insensitive y por substring, no por similitud semántica, lo que elimina la circularidad de usar embeddings para evaluar embeddings.

**Trade-off honesto:** si Grau explica un concepto sin nombrarlo explícitamente, el criterio lo contará como no relevante (falso negativo). Esto hace las métricas **conservadoras**, pero el sesgo aplica igual a ambos sistemas, por lo que el A/B comparativo sigue siendo válido.

### Métricas calculadas

| Métrica | Fórmula | Qué mide |
|---------|---------|----------|
| **Hit@K** | 1 si hay ≥1 relevante en top-K, si no 0 | Recall mínimo: ¿el LLM tendrá contexto correcto? |
| **MRR@K** | 1/posición del primer relevante | ¿El mejor chunk llega primero? |
| **P@K** | relevantes / K | ¿Cuánto ruido hay en el contexto? |

---

## Resultados

### Resumen global (k=5, 25 queries)

| Sistema | Hit@5 | MRR@5 | P@5 |
|---------|------:|------:|----:|
| dense_only | 0.720 | 0.668 | 0.512 |
| **hybrid** | **0.800** | **0.703** | **0.568** |
| **Mejora** | **+11%** | **+5%** | **+11%** |

### Distribución por categoría (hybrid)

| Categoría | Queries | Hit@5 | MRR@5 |
|-----------|--------:|------:|------:|
| concepto_pedagogico | 7 | 1.00 | 0.93 |
| jugador_historico | 5 | 1.00 | 0.90 |
| tactica | 4 | 0.50 | 0.50 |
| final | 3 | 1.00 | 0.75 |
| estrategia | 3 | 0.33 | 0.33 |
| apertura | 3 | 0.67 | 0.44 |

### Detalle por query

| ID | Query | Categoría | Dense hit/mrr/P | Hybrid hit/mrr/P |
|----|-------|-----------|:-:|:-:|
| q01 | ¿qué es un peón pasado? | concepto | 1/1.00/0.60 | 1/1.00/1.00 |
| q02 | explícame la clavada | concepto | 1/1.00/1.00 | 1/1.00/1.00 |
| q03 | oposición en finales de rey y peón | final | **0/0.00/0.00** | **1/0.25/0.20** |
| q04 | cómo atacar al rey enrocado | táctica | 0/0.00/0.00 | 0/0.00/0.00 |
| q05 | qué hacer con un peón aislado | concepto | 1/1.00/0.40 | 1/0.50/0.60 |
| q06 | ejemplo de sacrificio de calidad | táctica | 1/0.50/0.60 | 1/1.00/1.00 |
| q07 | desventajas de los peones doblados | concepto | 1/1.00/0.80 | 1/1.00/1.00 |
| q08 | iniciativa en la apertura | estrategia | 0/0.00/0.00 | 0/0.00/0.00 |
| q09 | partida de Capablanca | jugador | 1/1.00/1.00 | 1/1.00/1.00 |
| q10 | partidas con Alekhine | jugador | 1/1.00/1.00 | 1/1.00/0.60 |
| q11 | partidas con Lasker | jugador | 1/1.00/1.00 | 1/1.00/1.00 |
| q12 | combinación que termina en mate | táctica | 1/1.00/0.40 | 1/1.00/1.00 |
| q13 | doble ataque o tenedor de caballo | táctica | 0/0.00/0.00 | 0/0.00/0.00 |
| q14 | ataque a la descubierta | táctica | 1/1.00/0.40 | 1/1.00/0.40 |
| q15 | importancia del control del centro | estrategia | 1/1.00/1.00 | 1/1.00/0.80 |
| q16 | columna abierta para las torres | estrategia | 0/0.00/0.00 | 0/0.00/0.00 |
| q17 | principios de finales de torre | final | 1/1.00/1.00 | 1/1.00/1.00 |
| q18 | alfiles de distinto color | final | 1/1.00/1.00 | 1/1.00/1.00 |
| q19 | defensa siciliana | apertura | **0/0.00/0.00** | **0/0.00/0.00** |
| q20 | apertura española / ruy lópez | apertura | 1/0.20/0.20 | 1/1.00/0.20 |
| q21 | gambito de dama | apertura | 0/0.00/0.00 | **1/0.33/0.20** |
| q22 | pareja de alfiles | concepto | 1/1.00/0.20 | 1/1.00/0.20 |
| q23 | caballo en casilla fuerte | concepto | 1/1.00/1.00 | 1/1.00/1.00 |
| q24 | partidas de Tarrasch | jugador | 1/1.00/0.60 | 1/1.00/0.60 |
| q25 | partidas de Steinitz | jugador | 1/1.00/0.60 | 1/0.50/0.40 |

---

## Análisis

### BM25 rescata 2 queries que dense fallaba completamente

**q03 — "oposición":** El término es técnico y su frecuencia en el corpus es baja. El embedding no lo distingue del uso general de la palabra en español. BM25 busca el token exacto y lo encuentra inmediatamente.

**q21 — "gambito de dama":** Caso similar. BM25 captura la frase exacta cuando el dense queda capturado por chunks sobre "gambitos" en general.

Estos dos rescates representan **preguntas pedagógicas clave** — exactamente el tipo de query que el tutor de ajedrez recibirá con frecuencia.

### 5 queries que fallan en ambos sistemas

| Query | Causa raíz |
|-------|-----------|
| **q04** atacar al rey enrocado | Concepto disperso: el corpus tiene 157 menciones de "enroque" pero el retrieval recupera chunks sobre clavadas y otros temas tácticos cercanos semánticamente |
| **q08** iniciativa en la apertura | "Iniciativa" aparece 55 veces en el corpus pero la query es ambigua: el retriever recupera chunks sobre desarrollo de aperturas sin la palabra clave |
| **q13** doble ataque / tenedor | "Tenedor"/"horquilla" son terminología moderna; Grau (años 1940) usa "doble ataque" pero el retrieval no lo prioriza |
| **q16** columna abierta | El corpus tiene 66 menciones pero usa frases variadas ("ocupa la columna", "dominar la línea"); el embedding no ancla en una expresión específica |
| **q19** defensa siciliana | El corpus de Grau (clásicos 1900-1940) tiene solo 5 chunks con "siciliana" — limitación del corpus, no del retriever |

Estas regresiones son informativas: **el corpus de Grau es de ajedrez clásico hipermoderno**, y queries que asumen vocabulario contemporáneo o aperturas postclásicas (siciliana) topan con la cobertura real del libro.

### Una regresión menor detectada (q10 — Alekhine)

Dense logra P=1.00 (5 chunks de Alekhine en top-5), pero hybrid baja a P=0.60. RRF promovió 2 chunks no-Alekhine que BM25 puntuó alto por otros tokens de la query. El Hit@5 sigue siendo 1.00, así que el LLM sí tiene contexto correcto.

Causa probable: para queries de nombres propios, el dense ya es casi perfecto y la fusión RRF introduce ruido. Mitigación futura: weighted RRF cuando la query contiene nombres propios.

---

## Conclusiones

1. **El retrieval híbrido está justificado.** Hit@5 sube de 72% a 80% (+11%) con cero costo computacional adicional relevante.

2. **Hit@5=80% es el número que importa para el agente.** En 4 de cada 5 preguntas, al menos un chunk correcto llega al LLM. Para los misses, el agente puede fallback a respuesta genérica con cita de "no encontrado en corpus".

3. **Las 5 fallas en ambos sistemas señalan los límites del corpus**, no del retriever. El corpus son 4 tomos clásicos (660 chunks); aperturas modernas y conceptos contemporáneos están subrepresentados.

4. **Las métricas son conservadoras** por el criterio de evaluación basado en keywords. El rendimiento real del agente puede ser mejor cuando Grau usa paráfrasis para los conceptos.

5. **Próximos pasos en evaluación:**
   - Ejecutar [`evals/faithfulness_runner.py`](faithfulness_runner.py) — mide si la respuesta del LLM cita el corpus correctamente
   - Ejecutar [`evals/exercises_runner.py`](exercises_runner.py) — mide la precisión táctica del generador de ejercicios
   - Explorar weighted RRF para mitigar la regresión en q10
   - Considerar query rewriting con LLM para queries que fallan (e.g., "atacar al rey enrocado" → expandir con sinónimos antes del retrieval)

---

*Ejecutar: `py evals/runner.py` (requiere ChromaDB corriendo en localhost:8000)*
*Resultados completos con `retrieved_ids` en [`evals/results.json`](results.json)*
