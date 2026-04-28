"""Nodos del grafo LangGraph: router, tutor, evaluador, hitl_review."""
from __future__ import annotations
import re
from typing import TYPE_CHECKING, Optional

from langchain_core.messages import AIMessage

from agents.tools.exercise_gen import evaluate_answer, generate_exercise
from core.llm import get_llm
from core.logging import get_logger
from graph.state import TutorState
from memory.exercises import get_used_exercise_ids, mark_exercise_used
from memory.history import add_message
from memory.progress import get_progress_summary, upsert_progress
from rag.retrieval import GrauRetriever

if TYPE_CHECKING:
    from agents.react_agent import GrauAgent

logger = get_logger(__name__)

# Regex para detectar jugadas en SAN o UCI
_SAN_RE = re.compile(
    r"^[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:[+#]|=[NBRQK])?$|^O-O(?:-O)?[+#]?$",
    re.IGNORECASE,
)
_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][nbrq]?$", re.IGNORECASE)

_ROUTER_PROMPT = (
    "Eres un clasificador de intenciones para un tutor de ajedrez.\n"
    "Clasifica el mensaje en exactamente una palabra: 'tutor' o 'evaluador'.\n\n"
    "REGLA CLAVE: ante la duda, responde 'tutor'.\n\n"
    "tutor → pregunta conceptual, teórica, histórica, estratégica, o pide una explicación/partida de ejemplo.\n"
    "evaluador → pide explícitamente un ejercicio interactivo, un problema táctico para resolver, o quiere practicar/jugar.\n\n"
    "Ejemplos:\n"
    "  'qué es la clavada' → tutor\n"
    "  'dame una partida sobre la clavada' → tutor\n"
    "  'explícame el peón pasado' → tutor\n"
    "  'dame un ejercicio de táctica' → evaluador\n"
    "  'quiero practicar clavadas' → evaluador\n"
    "  'ponme un problema para resolver' → evaluador\n\n"
    "Mensaje: {message}\n\n"
    "Responde ÚNICAMENTE con 'tutor' o 'evaluador'."
)

# Palabras que señalan inequívocamente intención de ejercicio interactivo
_EVALUADOR_KEYWORDS = frozenset({
    "ejercicio", "ejercicios", "problema", "problemas", "táctica", "tácticas",
    "practica", "practicar", "practico", "resolver", "resuelve", "resuelvo",
    "jugar", "juega", "quiero jugar", "ponme un", "dame un problema",
    "entrena", "entrenamiento", "test", "puzzles", "puzzle", "gana", "mate",
    "trata de ganar", "gana la partida",
})

# Palabras que señalan inequívocamente intención conceptual/explicativa
_TUTOR_KEYWORDS = frozenset({
    "qué es", "que es", "explica", "explícame", "cómo funciona", "como funciona",
    "historia", "cuándo", "cuando", "por qué", "porque", "diferencia entre",
    "define", "definición", "enseña", "cuéntame", "cuentame", "ejemplo",
})

_TOPIC_PROMPT = (
    "Eres un asistente de ajedrez. Extrae en 1-3 palabras el tema de ajedrez del mensaje.\n"
    "Ejemplos válidos: 'clavada', 'peón pasado', 'apertura española', 'ataque al rey', "
    "'final de torres', 'estructura de peones', 'gambito de dama'.\n"
    "Si el mensaje no tiene un tema de ajedrez claro, responde 'táctica general'.\n"
    "Responde SOLO con el tema, sin explicación ni puntuación.\n\nMensaje: {message}"
)

_CHESS_FALLBACK_TOPIC = "táctica general"
# Mínimo razonable para un tema de ajedrez (e.g. "pin", "fork", "mat")
_MIN_TOPIC_LEN = 3

# LLM del router con cache singleton — evita reinicialización en cada mensaje
_ROUTER_LLM = None
def _get_router_llm():
    global _ROUTER_LLM
    if _ROUTER_LLM is None:
        _ROUTER_LLM = get_llm()
    return _ROUTER_LLM


_FEN_RE = re.compile(r'FEN:\s*(\S+)')
_EXPECTED_MOVE_RE = re.compile(r'EXPECTED_MOVE:\s*(\S+)')


def _extract_exercise_from_reasoning(
    reasoning: list[dict],
) -> tuple[Optional[str], Optional[str]]:
    """Extrae (fen, expected_move) de un reasoning trace que contiene generate_exercise.

    Permite que tutor_node actualice current_fen cuando el agente generó un ejercicio
    internamente, manteniendo sincronía con el evaluador para el turno siguiente.
    """
    for entry in reasoning:
        if entry.get("type") != "tool_result" or entry.get("name") != "generate_exercise":
            continue
        content = entry.get("content", "")
        fen_m = _FEN_RE.search(content)
        move_m = _EXPECTED_MOVE_RE.search(content)
        if fen_m:
            return fen_m.group(1), (move_m.group(1) if move_m else None)
    return None, None


def _last_text(state: TutorState) -> str:
    last = state["messages"][-1]
    content = last.content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def _looks_like_move(text: str) -> bool:
    t = text.strip()
    return bool(_SAN_RE.match(t) or _UCI_RE.match(t))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def _keyword_route(text: str) -> str | None:
    """Devuelve 'tutor' o 'evaluador' si hay señal léxica clara; None si ambiguo."""
    lower = text.lower()
    if any(kw in lower for kw in _TUTOR_KEYWORDS):
        return "tutor"
    if any(kw in lower for kw in _EVALUADOR_KEYWORDS):
        return "evaluador"
    return None


def router_node(state: TutorState) -> dict:
    text = _last_text(state)

    # Fast-path 1: ejercicio activo + mensaje parece jugada → evaluador
    if state.get("current_fen") and _looks_like_move(text):
        logger.debug("router → evaluador (fast-path: jugada detectada)")
        return {"mode": "evaluador"}

    # Fast-path 2: señal léxica inequívoca → evita latencia del LLM
    kw_mode = _keyword_route(text)
    if kw_mode is not None:
        logger.debug(f"router → {kw_mode} (fast-path: palabra clave detectada)")
        return {"mode": kw_mode}

    # Fallback: usar LLM del router en caché (singleton para evitar reinit en cada llamada)
    llm = _get_router_llm()
    resp = llm.invoke(_ROUTER_PROMPT.format(message=text[:500]))
    raw = (resp.content if isinstance(resp.content, str) else str(resp.content)).strip().lower()
    mode = "evaluador" if raw == "evaluador" else "tutor"  # default a tutor si respuesta inesperada
    logger.debug(f"router → {mode} (LLM clasificó: '{raw}')")
    return {"mode": mode}


# ---------------------------------------------------------------------------
# Tutor
# ---------------------------------------------------------------------------

def tutor_node(state: TutorState, agent: "GrauAgent") -> dict:
    text = _last_text(state)
    student_id = state.get("student_id", "alumno")
    thread_id = state.get("thread_id", "default")

    # Inyectar FEN activo como contexto explícito para el turno actual
    current_fen = state.get("current_fen")
    effective_text = (
        f"[Posición activa: {current_fen}]\n{text}" if current_fen else text
    )

    # Pasar el historial del grafo explícitamente al agente (stateless)
    history = state.get("messages", [])
    response = agent.chat(effective_text, thread_id=thread_id, history=history)

    add_message(student_id, thread_id, "user", text)
    add_message(student_id, thread_id, "assistant", response.reply)
    upsert_progress(student_id, tema="consultas_generales", delta_consultas=1)

    # Si el agente generó un ejercicio vía su tool, sincronizar estado del grafo
    # para que el router detecte la jugada del alumno en el turno siguiente.
    exercise_fen, exercise_move = _extract_exercise_from_reasoning(response.reasoning)

    result: dict = {
        "messages": [AIMessage(content=response.reply)],
        "reasoning_trace": response.reasoning,
        "progress_summary": get_progress_summary(student_id),
        "hitl_pending": False,
    }
    if exercise_fen:
        result["current_fen"] = exercise_fen
        result["expected_move"] = exercise_move
    return result


# ---------------------------------------------------------------------------
# Evaluador
# ---------------------------------------------------------------------------

def evaluador_node(state: TutorState, retriever: GrauRetriever) -> dict:
    text = _last_text(state)
    student_id = state.get("student_id", "alumno")
    thread_id = state.get("thread_id", "default")
    current_fen = state.get("current_fen")

    # --- Evaluación de respuesta ---
    if current_fen:
        ev = evaluate_answer(current_fen, text, state.get("expected_move"))
        upsert_progress(
            student_id,
            tema="ejercicios",
            delta_ejercicios_intentados=1,
            delta_ejercicios_correctos=1 if ev.correcta else 0,
        )

        trace = [
            {"type": "tool_call", "name": "evaluate_answer", "args": {"fen": current_fen[:30], "jugada_alumno": text}},
            {"type": "tool_result", "name": "evaluate_answer", "content": ev.feedback[:300]},
        ]

        # El evaluador actualiza state["messages"] directamente (vía AIMessage)
        # El tutor del siguiente turno verá el contexto en el historial del grafo

        # Jugada legal pero incorrecta → HITL (Bug #1: ahora ev.correcta puede ser False)
        if ev.correcta is False and ev.legal:
            logger.info(f"evaluador → hitl_pending=True para student={student_id}")
            return {
                "messages": [AIMessage(content=ev.feedback)],
                "evaluation_reasoning": ev.feedback,
                "hitl_pending": True,
                "reasoning_trace": trace,
            }

        # Correcta o evaluación abierta (sin ground truth)
        new_fen = None if ev.correcta else current_fen
        return {
            "messages": [AIMessage(content=ev.feedback)],
            "current_fen": new_fen,
            "evaluation_reasoning": None,
            "hitl_pending": False,
            "reasoning_trace": trace,
            "progress_summary": get_progress_summary(student_id),
        }

    # --- Generación de ejercicio ---
    llm = get_llm()
    topic_resp = llm.invoke(_TOPIC_PROMPT.format(message=text[:800]))
    raw_tema = (
        topic_resp.content if isinstance(topic_resp.content, str) else str(topic_resp.content)
    ).strip()
    # Sanear: sin puntuación inicial/final, longitud razonable, fallback si incoherente
    tema = re.sub(r"^[^\w\s]+|[^\w\s]+$", "", raw_tema).strip()[:60]
    if len(tema) < _MIN_TOPIC_LEN:
        logger.warning(f"evaluador → tema extraído inválido '{raw_tema}', usando fallback")
        tema = _CHESS_FALLBACK_TOPIC

    trace = [
        {"type": "tool_call", "name": "search_grau", "args": {"query": tema}},
    ]

    used_ids = frozenset(get_used_exercise_ids(student_id))
    ejercicio = generate_exercise(retriever, tema=tema, exclude_ids=used_ids)
    if ejercicio is None:
        exhausted = len(used_ids) > 0
        reply = (
            f"No encontré posiciones nuevas del corpus de Grau para el tema '{tema}'. "
            + ("Ya practicaste todos los ejercicios disponibles sobre este tema. "
               if exhausted else "")
            + "Prueba con otro tema o pide una explicación conceptual."
        )
        trace.append({"type": "tool_result", "name": "search_grau", "content": f"Sin FEN disponibles para '{tema}'"})
        return {"messages": [AIMessage(content=reply)], "reasoning_trace": trace}

    mark_exercise_used(student_id, ejercicio.partida_id)

    trace.append({
        "type": "tool_result", "name": "search_grau",
        "content": f"Tomo {ejercicio.tomo} | partida {ejercicio.partida_id} | FEN={ejercicio.fen[:30]}...",
    })
    trace.append({"type": "tool_call", "name": "pick_best_move", "args": {"fen": ejercicio.fen[:30]}})
    trace.append({
        "type": "tool_result", "name": "pick_best_move",
        "content": "Jugada candidata: [evaluación interna]",
    })

    reply = (
        f"**Ejercicio — Tomo {ejercicio.tomo}** (tema: {tema})\n\n"
        f"FEN: `{ejercicio.fen}`\n\n"
        f"{ejercicio.pregunta}\n\n"
        f"---\n*Grau: {ejercicio.comentario_grau[:300]}{'...' if len(ejercicio.comentario_grau) > 300 else ''}*"
    )
    logger.info(f"evaluador → ejercicio generado FEN={ejercicio.fen[:30]}... para tema='{tema}'")

    return {
        "messages": [AIMessage(content=reply)],
        "current_fen": ejercicio.fen,
        "expected_move": ejercicio.jugada_correcta,
        "hitl_pending": False,
        "reasoning_trace": trace,
    }


# ---------------------------------------------------------------------------
# HITL Review
# ---------------------------------------------------------------------------

def hitl_review_node(state: TutorState) -> dict:
    from datetime import datetime, timezone
    from memory.database import get_connection, init_db

    decision = (state.get("hitl_decision") or "acepto").strip().lower()
    student_id = state.get("student_id", "alumno")
    reasoning = state.get("evaluation_reasoning") or "La jugada no coincide con la esperada."
    current_fen = state.get("current_fen")

    if "disputo" in decision:
        # El alumno disputa la evaluación
        # Validar que haya proporcionado razonamiento mínimo
        dispute_text = state.get("hitl_decision", "").strip()
        has_reasoning = len(dispute_text) > 20  # Mínimo de 20 caracteres de razonamiento

        if not has_reasoning:
            reply = (
                "Para disputar una evaluación, por favor proporciona un razonamiento "
                "detallado (mínimo 20 caracteres). ¿Por qué crees que tu jugada es correcta?"
            )
            logger.info(f"hitl_review → disputa rechazada (sin razonamiento) para student={student_id}")
            return {
                "messages": [AIMessage(content=reply)],
                "hitl_pending": True,
                "hitl_decision": None,
                "progress_summary": get_progress_summary(student_id),
            }

        # Guardar la disputa en la base de datos para revisión asíncrona
        init_db()
        now = datetime.now(timezone.utc).isoformat()
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO ejercicios_disputados
                (student_id, partida_id, fen, jugada_alumno, jugada_esperada, razonamiento, timestamp_disputa, estado)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'en_disputa')
                """,
                (
                    student_id,
                    "unknown",  # Podría extraerse del state si se guardara
                    current_fen or "",
                    state.get("hitl_decision", ""),
                    "",
                    dispute_text,
                    now,
                ),
            )

        reply = (
            "Tu disputa ha sido registrada. Un tutor la revisará y te notificará del resultado. "
            "Mientras tanto, continúa practicando. ¡Gracias por el feedback!"
        )
        logger.info(f"hitl_review → disputa registrada para student={student_id}")
    else:
        # El alumno acepta la evaluación
        reply = (
            f"Evaluación confirmada. {reasoning} "
            "Revisa el análisis de Grau para esta posición y continúa practicando."
        )
        logger.info(f"hitl_review → evaluación aceptada para student={student_id}")

    return {
        "messages": [AIMessage(content=reply)],
        "hitl_pending": False,
        "hitl_decision": None,
        "evaluation_reasoning": None,
        "current_fen": None,
        "progress_summary": get_progress_summary(student_id),
    }
