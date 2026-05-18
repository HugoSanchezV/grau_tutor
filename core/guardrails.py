"""Detección de prompt injection y mensajes de rechazo.

Capa 1 de defensa: filtros deterministas (regex) ejecutados en el router
ANTES de invocar al LLM. Bloquea manipulación del system prompt sin coste
de inferencia y sin riesgo de que el modelo sea persuadido por el ataque.

El scope (chess vs off-topic) lo decide la capa LLM allowlist en el router
(ver graph/nodes.py: _ROUTER_PROMPT). Aquí NO se enumera "lo prohibido".
"""
from __future__ import annotations
import re


# Patrones de manipulación del system prompt o cambio de rol.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignor[ae][rs]?\s+(las?\s+)?(instrucciones?|reglas?|prompts?)", re.I),
    re.compile(r"olvid[ae][rs]?\s+(tu\s+)?(rol|reglas?|instrucciones?|sistema)", re.I),
    re.compile(r"ignore\s+(?:(?:previous|prior|above|all)\s+){1,3}instructions?", re.I),
    re.compile(r"disregard\s+(?:(?:previous|prior|above|all)\s+){1,3}instructions?", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"\bsystem\s*:\s*", re.I),
    re.compile(r"reveal\s+(your|the)\s+(system\s+)?prompt", re.I),
    re.compile(r"(muestra|muéstrame|dame|imprime|revela)\s+(tu|el)\s+(prompt|sistema|instrucciones)", re.I),
    re.compile(r"prompt\s+(inicial|original|de\s+sistema)", re.I),
    re.compile(r"new\s+instructions?\s*:", re.I),
)


REFUSAL_OFF_TOPIC = (
    "Soy el Tutor Grau, especializado en ajedrez basado en el Tratado de "
    "Roberto Grau. No puedo ayudarte con esa consulta. ¿Tienes alguna duda "
    "sobre ajedrez? Puedo explicarte conceptos, darte ejercicios o analizar "
    "posiciones."
)

REFUSAL_INJECTION = (
    "He detectado un intento de modificar mis instrucciones. Mi rol es "
    "enseñar ajedrez basándome en el Tratado de Grau, y no voy a salirme "
    "de él. ¿En qué tema de ajedrez te ayudo?"
)

AGENT_ERROR_FALLBACK = (
    "Tuve un problema técnico procesando tu mensaje. ¿Puedes reformularlo, "
    "por favor? Si pedías un ejercicio, indícame el tema (p.ej. 'táctica', "
    "'final de torres')."
)

AGENT_CONFIG_ERROR_FALLBACK = (
    "El tutor no puede responder: hay un problema de configuración del modelo "
    "de IA (modelo no encontrado o credenciales inválidas). Revisa las variables "
    "LLM_PROVIDER y LLM_MODEL en tu .env y asegúrate de que sean compatibles "
    "(ej. LLM_PROVIDER=openai requiere LLM_MODEL=gpt-4o, no llama-3.3-70b-versatile)."
)

AGENT_RECURSION_FALLBACK = (
    "Necesité demasiados pasos para responder esa pregunta y no llegué a una "
    "conclusión. Intenta reformularla de forma más concreta, por ejemplo "
    "indicando el tema exacto ('clavada', 'enroque', 'final de peones') o la "
    "posición FEN si tienes una."
)


def is_prompt_injection(text: str) -> bool:
    """True si el texto contiene un patrón conocido de manipulación de prompt."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)
