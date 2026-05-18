"""Detección de errores transitorios y de configuración para LLMs.

Agnóstico al provider: usa keywords en el mensaje de excepción porque
cada SDK (Groq, OpenAI, Anthropic) lanza sus propias clases de error,
pero el texto del mensaje es suficientemente consistente para distinguir.
"""
from __future__ import annotations


# Errores que vale la pena reintentar: son transitorios y desaparecen solos.
_TRANSIENT_KEYWORDS = (
    "tool_use_failed",           # Groq + Llama: JSON malformado en tool_calls
    "failed to call a function", # Groq tool_use variant
    "malformed",                 # JSON malformado genérico
    "invalid json",
    "rate limit",                # Rate limiting de cualquier provider
    "429",                       # HTTP rate limit
    "timeout", "timed out",      # Red / latencia
    "connection",                # Red
    "502", "503", "504",         # Errores HTTP transitorios
    "overloaded",                # Anthropic 529
    "service unavailable",
    "try again",
)

# Errores de configuración: no tienen sentido reintentar porque son permanentes
# hasta que el usuario corrige .env o sus credenciales.
_CONFIG_ERROR_KEYWORDS = (
    "model_not_found",           # OpenAI/Groq: modelo no existe
    "does not exist",            # OpenAI: "The model X does not exist"
    "no access to it",           # OpenAI: "you do not have access to it"
    "invalid_api_key",           # Cualquier provider
    "incorrect api key",
    "authentication",            # Auth genérico
    "401", "403",                # HTTP auth errors
    "404",                       # Not found (modelo o endpoint)
    "not found",                 # Modelo no encontrado
    "invalid request",           # Petición inválida permanente
)


def is_transient_error(exc: BaseException) -> bool:
    """True si el error es transitorio y vale la pena reintentar."""
    msg = str(exc).lower()
    # Un error de configuración nunca es transitorio aunque contenga
    # palabras como "timeout" en el mensaje combinado.
    if is_config_error(exc):
        return False
    return any(kw in msg for kw in _TRANSIENT_KEYWORDS)


def is_config_error(exc: BaseException) -> bool:
    """True si el error indica un problema de configuración permanente."""
    msg = str(exc).lower()
    return any(kw in msg for kw in _CONFIG_ERROR_KEYWORDS)
