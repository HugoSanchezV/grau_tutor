from langchain_core.language_models import BaseChatModel
from core.config import settings

_PROVIDERS = {
    "anthropic": "_build_anthropic",
    "groq": "_build_groq",
    "openai": "_build_openai",
}


def _build_anthropic(model: str, temperature: float) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=model,
        anthropic_api_key=settings.anthropic_api_key,
        temperature=temperature,
    )


def _build_groq(model: str, temperature: float) -> BaseChatModel:
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=model,
        groq_api_key=settings.groq_api_key,
        temperature=temperature,
    )


def _build_openai(model: str, temperature: float) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        openai_api_key=settings.openai_api_key,
        temperature=temperature,
    )


def get_llm(
    temperature: float = 0.2,
    provider: str | None = None,
    model: str | None = None,
) -> BaseChatModel:
    resolved_provider = (provider or settings.llm_provider).lower()
    resolved_model = model or settings.llm_model

    builder_name = _PROVIDERS.get(resolved_provider)
    if builder_name is None:
        raise ValueError(
            f"LLM_PROVIDER '{resolved_provider}' no reconocido. "
            f"Opciones: {list(_PROVIDERS)}"
        )

    builder = globals()[builder_name]
    return builder(resolved_model, temperature)
