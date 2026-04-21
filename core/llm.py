from langchain_anthropic import ChatAnthropic
from core.config import settings


def get_llm(temperature: float = 0.2) -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.llm_model,
        anthropic_api_key=settings.anthropic_api_key,
        temperature=temperature,
    )
