from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM
    llm_provider: str = Field("groq", alias="LLM_PROVIDER")  # groq | anthropic | openai
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    groq_api_key: str = Field("", alias="GROQ_API_KEY")
    llm_model: str = Field("llama-3.3-70b-versatile", alias="LLM_MODEL")

    # Embeddings
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    embedding_model: str = Field("text-embedding-3-small", alias="EMBEDDING_MODEL")

    # ChromaDB
    chroma_host: str = Field("localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(8000, alias="CHROMA_PORT")
    chroma_collection: str = Field("grau_partidas", alias="CHROMA_COLLECTION")

    # SQLite
    sqlite_db_path: str = Field("db/chess_tutor.db", alias="SQLITE_DB_PATH")

    # BM25 (índice léxico para retrieval híbrido)
    bm25_index_path: str = Field("db/bm25.pkl", alias="BM25_INDEX_PATH")

    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Data
    data_dir: str = Field("data", alias="DATA_DIR")

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
        "populate_by_name": True,
    }


settings = Settings()
