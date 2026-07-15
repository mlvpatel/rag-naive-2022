"""Configuration for RagFlow, the naive RAG baseline (2022 generation).

Settings load from environment variables, with a .env file supported for local
development. Naive RAG has few knobs on purpose, so this stays short.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Provider selection. "openai" is the authentic 2022 default. "ollama"
    # runs fully local at no cost for a keyless demo.
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-ada-002"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"

    # Generation. The model name decides the provider: gpt* uses OpenAI,
    # llama*, qwen*, mistral* use Ollama.
    llm_model: str = "gpt-3.5-turbo"

    # Keys (optional for a local Ollama run).
    openai_api_key: str = ""

    # Vector store.
    chroma_dir: str = "./chroma_db"
    collection_name: str = "ragflow_documents"

    # Retrieval. Naive RAG performs a single dense similarity search with no
    # reranking and no hybrid search.
    top_k: int = 4
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # API. This baseline is a local reference service, so the default origin is
    # the local frontend only. Widen it deliberately, never by default.
    cors_origins: str = "http://localhost:8501"

    # Uploads are embedded in full, so an unbounded file is an unbounded bill
    # and an unbounded disk. Cap it.
    max_upload_mb: int = 25


@lru_cache
def get_settings() -> Settings:
    return Settings()
