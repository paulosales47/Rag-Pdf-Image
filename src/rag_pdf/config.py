from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Ingestão / chunking
    chunk_size: int = 700
    chunk_overlap: int = 100

    # Embeddings (servidos pelo LM Studio, endpoint OpenAI-compatible /v1/embeddings)
    lm_studio_embedding_model: str = "text-embedding-embeddinggemma-300m-q8_0"

    # Vector store (Chroma, embutido)
    vectorstore_path: Path = Path("data/vectorstore")
    vectorstore_collection: str = "documents"
    top_k: int = 4

    # LM Studio (servidor local, API compatível com OpenAI)
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_api_key: str = "lm-studio"
    lm_studio_model: str = "llama-3.1-8b-instruct"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048

    # Resumo (map-reduce): tamanho aproximado, em caracteres, de cada bloco
    # de texto enviado ao LLM por chamada. Blocos maiores = menos chamadas =
    # mais rápido, mas exigem um modelo com contexto suficiente no LM Studio.
    summarize_batch_chars: int = 24000


settings = Settings()
