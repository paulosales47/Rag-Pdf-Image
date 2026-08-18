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

    # --- Imagens (busca multimodal) ---
    # Imagens originais salvas em disco (o caminho vai pros metadados, pra exibir depois).
    images_raw_path: Path = Path("data/raw/images")

    # Modelo com suporte a visão, carregado no LM Studio junto com o LLM de chat,
    # usado só para gerar a descrição textual de cada imagem indexada.
    lm_studio_vision_model: str = "gemma4-12b-qat-uncensored-hauhaucs-balanced"
    # Modelos de visão "thinking" gastam parte desse orçamento raciocinando antes de
    # escrever a descrição — por isso o padrão é mais folgado que um max_tokens comum.
    vision_max_tokens: int = 4096

    # CrispEmbed (github.com/CrispStrobe/CrispEmbed): runtime local com build CUDA,
    # servindo o SigLIP GGUF via HTTP — o LM Studio ainda não suporta esse modelo.
    crispembed_base_url: str = "http://localhost:8081"

    # Coleções Chroma para imagens (dimensões diferentes exigem coleções separadas):
    # embedding da imagem em si (SigLIP, 1152d) e embedding da descrição (Gemma, 768d).
    image_vectorstore_collection_visual: str = "images_visual"
    image_vectorstore_collection_text: str = "images_text"


settings = Settings()
