from functools import lru_cache

from openai import OpenAI

from rag_pdf.config import settings


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    return OpenAI(base_url=settings.lm_studio_base_url, api_key=settings.lm_studio_api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Gera embeddings para documentos a indexar, via EmbeddingGemma no LM Studio.

    O EmbeddingGemma espera prompts com prefixo de tarefa distintos para
    documentos ("title: none | text: ...") e consultas ("task: search
    result | query: ...") — isso melhora a qualidade da busca.
    """
    prefixed = [f"title: none | text: {t}" for t in texts]
    response = _get_client().embeddings.create(
        model=settings.lm_studio_embedding_model, input=prefixed
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def embed_query(query: str) -> list[float]:
    response = _get_client().embeddings.create(
        model=settings.lm_studio_embedding_model,
        input=f"task: search result | query: {query}",
    )
    return response.data[0].embedding
