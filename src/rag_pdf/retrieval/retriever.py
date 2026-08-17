from dataclasses import dataclass

from rag_pdf.config import settings
from rag_pdf.processing.embedder import embed_query
from rag_pdf.vectorstore.base import VectorStore


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page_number: int
    distance: float


def _to_chunks(matches: list[dict]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text=m["text"],
            source=m["metadata"]["source"],
            page_number=m["metadata"]["page_number"],
            distance=m["distance"],
        )
        for m in matches
    ]


def _sources_filter(sources: list[str] | None) -> dict | None:
    return {"source": {"$in": sources}} if sources else None


def retrieve(
    query: str,
    store: VectorStore,
    top_k: int | None = None,
    sources: list[str] | None = None,
) -> list[RetrievedChunk]:
    query_embedding = embed_query(query)
    matches = store.query(
        query_embedding, top_k=top_k or settings.top_k, where=_sources_filter(sources)
    )
    return _to_chunks(matches)


# Teto de segurança: nº máximo de candidatos buscados no vector store antes de
# aplicar os filtros de margem/contexto, para não pedir uma busca ilimitada.
_SAFETY_MAX_K = 50


def retrieve_adaptive(
    query: str,
    store: VectorStore,
    margin_pct: float,
    max_context_chars: int,
    sources: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Busca adaptativa: pega o trecho mais similar e vai incluindo os

    próximos (em ordem de similaridade) enquanto a distância deles estiver
    dentro de `margin_pct` (%) da distância do melhor resultado, parando
    assim que o primeiro dos dois limites for atingido: a margem ou o teto
    de contexto (`max_context_chars`, em caracteres). Sempre retorna pelo
    menos o melhor trecho, mesmo que ele sozinho já exceda o teto.
    """
    query_embedding = embed_query(query)
    matches = store.query(query_embedding, top_k=_SAFETY_MAX_K, where=_sources_filter(sources))
    if not matches:
        return []

    best_distance = matches[0]["distance"]
    threshold = best_distance * (1 + margin_pct / 100)

    selected: list[dict] = []
    total_chars = 0
    for match in matches:
        if match["distance"] > threshold:
            break
        if selected and total_chars + len(match["text"]) > max_context_chars:
            break
        selected.append(match)
        total_chars += len(match["text"])

    return _to_chunks(selected)
