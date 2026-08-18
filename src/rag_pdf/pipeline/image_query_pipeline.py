from dataclasses import dataclass
from pathlib import Path

from rag_pdf.processing.embedder import embed_query
from rag_pdf.processing.image_embedder import embed_image, embed_text_siglip
from rag_pdf.vectorstore.chroma_image_store import ImageVectorStore


@dataclass
class ImageResult:
    source: str
    image_path: str
    description: str
    distance: float
    folder: str


def _to_result(hit: dict) -> ImageResult:
    meta = hit["metadata"]
    return ImageResult(
        source=meta["source"],
        image_path=meta["image_path"],
        description=hit["description"],
        distance=hit["distance"],
        folder=meta.get("folder", ""),
    )


def run_image_text_query(query: str, top_k: int = 4) -> list[ImageResult]:
    store = ImageVectorStore()
    if store.count() == 0:
        return []
    embedding = embed_query(query)
    hits = store.query_by_text(embedding, top_k=top_k)
    return [_to_result(hit) for hit in hits]


def run_image_similarity_query(query_image_path: Path, top_k: int = 4) -> list[ImageResult]:
    store = ImageVectorStore()
    if store.count() == 0:
        return []
    embedding = embed_image(query_image_path)
    hits = store.query_by_image(embedding, top_k=top_k)
    return [_to_result(hit) for hit in hits]


def run_image_siglip_text_query(query: str, top_k: int = 4) -> list[ImageResult]:
    """Busca por texto direto no espaço visual do SigLIP (sem passar pela descrição
    do modelo de visão) — compara com os mesmos embeddings usados na busca por
    imagem parecida, já que texto e imagem caem no mesmo espaço vetorial.
    """
    store = ImageVectorStore()
    if store.count() == 0:
        return []
    embedding = embed_text_siglip(query)
    hits = store.query_by_image(embedding, top_k=top_k)
    return [_to_result(hit) for hit in hits]
