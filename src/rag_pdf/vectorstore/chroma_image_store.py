import chromadb

from rag_pdf.config import settings


class ImageVectorStore:
    """Guarda imagens indexadas em duas coleções Chroma, uma por embedding.

    As dimensões do embedding visual (SigLIP, 1152d) e do embedding da descrição
    (Gemma, 768d) são diferentes, então não cabem numa coleção Chroma só — cada
    coleção só suporta uma dimensão de vetor. As duas coleções compartilham o
    mesmo `id` (nome do arquivo) e os mesmos metadados, então um resultado de
    qualquer uma das duas dá acesso à imagem e à descrição completas.
    """

    def __init__(self) -> None:
        settings.vectorstore_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(settings.vectorstore_path))
        self._visual = client.get_or_create_collection(
            name=settings.image_vectorstore_collection_visual,
            metadata={"hnsw:space": "cosine"},
        )
        self._text = client.get_or_create_collection(
            name=settings.image_vectorstore_collection_text,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        id: str,
        visual_embedding: list[float],
        text_embedding: list[float],
        description: str,
        metadata: dict,
    ) -> None:
        self._visual.upsert(
            ids=[id], embeddings=[visual_embedding], documents=[description], metadatas=[metadata]
        )
        self._text.upsert(
            ids=[id], embeddings=[text_embedding], documents=[description], metadatas=[metadata]
        )

    def query_by_text(self, embedding: list[float], top_k: int) -> list[dict]:
        return self._query(self._text, embedding, top_k)

    def query_by_image(self, embedding: list[float], top_k: int) -> list[dict]:
        return self._query(self._visual, embedding, top_k)

    @staticmethod
    def _query(collection, embedding: list[float], top_k: int) -> list[dict]:
        result = collection.query(query_embeddings=[embedding], n_results=top_k)
        return [
            {"description": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]

    def count(self) -> int:
        return self._text.count()

    def list_images(self) -> list[dict]:
        result = self._text.get(include=["metadatas"])
        return sorted(result["metadatas"], key=lambda meta: meta["source"])

    def delete_image(self, source: str) -> None:
        self._visual.delete(ids=[source])
        self._text.delete(ids=[source])
