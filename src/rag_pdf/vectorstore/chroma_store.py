import chromadb

from rag_pdf.config import settings


class ChromaVectorStore:
    def __init__(self) -> None:
        settings.vectorstore_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(settings.vectorstore_path))
        self._collection = client.get_or_create_collection(
            name=settings.vectorstore_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(self, embedding: list[float], top_k: int, where: dict | None = None) -> list[dict]:
        result = self._collection.query(
            query_embeddings=[embedding], n_results=top_k, where=where
        )
        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def count(self) -> int:
        return self._collection.count()

    def get_by_source(self, source: str) -> list[dict]:
        result = self._collection.get(where={"source": source})
        items = [
            {"text": doc, "page_number": meta["page_number"], "chunk_index": meta["chunk_index"]}
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]
        return sorted(items, key=lambda i: (i["page_number"], i["chunk_index"]))

    def list_sources(self) -> list[str]:
        result = self._collection.get(include=["metadatas"])
        return sorted({meta["source"] for meta in result["metadatas"]})

    def list_documents(self) -> list[dict]:
        result = self._collection.get(include=["metadatas"])
        by_source = {meta["source"]: meta.get("folder", "") for meta in result["metadatas"]}
        return sorted(
            ({"source": source, "folder": folder} for source, folder in by_source.items()),
            key=lambda d: (d["folder"], d["source"]),
        )

    def list_folders(self) -> list[str]:
        result = self._collection.get(include=["metadatas"])
        folders = {meta.get("folder", "") for meta in result["metadatas"]}
        return sorted(f for f in folders if f)

    def delete_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def move_source(self, source: str, new_folder: str) -> None:
        result = self._collection.get(where={"source": source}, include=["metadatas"])
        metadatas = [{**meta, "folder": new_folder} for meta in result["metadatas"]]
        self._collection.update(ids=result["ids"], metadatas=metadatas)

    def rename_folder(self, old_folder: str, new_folder: str) -> None:
        result = self._collection.get(include=["metadatas"])
        ids: list[str] = []
        metadatas: list[dict] = []
        for doc_id, meta in zip(result["ids"], result["metadatas"]):
            folder = meta.get("folder", "")
            if folder == old_folder:
                new_value = new_folder
            elif folder.startswith(old_folder + "/"):
                new_value = new_folder + folder[len(old_folder) :]
            else:
                continue
            ids.append(doc_id)
            metadatas.append({**meta, "folder": new_value})
        if ids:
            self._collection.update(ids=ids, metadatas=metadatas)

    def delete_folder(self, folder: str) -> None:
        result = self._collection.get(include=["metadatas"])
        ids = [
            doc_id
            for doc_id, meta in zip(result["ids"], result["metadatas"])
            if meta.get("folder", "") == folder
            or meta.get("folder", "").startswith(folder + "/")
        ]
        if ids:
            self._collection.delete(ids=ids)
