from pathlib import Path

from loguru import logger

from rag_pdf.ingestion.cleaner import clean_text
from rag_pdf.ingestion.pdf_loader import load_pdf
from rag_pdf.processing.chunker import split_pages
from rag_pdf.processing.embedder import embed_texts
from rag_pdf.vectorstore.chroma_store import ChromaVectorStore


def run_ingest(pdf_path: Path, folder: str = "") -> int:
    pages = load_pdf(pdf_path)
    for page in pages:
        page.text = clean_text(page.text)

    chunks = split_pages(pages)
    if not chunks:
        logger.warning("Nenhum chunk gerado a partir do PDF")
        return 0

    logger.info(f"Gerando embeddings para {len(chunks)} chunks...")
    embeddings = embed_texts([c.text for c in chunks])

    store = ChromaVectorStore()
    store.add(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "source": c.source,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "folder": folder,
            }
            for c in chunks
        ],
    )
    logger.info(f"Indexação concluída. Total no vector store: {store.count()}")
    return len(chunks)
