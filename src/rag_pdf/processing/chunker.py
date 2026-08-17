from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pdf.config import settings
from rag_pdf.ingestion.pdf_loader import PageContent


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page_number: int
    chunk_index: int


def split_pages(pages: list[PageContent]) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for page in pages:
        pieces = splitter.split_text(page.text)
        for idx, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=f"{page.source}-p{page.page_number}-c{idx}",
                    text=piece,
                    source=page.source,
                    page_number=page.page_number,
                    chunk_index=idx,
                )
            )
    return chunks
