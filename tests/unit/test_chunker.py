from rag_pdf.ingestion.pdf_loader import PageContent
from rag_pdf.processing.chunker import split_pages


def test_split_pages_creates_chunks_with_metadata():
    pages = [PageContent(page_number=1, text="a" * 1500, source="doc.pdf")]

    chunks = split_pages(pages)

    assert len(chunks) > 1
    assert all(c.source == "doc.pdf" for c in chunks)
    assert all(c.page_number == 1 for c in chunks)
    assert chunks[0].id == "doc.pdf-p1-c0"


def test_split_pages_handles_empty_text():
    pages = [PageContent(page_number=1, text="", source="doc.pdf")]

    chunks = split_pages(pages)

    assert chunks == []
