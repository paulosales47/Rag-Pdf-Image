from dataclasses import dataclass
from pathlib import Path

import pypdf
from loguru import logger


@dataclass
class PageContent:
    page_number: int
    text: str
    source: str


def load_pdf(pdf_path: Path) -> list[PageContent]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    reader = pypdf.PdfReader(str(pdf_path))
    pages: list[PageContent] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(PageContent(page_number=i, text=text, source=pdf_path.name))
        else:
            logger.warning(
                f"Página {i} de {pdf_path.name} não retornou texto "
                "(pode ser um PDF escaneado/imagem — considere OCR)"
            )

    logger.info(f"{pdf_path.name}: {len(pages)} páginas com texto extraídas")
    return pages
