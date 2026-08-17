"""Exemplo de uso direto do pipeline de ingestão, sem passar pela CLI."""

from pathlib import Path

from rag_pdf.logging_config import configure_logging
from rag_pdf.pipeline.ingest_pipeline import run_ingest

if __name__ == "__main__":
    configure_logging()
    run_ingest(Path("data/raw/documento.pdf"))
