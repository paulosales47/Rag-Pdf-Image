from pathlib import Path

import typer

from rag_pdf.logging_config import configure_logging
from rag_pdf.pipeline.ingest_pipeline import run_ingest
from rag_pdf.pipeline.query_pipeline import run_query
from rag_pdf.pipeline.summarize_pipeline import SummarizeProgress, list_sources, run_summarize

app = typer.Typer(help="RAG local: PDF -> banco vetorial -> consulta via LM Studio")


@app.command()
def ingest(
    pdf: Path = typer.Option(..., "--pdf", help="Caminho do PDF a indexar"),
    folder: str = typer.Option("", "--folder", help="Pasta virtual (ex.: 'Suporte/2024')"),
) -> None:
    """Extrai, faz chunk, gera embeddings e indexa um PDF no vector store local."""
    configure_logging()
    n_chunks = run_ingest(pdf, folder=folder)
    typer.echo(f"{n_chunks} chunks indexados a partir de {pdf.name}")


@app.command()
def ask(question: str = typer.Argument(..., help="Pergunta sobre o documento indexado")) -> None:
    """Consulta o vector store e pergunta ao LLM servido pelo LM Studio."""
    configure_logging()
    result = run_query(question)
    typer.echo("\n" + result.answer + "\n")
    if result.sources:
        typer.echo("Fontes:")
        for s in result.sources:
            typer.echo(f"  - {s.source} (página {s.page_number}, distância {s.distance:.3f})")


@app.command()
def summarize(
    source: str = typer.Option(
        None,
        "--source",
        help="Nome do PDF indexado (ex.: documento.pdf). Se omitido e houver só um, usa ele.",
    ),
) -> None:
    """Gera um resumo dos principais pontos de um documento indexado (lê o documento inteiro)."""
    configure_logging()
    sources = list_sources()
    if not sources:
        typer.echo("Nenhum documento indexado ainda. Rode `rag-pdf ingest --pdf <arquivo>` primeiro.")
        raise typer.Exit(1)

    if source is None:
        if len(sources) > 1:
            typer.echo("Mais de um documento indexado. Use --source para escolher:")
            for s in sources:
                typer.echo(f"  - {s}")
            raise typer.Exit(1)
        source = sources[0]

    def report(p: SummarizeProgress) -> None:
        remaining = max(p.total - p.step, 0) * p.avg_seconds_per_call
        etapa = "Lendo blocos" if p.stage == "map" else f"Consolidando (nível {p.level})"
        typer.echo(
            f"{etapa}: {p.step}/{p.total} — média {p.avg_seconds_per_call:.1f}s/chamada, "
            f"restante ~{remaining / 60:.1f} min"
        )

    summary = run_summarize(source, on_progress=report)
    typer.echo("\n" + summary + "\n")


if __name__ == "__main__":
    app()
