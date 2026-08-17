import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "Teste de integração real: requer um PDF de exemplo em tests/fixtures "
        "e o servidor do LM Studio rodando em http://localhost:1234. "
        "Rode manualmente removendo o skip quando o ambiente estiver de pé."
    )
)


def test_ingest_and_query_end_to_end():
    from pathlib import Path

    from rag_pdf.pipeline.ingest_pipeline import run_ingest
    from rag_pdf.pipeline.query_pipeline import run_query

    n_chunks = run_ingest(Path("tests/fixtures/sample.pdf"))
    assert n_chunks > 0

    result = run_query("Do que trata este documento?")
    assert result.answer
    assert result.sources
