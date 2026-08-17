from rag_pdf.ingestion.cleaner import clean_text


def test_clean_text_collapses_whitespace_and_blank_lines():
    dirty = "Título   com espaços\n\n\n\nParágrafo seguinte.\x00"

    result = clean_text(dirty)

    assert "   " not in result
    assert "\n\n\n" not in result
    assert "\x00" not in result
    assert result.startswith("Título com espaços")
