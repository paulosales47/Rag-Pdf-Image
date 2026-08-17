from unittest.mock import MagicMock, patch

from rag_pdf.retrieval.retriever import retrieve, retrieve_adaptive


@patch("rag_pdf.retrieval.retriever.embed_query", return_value=[0.1, 0.2, 0.3])
def test_retrieve_maps_store_results_to_retrieved_chunks(mock_embed):
    store = MagicMock()
    store.query.return_value = [
        {
            "text": "trecho relevante",
            "metadata": {"source": "doc.pdf", "page_number": 2},
            "distance": 0.12,
        }
    ]

    results = retrieve("pergunta", store, top_k=1)

    assert len(results) == 1
    assert results[0].text == "trecho relevante"
    assert results[0].source == "doc.pdf"
    assert results[0].page_number == 2
    store.query.assert_called_once_with([0.1, 0.2, 0.3], top_k=1, where=None)


@patch("rag_pdf.retrieval.retriever.embed_query", return_value=[0.1, 0.2, 0.3])
def test_retrieve_with_sources_filters_by_source(mock_embed):
    store = MagicMock()
    store.query.return_value = []

    retrieve("pergunta", store, top_k=5, sources=["a.pdf", "b.pdf"])

    store.query.assert_called_once_with(
        [0.1, 0.2, 0.3], top_k=5, where={"source": {"$in": ["a.pdf", "b.pdf"]}}
    )


@patch("rag_pdf.retrieval.retriever.embed_query", return_value=[0.1, 0.2, 0.3])
def test_retrieve_adaptive_keeps_only_chunks_within_margin(mock_embed):
    store = MagicMock()
    store.query.return_value = [
        {"text": "melhor", "metadata": {"source": "doc.pdf", "page_number": 1}, "distance": 0.10},
        {"text": "dentro da margem", "metadata": {"source": "doc.pdf", "page_number": 2}, "distance": 0.104},
        {"text": "fora da margem", "metadata": {"source": "doc.pdf", "page_number": 3}, "distance": 0.30},
    ]

    results = retrieve_adaptive("pergunta", store, margin_pct=5.0, max_context_chars=10_000)

    assert [r.text for r in results] == ["melhor", "dentro da margem"]


@patch("rag_pdf.retrieval.retriever.embed_query", return_value=[0.1, 0.2, 0.3])
def test_retrieve_adaptive_stops_at_context_budget(mock_embed):
    store = MagicMock()
    store.query.return_value = [
        {"text": "a" * 100, "metadata": {"source": "doc.pdf", "page_number": 1}, "distance": 0.10},
        {"text": "b" * 100, "metadata": {"source": "doc.pdf", "page_number": 2}, "distance": 0.101},
        {"text": "c" * 100, "metadata": {"source": "doc.pdf", "page_number": 3}, "distance": 0.102},
    ]

    results = retrieve_adaptive("pergunta", store, margin_pct=100.0, max_context_chars=150)

    assert len(results) == 1
    assert results[0].text == "a" * 100


@patch("rag_pdf.retrieval.retriever.embed_query", return_value=[0.1, 0.2, 0.3])
def test_retrieve_adaptive_always_returns_best_even_if_it_exceeds_budget(mock_embed):
    store = MagicMock()
    store.query.return_value = [
        {"text": "a" * 5000, "metadata": {"source": "doc.pdf", "page_number": 1}, "distance": 0.10},
    ]

    results = retrieve_adaptive("pergunta", store, margin_pct=5.0, max_context_chars=100)

    assert len(results) == 1


@patch("rag_pdf.retrieval.retriever.embed_query", return_value=[0.1, 0.2, 0.3])
def test_retrieve_adaptive_empty_store(mock_embed):
    store = MagicMock()
    store.query.return_value = []

    assert retrieve_adaptive("pergunta", store, margin_pct=5.0, max_context_chars=6000) == []


@patch("rag_pdf.retrieval.retriever.embed_query", return_value=[0.1, 0.2, 0.3])
def test_retrieve_adaptive_with_sources_filters_by_source(mock_embed):
    store = MagicMock()
    store.query.return_value = []

    retrieve_adaptive(
        "pergunta", store, margin_pct=5.0, max_context_chars=6000, sources=["a.pdf"]
    )

    _, kwargs = store.query.call_args
    assert kwargs["where"] == {"source": {"$in": ["a.pdf"]}}
