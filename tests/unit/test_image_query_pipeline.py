from unittest.mock import MagicMock, patch

from rag_pdf.pipeline.image_query_pipeline import (
    run_image_siglip_text_query,
    run_image_similarity_query,
    run_image_text_query,
)


@patch("rag_pdf.pipeline.image_query_pipeline.ImageVectorStore")
@patch("rag_pdf.pipeline.image_query_pipeline.embed_query", return_value=[0.1, 0.2])
def test_run_image_text_query_maps_hits_to_results(mock_embed_query, mock_store_cls):
    mock_store = MagicMock()
    mock_store.count.return_value = 1
    mock_store.query_by_text.return_value = [
        {
            "description": "um gato preto",
            "metadata": {"source": "gato.png", "image_path": "/tmp/gato.png", "folder": "Pets"},
            "distance": 0.12,
        }
    ]
    mock_store_cls.return_value = mock_store

    results = run_image_text_query("gato preto", top_k=3)

    assert len(results) == 1
    assert results[0].source == "gato.png"
    assert results[0].image_path == "/tmp/gato.png"
    assert results[0].description == "um gato preto"
    assert results[0].distance == 0.12
    assert results[0].folder == "Pets"
    mock_store.query_by_text.assert_called_once_with([0.1, 0.2], top_k=3)


@patch("rag_pdf.pipeline.image_query_pipeline.ImageVectorStore")
def test_run_image_text_query_returns_empty_when_store_is_empty(mock_store_cls):
    mock_store = MagicMock()
    mock_store.count.return_value = 0
    mock_store_cls.return_value = mock_store

    assert run_image_text_query("qualquer coisa") == []
    mock_store.query_by_text.assert_not_called()


@patch("rag_pdf.pipeline.image_query_pipeline.ImageVectorStore")
@patch("rag_pdf.pipeline.image_query_pipeline.embed_image", return_value=[0.5, 0.6])
def test_run_image_similarity_query_maps_hits_to_results(
    mock_embed_image, mock_store_cls, tmp_path
):
    mock_store = MagicMock()
    mock_store.count.return_value = 1
    mock_store.query_by_image.return_value = [
        {
            "description": "um carro azul",
            "metadata": {"source": "carro.png", "image_path": "/tmp/carro.png", "folder": ""},
            "distance": 0.05,
        }
    ]
    mock_store_cls.return_value = mock_store

    query_path = tmp_path / "query.png"
    query_path.write_bytes(b"fake")

    results = run_image_similarity_query(query_path, top_k=2)

    assert len(results) == 1
    assert results[0].source == "carro.png"
    mock_embed_image.assert_called_once_with(query_path)
    mock_store.query_by_image.assert_called_once_with([0.5, 0.6], top_k=2)


@patch("rag_pdf.pipeline.image_query_pipeline.ImageVectorStore")
@patch("rag_pdf.pipeline.image_query_pipeline.embed_text_siglip", return_value=[0.7, 0.8])
def test_run_image_siglip_text_query_searches_visual_collection(mock_embed_text, mock_store_cls):
    mock_store = MagicMock()
    mock_store.count.return_value = 1
    mock_store.query_by_image.return_value = [
        {
            "description": "uma moto vermelha",
            "metadata": {"source": "moto.png", "image_path": "/tmp/moto.png", "folder": ""},
            "distance": 0.93,
        }
    ]
    mock_store_cls.return_value = mock_store

    results = run_image_siglip_text_query("moto", top_k=4)

    assert len(results) == 1
    assert results[0].source == "moto.png"
    mock_embed_text.assert_called_once_with("moto")
    mock_store.query_by_image.assert_called_once_with([0.7, 0.8], top_k=4)


@patch("rag_pdf.pipeline.image_query_pipeline.ImageVectorStore")
def test_run_image_siglip_text_query_returns_empty_when_store_is_empty(mock_store_cls):
    mock_store = MagicMock()
    mock_store.count.return_value = 0
    mock_store_cls.return_value = mock_store

    assert run_image_siglip_text_query("qualquer coisa") == []
    mock_store.query_by_image.assert_not_called()
