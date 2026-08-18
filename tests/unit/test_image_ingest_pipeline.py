from unittest.mock import MagicMock, patch

from PIL import Image

from rag_pdf.pipeline.image_ingest_pipeline import run_ingest_image


def _make_image(tmp_path):
    path = tmp_path / "foto.png"
    Image.new("RGB", (10, 20), color="red").save(path)
    return path


@patch("rag_pdf.pipeline.image_ingest_pipeline.ImageVectorStore")
@patch("rag_pdf.pipeline.image_ingest_pipeline.embed_image", return_value=[0.1, 0.2])
@patch("rag_pdf.pipeline.image_ingest_pipeline.embed_texts", return_value=[[0.3, 0.4]])
@patch("rag_pdf.pipeline.image_ingest_pipeline.describe_image", return_value="uma imagem vermelha")
def test_run_ingest_image_stores_both_embeddings_and_metadata(
    mock_describe, mock_embed_texts, mock_embed_image, mock_store_cls, tmp_path
):
    image_path = _make_image(tmp_path)
    mock_store = MagicMock()
    mock_store.count.return_value = 1
    mock_store_cls.return_value = mock_store

    result = run_ingest_image(image_path, folder="Fotos")

    assert result.source == "foto.png"
    assert result.description == "uma imagem vermelha"
    mock_describe.assert_called_once_with(image_path)
    mock_embed_image.assert_called_once_with(image_path)
    mock_embed_texts.assert_called_once_with(["uma imagem vermelha"])

    mock_store.add.assert_called_once()
    _, kwargs = mock_store.add.call_args
    assert kwargs["id"] == "foto.png"
    assert kwargs["visual_embedding"] == [0.1, 0.2]
    assert kwargs["text_embedding"] == [0.3, 0.4]
    assert kwargs["metadata"]["folder"] == "Fotos"
    assert kwargs["metadata"]["width"] == 10
    assert kwargs["metadata"]["height"] == 20
