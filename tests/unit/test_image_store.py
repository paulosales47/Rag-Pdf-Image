from unittest.mock import patch

from rag_pdf.vectorstore.chroma_image_store import ImageVectorStore


def _make_store(tmp_path):
    patcher = patch("rag_pdf.vectorstore.chroma_image_store.settings")
    mock_settings = patcher.start()
    mock_settings.vectorstore_path = tmp_path
    mock_settings.image_vectorstore_collection_visual = "images_visual"
    mock_settings.image_vectorstore_collection_text = "images_text"
    return ImageVectorStore(), patcher


def test_add_and_count(tmp_path):
    store, patcher = _make_store(tmp_path)
    try:
        store.add(
            id="foto.png",
            visual_embedding=[0.1, 0.2, 0.3],
            text_embedding=[0.4, 0.5],
            description="uma foto de teste",
            metadata={"source": "foto.png", "image_path": "/tmp/foto.png", "folder": ""},
        )
        assert store.count() == 1
    finally:
        patcher.stop()


def test_query_by_text_and_by_image(tmp_path):
    store, patcher = _make_store(tmp_path)
    try:
        store.add(
            id="foto.png",
            visual_embedding=[1.0, 0.0, 0.0],
            text_embedding=[1.0, 0.0],
            description="um gato preto",
            metadata={"source": "foto.png", "image_path": "/tmp/foto.png", "folder": ""},
        )

        text_hits = store.query_by_text([1.0, 0.0], top_k=1)
        assert len(text_hits) == 1
        assert text_hits[0]["description"] == "um gato preto"
        assert text_hits[0]["metadata"]["source"] == "foto.png"

        image_hits = store.query_by_image([1.0, 0.0, 0.0], top_k=1)
        assert len(image_hits) == 1
        assert image_hits[0]["metadata"]["source"] == "foto.png"
    finally:
        patcher.stop()


def test_list_images(tmp_path):
    store, patcher = _make_store(tmp_path)
    try:
        store.add(
            id="b.png",
            visual_embedding=[0.1, 0.2],
            text_embedding=[0.1, 0.2],
            description="b",
            metadata={"source": "b.png", "image_path": "/tmp/b.png", "folder": ""},
        )
        store.add(
            id="a.png",
            visual_embedding=[0.3, 0.4],
            text_embedding=[0.3, 0.4],
            description="a",
            metadata={"source": "a.png", "image_path": "/tmp/a.png", "folder": ""},
        )

        images = store.list_images()
        assert [img["source"] for img in images] == ["a.png", "b.png"]
    finally:
        patcher.stop()


def test_delete_image(tmp_path):
    store, patcher = _make_store(tmp_path)
    try:
        store.add(
            id="foto.png",
            visual_embedding=[0.1, 0.2],
            text_embedding=[0.1, 0.2],
            description="foto",
            metadata={"source": "foto.png", "image_path": "/tmp/foto.png", "folder": ""},
        )
        store.delete_image("foto.png")
        assert store.count() == 0
    finally:
        patcher.stop()
