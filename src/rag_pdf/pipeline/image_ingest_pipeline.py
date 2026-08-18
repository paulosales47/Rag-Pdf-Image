from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from PIL import Image

from rag_pdf.llm.vision_client import describe_image
from rag_pdf.processing.embedder import embed_texts
from rag_pdf.processing.image_embedder import embed_image
from rag_pdf.vectorstore.chroma_image_store import ImageVectorStore


@dataclass
class ImageIngestResult:
    source: str
    description: str


def run_ingest_image(image_path: Path, folder: str = "") -> ImageIngestResult:
    with Image.open(image_path) as img:
        width, height = img.size
        image_format = (img.format or image_path.suffix.lstrip(".")).lower()

    logger.info(f"Gerando descrição de {image_path.name} via modelo de visão...")
    description = describe_image(image_path)

    logger.info(f"Gerando embeddings (visual + texto) para {image_path.name}...")
    visual_embedding = embed_image(image_path)
    text_embedding = embed_texts([description])[0]

    store = ImageVectorStore()
    store.add(
        id=image_path.name,
        visual_embedding=visual_embedding,
        text_embedding=text_embedding,
        description=description,
        metadata={
            "source": image_path.name,
            "image_path": str(image_path.resolve()),
            "description": description,
            "folder": folder,
            "width": width,
            "height": height,
            "format": image_format,
        },
    )
    logger.info(f"Indexação concluída. Total de imagens: {store.count()}")
    return ImageIngestResult(source=image_path.name, description=description)
