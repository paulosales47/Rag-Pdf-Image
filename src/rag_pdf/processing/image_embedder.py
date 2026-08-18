from functools import lru_cache
from pathlib import Path

import httpx

from rag_pdf.config import settings


@lru_cache(maxsize=1)
def _get_client() -> httpx.Client:
    return httpx.Client(base_url=settings.crispembed_base_url, timeout=60.0)


def embed_image(image_path: Path) -> list[float]:
    """Gera o embedding visual (SigLIP, 1152d) de uma imagem via CrispEmbed.

    CrispEmbed roda localmente com aceleração CUDA (o LM Studio ainda não suporta
    esse modelo) e lê a imagem por caminho no disco do servidor, não por upload.
    """
    # Servidor precisa ter sido iniciado com --vit (não -m) para expor esse endpoint —
    # ver embedding_image_server/start-server.ps1. Formato confirmado lendo
    # examples/server/server.cpp do CrispEmbed: {"embedding": [...], "dim": N}.
    response = _get_client().post("/vit/encode", json={"image": str(image_path.resolve())})
    response.raise_for_status()
    return response.json()["embedding"]
