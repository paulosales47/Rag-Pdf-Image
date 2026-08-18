from pathlib import Path

import streamlit as st

from rag_pdf.app.ui_helpers import confirm_icon_action
from rag_pdf.config import settings
from rag_pdf.pipeline.image_ingest_pipeline import run_ingest_image
from rag_pdf.pipeline.image_query_pipeline import (
    run_image_siglip_text_query,
    run_image_similarity_query,
    run_image_text_query,
)
from rag_pdf.vectorstore.chroma_image_store import ImageVectorStore

TEXT_SEARCH = "Por texto"
IMAGE_SEARCH = "Por imagem parecida"
SIGLIP_TEXT_SEARCH = "Por texto (nativo do SigLIP)"


@st.cache_resource
def get_image_store() -> ImageVectorStore:
    return ImageVectorStore()


def render_upload_section() -> None:
    st.subheader("Indexar imagens")
    uploaded = st.file_uploader(
        "Importar imagens",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="image_uploader",
    )
    if uploaded and st.button("Indexar imagens", type="primary", key="index_images_btn"):
        settings.images_raw_path.mkdir(parents=True, exist_ok=True)
        for file in uploaded:
            dest = settings.images_raw_path / file.name
            dest.write_bytes(file.getvalue())
            with st.spinner(f"Descrevendo e indexando {file.name}..."):
                try:
                    result = run_ingest_image(dest)
                    get_image_store.clear()
                    st.success(f"{file.name} indexada.")
                    st.caption(result.description)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Falha ao indexar {file.name}: {exc}")


def render_gallery_section() -> None:
    st.subheader("Imagens indexadas")
    try:
        images = get_image_store().list_images()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Vector store de imagens indisponível: {exc}")
        return

    if not images:
        st.caption("Nenhuma imagem indexada ainda.")
        return

    cols = st.columns(4)
    for i, meta in enumerate(images):
        with cols[i % 4]:
            image_path = Path(meta["image_path"])
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            st.caption(meta["source"])
            delete_key = f"delete_image::{meta['source']}"
            if confirm_icon_action("🗑️", key=delete_key, help="Apagar imagem"):
                get_image_store().delete_image(meta["source"])
                get_image_store.clear()
                st.rerun()


def render_result(result) -> None:
    image_path = Path(result.image_path)
    cols = st.columns([1, 3])
    if image_path.exists():
        cols[0].image(str(image_path), use_container_width=True)
    with cols[1]:
        st.markdown(f"**{result.source}**" + (f" — {result.folder}" if result.folder else ""))
        st.caption(f"distância: {result.distance:.3f}")
        st.write(result.description)


def render_search_section() -> None:
    st.subheader("Buscar imagens")
    mode = st.radio(
        "Buscar",
        [TEXT_SEARCH, SIGLIP_TEXT_SEARCH, IMAGE_SEARCH],
        horizontal=True,
        key="image_search_mode",
    )

    if mode == TEXT_SEARCH:
        query = st.text_input("Descreva o que você procura", key="image_text_query")
        if st.button("Buscar", key="image_text_search_btn") and query:
            with st.spinner("Buscando..."):
                try:
                    results = run_image_text_query(query)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Erro na busca: {exc}")
                    return
            if not results:
                st.caption("Nenhuma imagem indexada ainda.")
            for result in results:
                render_result(result)
    elif mode == SIGLIP_TEXT_SEARCH:
        st.caption(
            "Compara o texto direto com o embedding visual (SigLIP), sem passar pela "
            "descrição do modelo de visão — bom pra palavras/frases curtas. As distâncias "
            "aqui tendem a ficar mais altas que no modo 'Por texto' (é uma característica "
            "do SigLIP, não indica busca pior); o que importa é a ordem dos resultados."
        )
        query = st.text_input(
            "O que você procura (palavra ou frase curta)", key="image_siglip_text_query"
        )
        if st.button("Buscar", key="image_siglip_text_search_btn") and query:
            with st.spinner("Buscando..."):
                try:
                    results = run_image_siglip_text_query(query)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Erro na busca: {exc}")
                    return
            if not results:
                st.caption("Nenhuma imagem indexada ainda.")
            for result in results:
                render_result(result)
    else:
        query_file = st.file_uploader(
            "Suba uma imagem parecida com o que você procura",
            type=["png", "jpg", "jpeg", "webp"],
            key="image_similarity_query",
        )
        if query_file and st.button("Buscar", key="image_similarity_search_btn"):
            settings.images_raw_path.mkdir(parents=True, exist_ok=True)
            query_path = settings.images_raw_path / f"_query_{query_file.name}"
            query_path.write_bytes(query_file.getvalue())
            with st.spinner("Buscando..."):
                try:
                    results = run_image_similarity_query(query_path)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Erro na busca: {exc}")
                    return
                finally:
                    query_path.unlink(missing_ok=True)
            if not results:
                st.caption("Nenhuma imagem indexada ainda.")
            for result in results:
                render_result(result)


def render_image_tab() -> None:
    with st.sidebar:
        st.divider()
        try:
            st.metric("Imagens no vector store", get_image_store().count())
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Vector store de imagens indisponível: {exc}")

    render_search_section()
    st.divider()
    render_upload_section()
    st.divider()
    render_gallery_section()
