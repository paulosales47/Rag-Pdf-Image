from pathlib import Path

import streamlit as st

from rag_pdf.config import settings
from rag_pdf.logging_config import configure_logging
from rag_pdf.pipeline.ingest_pipeline import run_ingest
from rag_pdf.pipeline.query_pipeline import (
    RETRIEVAL_PROFILES,
    RetrievalProfile,
    run_query,
    run_query_adaptive,
)
from rag_pdf.pipeline.summarize_pipeline import SummarizeProgress, run_summarize
from rag_pdf.vectorstore import folder_registry
from rag_pdf.vectorstore.chroma_store import ChromaVectorStore

configure_logging()

st.set_page_config(page_title="RAG PDF Local", page_icon="📄")

TOPK_MODE = "Busca (top-k fixo)"
ADAPTIVE_MODE = "Busca adaptativa (restrita / padrão / abrangente)"
MAPREDUCE_MODE = "Resumo (map-reduce, documento inteiro)"

ROOT_LABEL = "(raiz, sem pasta)"


@st.cache_resource
def get_store() -> ChromaVectorStore:
    return ChromaVectorStore()


def get_all_folders() -> list[str]:
    try:
        from_documents = set(get_store().list_folders())
    except Exception:  # noqa: BLE001
        from_documents = set()
    return sorted(from_documents | set(folder_registry.list_folders()))


def build_tree(documents: list[dict], all_folders: list[str] = ()) -> dict:
    root: dict = {"files": [], "children": {}}

    def ensure_path(parts: list[str]) -> dict:
        node = root
        for part in parts:
            node = node["children"].setdefault(part, {"files": [], "children": {}})
        return node

    for doc in documents:
        parts = [p for p in (doc.get("folder") or "").split("/") if p]
        ensure_path(parts)["files"].append(doc["source"])
    for folder in all_folders:
        ensure_path([p for p in folder.split("/") if p])
    return root


def _all_files_in(node: dict) -> list[str]:
    files = list(node["files"])
    for child in node["children"].values():
        files.extend(_all_files_in(child))
    return files


def selected_sources_from_tree() -> list[str]:
    return [
        key.removeprefix("scope_file::")
        for key, checked in st.session_state.items()
        if key.startswith("scope_file::") and checked
    ]


def confirm_icon_action(icon: str, key: str, help: str | None = None) -> bool:
    """Botão-ícone de ação destrutiva, com confirmação compacta em duas etapas."""
    confirm_key = f"confirm::{key}"
    if not st.session_state.get(confirm_key, False):
        if st.button(icon, key=f"{key}_ask", help=help):
            st.session_state[confirm_key] = True
            st.rerun()
        return False

    st.caption("Confirma?")
    c1, c2 = st.columns(2)
    confirmed = c1.button("✅", key=f"{key}_yes")
    if c2.button("❌", key=f"{key}_no"):
        st.session_state[confirm_key] = False
    if confirmed:
        st.session_state[confirm_key] = False
    return confirmed


def render_unified_tree(node: dict, all_folders: list[str], path: str = "") -> None:
    for fname in sorted(node["files"]):
        cols = st.columns([0.4, 4.6, 0.7, 0.7])
        cols[0].checkbox("", key=f"scope_file::{fname}", label_visibility="collapsed")
        cols[1].write(f"📄 {fname}")
        with cols[2]:
            with st.popover("🔀", help="Mover para outra pasta"):
                target = st.selectbox(
                    "Mover para", [ROOT_LABEL, *all_folders], key=f"move_target::{fname}"
                )
                if st.button("Mover", key=f"move_btn::{fname}"):
                    new_folder = "" if target == ROOT_LABEL else target
                    get_store().move_source(fname, new_folder)
                    folder_registry.create_folder(new_folder)
                    get_store.clear()
                    st.rerun()
        with cols[3]:
            if confirm_icon_action("🗑️", key=f"delete_file::{fname}", help="Apagar arquivo"):
                get_store().delete_source(fname)
                get_store.clear()
                st.rerun()

    for folder_name in sorted(node["children"]):
        full_path = f"{path}/{folder_name}" if path else folder_name
        child = node["children"][folder_name]

        header = st.columns([0.4, 3.3, 0.7, 0.7, 0.7, 1.2])
        folder_key = f"scope_folder::{full_path}"
        prev_key = f"scope_folder_prev::{full_path}"
        folder_checked = header[0].checkbox("", key=folder_key, label_visibility="collapsed")
        header[1].markdown(f"**📁 {folder_name}**")
        with header[2]:
            with st.popover("✏️", help="Renomear pasta"):
                new_name = st.text_input(
                    "Novo caminho", value=full_path, key=f"rename_input::{full_path}"
                )
                if st.button("Salvar", key=f"rename_btn::{full_path}"):
                    if new_name and new_name != full_path:
                        get_store().rename_folder(full_path, new_name)
                        folder_registry.rename_folder(full_path, new_name)
                        get_store.clear()
                        st.rerun()
        with header[3]:
            with st.popover("➕", help="Criar subpasta aqui"):
                new_sub = st.text_input("Nome da subpasta", key=f"newsub_input::{full_path}")
                if st.button("Criar", key=f"newsub_btn::{full_path}") and new_sub.strip():
                    folder_registry.create_folder(f"{full_path}/{new_sub.strip()}")
                    st.rerun()
        with header[4]:
            if confirm_icon_action("🗑️", key=f"delete_folder::{full_path}", help="Apagar pasta e conteúdo"):
                get_store().delete_folder(full_path)
                folder_registry.delete_folder(full_path)
                get_store.clear()
                st.rerun()
        with header[5]:
            open_folder = st.toggle("Abrir", key=f"open_folder::{full_path}", label_visibility="collapsed")

        if folder_checked != st.session_state.get(prev_key, False):
            st.session_state[prev_key] = folder_checked
            for f in _all_files_in(child):
                st.session_state[f"scope_file::{f}"] = folder_checked
            st.rerun()

        if open_folder:
            with st.container(border=True):
                render_unified_tree(child, all_folders, path=full_path)


def render_folder_picker(node: dict, path: str = "") -> None:
    for folder_name in sorted(node["children"]):
        full_path = f"{path}/{folder_name}" if path else folder_name
        with st.expander(f"📁 {folder_name}"):
            if st.button(f"📌 Usar '{full_path}'", key=f"use_folder::{full_path}"):
                st.session_state.upload_target_folder = full_path
                st.rerun()
            render_folder_picker(node["children"][folder_name], path=full_path)


st.title("📄 RAG PDF Local")
st.caption(f"LLM: {settings.lm_studio_model} via {settings.lm_studio_base_url}")

try:
    documents = get_store().list_documents()
except Exception:  # noqa: BLE001
    documents = []
all_folders = get_all_folders()
tree = build_tree(documents, all_folders)

with st.expander("🗂️ Pastas e arquivos (seleção + gerenciamento)", expanded=False):
    st.caption(
        "Marque pastas/arquivos pra restringir a busca (sem marcação = busca em tudo). "
        "Use os ícones para mover (🔀), renomear pasta (✏️), criar subpasta (➕) e apagar (🗑️)."
    )
    new_root_folder = st.text_input("Criar nova pasta na raiz", key="new_root_folder_input")
    if st.button("💾 Salvar pasta", key="save_root_folder") and new_root_folder.strip():
        folder_registry.create_folder(new_root_folder.strip())
        st.rerun()
    st.divider()
    if not documents and not all_folders:
        st.caption("Nenhum documento ou pasta ainda.")
    else:
        render_unified_tree(tree, all_folders)

if "upload_target_folder" not in st.session_state:
    st.session_state.upload_target_folder = ""

with st.sidebar:
    st.header("Documento")
    st.caption(f"Pasta selecionada: **{st.session_state.upload_target_folder or ROOT_LABEL}**")
    with st.expander("📁 Escolher pasta de destino", expanded=False):
        if st.button(f"📌 Usar {ROOT_LABEL}", key="use_root_folder"):
            st.session_state.upload_target_folder = ""
            st.rerun()
        render_folder_picker(tree)
        st.divider()
        new_folder_name = st.text_input("Criar nova pasta (ex.: Suporte/2024)", key="new_folder_name_input")
        if st.button("💾 Salvar pasta", key="save_new_folder") and new_folder_name.strip():
            st.session_state.upload_target_folder = new_folder_name.strip()
            folder_registry.create_folder(new_folder_name.strip())
            st.rerun()

    folder_to_use = st.session_state.upload_target_folder

    uploaded = st.file_uploader("Importar PDF", type="pdf")
    if uploaded is not None and st.button("Indexar PDF", type="primary"):
        dest = Path("data/raw") / uploaded.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(uploaded.getvalue())
        with st.spinner(f"Indexando {uploaded.name}..."):
            try:
                n_chunks = run_ingest(dest, folder=folder_to_use)
                get_store.clear()
                destino = f" em '{folder_to_use}'" if folder_to_use else ""
                st.success(f"{n_chunks} chunks indexados de {uploaded.name}{destino}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Falha ao indexar: {exc}")

    st.divider()
    try:
        st.metric("Chunks no vector store", get_store().count())
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Vector store indisponível: {exc}")

    st.divider()
    st.header("Modo de resposta")
    mode = st.radio("Como responder às perguntas do chat", [TOPK_MODE, ADAPTIVE_MODE, MAPREDUCE_MODE])

    top_k = settings.top_k
    profile = RETRIEVAL_PROFILES["padrao"]
    summarize_source = None
    selected_sources: list[str] = []

    if mode == TOPK_MODE:
        st.caption("Busca sempre os k trechos mais parecidos com a pergunta — rápido, bom para fatos pontuais.")
        top_k = st.number_input("Quantos trechos buscar (k)", min_value=1, max_value=50, value=settings.top_k)
    elif mode == ADAPTIVE_MODE:
        st.caption(
            "Pega o trecho mais parecido com a pergunta e vai incluindo os próximos, "
            "parando assim que a margem de similaridade ou o teto de contexto for atingido."
        )
        profile_keys = [*RETRIEVAL_PROFILES.keys(), "personalizado"]
        profile_labels = {**{k: p.label for k, p in RETRIEVAL_PROFILES.items()}, "personalizado": "Personalizado"}
        profile_key = st.radio(
            "Perfil de busca",
            profile_keys,
            format_func=lambda k: profile_labels[k],
            index=1,
            horizontal=True,
        )
        if profile_key == "personalizado":
            margin_pct = st.slider("Margem de similaridade (%)", min_value=0, max_value=100, value=5, step=1)
            max_context_chars = st.slider(
                "Teto de contexto (caracteres)", min_value=500, max_value=20_000, value=6000, step=500
            )
            profile = RetrievalProfile("Personalizado", margin_pct=margin_pct, max_context_chars=max_context_chars)
        else:
            profile = RETRIEVAL_PROFILES[profile_key]
            st.caption(
                f"Margem: {profile.margin_pct:.0f}% — "
                f"teto de contexto: {profile.max_context_chars:,} caracteres".replace(",", ".")
            )
    else:
        st.caption(
            "Lê o documento inteiro em blocos e resume — mais lento, bom para "
            "'quais os principais pontos?'. O texto da pergunta é ignorado nesse modo."
        )
        if documents:
            summarize_source = st.selectbox(
                "Documento a resumir",
                [d["source"] for d in documents],
                format_func=lambda s: next(
                    (f"{d['folder']}/{s}" if d["folder"] else s for d in documents if d["source"] == s), s
                ),
            )
        else:
            st.caption("Nenhum documento indexado ainda.")

    if mode in (TOPK_MODE, ADAPTIVE_MODE):
        selected_sources = selected_sources_from_tree()
        total_docs = len(documents)
        if selected_sources:
            st.caption(
                f"🗂️ Escopo: {len(selected_sources)} de {total_docs} documento(s) selecionado(s) "
                "(ajuste em 'Pastas e arquivos' acima do chat)."
            )
        else:
            st.caption(f"🗂️ Escopo: todos os {total_docs} documento(s) indexados.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}min {secs}s" if minutes else f"{secs}s"


def render_sources(sources: list) -> None:
    with st.expander("Fontes"):
        for s in sources:
            st.caption(f"{s.source} — página {s.page_number} (distância {s.distance:.3f})")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

placeholder = (
    "Escreva algo e envie para gerar o resumo do documento selecionado..."
    if mode == MAPREDUCE_MODE
    else "Pergunte sobre o documento indexado..."
)
question = st.chat_input(placeholder)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if mode in (TOPK_MODE, ADAPTIVE_MODE):
            with st.spinner("Consultando..."):
                try:
                    sources_filter = selected_sources or None
                    if mode == TOPK_MODE:
                        result = run_query(question, top_k=int(top_k), sources=sources_filter)
                    else:
                        result = run_query_adaptive(question, profile, sources=sources_filter)
                    st.markdown(result.answer)
                    if result.sources:
                        st.caption(f"{len(result.sources)} trecho(s) usado(s) como contexto")
                        render_sources(result.sources)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result.answer, "sources": result.sources}
                    )
                except Exception as exc:  # noqa: BLE001
                    error_msg = (
                        f"Erro ao consultar o LM Studio: {exc}\n\n"
                        "Verifique se o servidor local está ativo (Developer → Start Server)."
                    )
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            if not summarize_source:
                st.warning("Nenhum documento indexado para resumir.")
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def handle_progress(p: SummarizeProgress) -> None:
                    fraction = min(p.step / p.total, 1.0)
                    progress_bar.progress(fraction)
                    remaining = max(p.total - p.step, 0) * p.avg_seconds_per_call
                    stage_label = (
                        "Lendo blocos do documento"
                        if p.stage == "map"
                        else f"Consolidando resumos (nível {p.level})"
                    )
                    status_text.caption(
                        f"{stage_label}: {p.step}/{p.total} — "
                        f"~{p.avg_seconds_per_call:.0f}s/chamada — "
                        f"tempo restante estimado nesta etapa: {format_duration(remaining)}"
                    )

                try:
                    summary = run_summarize(summarize_source, on_progress=handle_progress)
                    progress_bar.empty()
                    status_text.empty()
                    content = f"**Resumo de {summarize_source}:**\n\n{summary}"
                    st.markdown(content)
                    st.session_state.messages.append({"role": "assistant", "content": content})
                except Exception as exc:  # noqa: BLE001
                    progress_bar.empty()
                    status_text.empty()
                    error_msg = f"Falha ao gerar resumo: {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
