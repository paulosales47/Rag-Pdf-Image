from dataclasses import dataclass

from rag_pdf.llm.lmstudio_client import LMStudioClient
from rag_pdf.prompts.qa_prompt import SYSTEM_PROMPT, build_user_prompt
from rag_pdf.retrieval.retriever import RetrievedChunk, retrieve, retrieve_adaptive
from rag_pdf.vectorstore.chroma_store import ChromaVectorStore


@dataclass
class RetrievalProfile:
    label: str
    margin_pct: float
    max_context_chars: int


# Perfis de busca adaptativa: da mais restrita (poucos trechos, bem parecidos)
# até a mais abrangente (mais trechos, tolera trechos menos parecidos).
RETRIEVAL_PROFILES: dict[str, RetrievalProfile] = {
    "restrita": RetrievalProfile("Restrita", margin_pct=30, max_context_chars=3000),
    "padrao": RetrievalProfile("Padrão", margin_pct=45, max_context_chars=6000),
    "abrangente": RetrievalProfile("Abrangente", margin_pct=60, max_context_chars=14000),
}


@dataclass
class QAResult:
    answer: str
    sources: list[RetrievedChunk]


def _answer(question: str, chunks: list[RetrievedChunk]) -> QAResult:
    if not chunks:
        return QAResult(
            answer="Nenhum documento indexado ainda. Rode `rag-pdf ingest --pdf <arquivo>` primeiro.",
            sources=[],
        )

    user_prompt = build_user_prompt(question, chunks)
    llm = LMStudioClient()
    answer = llm.generate(SYSTEM_PROMPT, user_prompt)
    return QAResult(answer=answer, sources=chunks)


def run_query(
    question: str, top_k: int | None = None, sources: list[str] | None = None
) -> QAResult:
    store = ChromaVectorStore()
    chunks = retrieve(question, store, top_k=top_k, sources=sources)
    return _answer(question, chunks)


def run_query_adaptive(
    question: str, profile: RetrievalProfile, sources: list[str] | None = None
) -> QAResult:
    store = ChromaVectorStore()
    chunks = retrieve_adaptive(
        question,
        store,
        margin_pct=profile.margin_pct,
        max_context_chars=profile.max_context_chars,
        sources=sources,
    )
    return _answer(question, chunks)
