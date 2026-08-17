from rag_pdf.retrieval.retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "Você é um assistente que responde perguntas exclusivamente com base no contexto "
    "fornecido, extraído de um documento PDF. Se a resposta não estiver no contexto, "
    "diga claramente que não encontrou a informação no documento. Sempre que possível, "
    "cite a página de origem."
)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[Fonte: {c.source}, página {c.page_number}]\n{c.text}" for c in chunks
    )
    return f"Contexto:\n{context}\n\nPergunta: {question}\n\nResposta:"
