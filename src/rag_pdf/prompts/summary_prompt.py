MAP_SYSTEM_PROMPT = (
    "Você é um assistente que resume trechos de um documento em português, "
    "extraindo os pontos principais, ideias-chave e informações relevantes de "
    "forma objetiva, em tópicos. Não invente informações que não estejam no trecho."
)

REDUCE_SYSTEM_PROMPT = (
    "Você é um assistente que consolida vários resumos parciais de um mesmo "
    "documento em um único resumo final, coeso e organizado em tópicos, "
    "cobrindo os principais pontos do documento como um todo, em português."
)


def build_map_prompt(source: str, text: str) -> str:
    return f"Trecho do documento '{source}':\n\n{text}\n\nResuma os pontos principais deste trecho:"


def build_reduce_prompt(source: str, partial_summaries: str) -> str:
    return (
        f"Resumos parciais do documento '{source}', na ordem em que aparecem:\n\n"
        f"{partial_summaries}\n\n"
        "Consolide-os em um resumo final único com os principais pontos do documento:"
    )
