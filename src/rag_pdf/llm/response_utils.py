import re

# Alguns modelos "thinking" vazam o raciocínio interno dentro do próprio
# `content` (em vez de `reasoning_content`) quando a extração de raciocínio
# está desativada no LM Studio, separado da resposta final por um marcador
# de canal (ex.: "<channel|>"). Descartamos tudo antes do último marcador.
_CHANNEL_MARKER_RE = re.compile(r"<\|?channel\|?>", re.IGNORECASE)


def strip_leaked_reasoning(content: str) -> str:
    parts = _CHANNEL_MARKER_RE.split(content)
    return parts[-1] if len(parts) > 1 else content


def raise_if_truncated_by_reasoning(
    content: str, finish_reason: str | None, config_hint: str
) -> None:
    """Detecta quando um modelo "thinking" gastou todo o orçamento de tokens
    raciocinando e nunca chegou a escrever a resposta final (`content` vazio
    com `finish_reason=length`), em vez de devolver uma resposta em branco
    silenciosamente.
    """
    if not content.strip() and finish_reason == "length":
        raise RuntimeError(
            "O modelo esgotou o limite de tokens raciocinando (finish_reason=length) "
            f"antes de gerar a resposta. Aumente {config_hint} no .env."
        )
