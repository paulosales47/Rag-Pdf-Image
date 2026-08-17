import re

from openai import OpenAI

from rag_pdf.config import settings

# Alguns modelos "thinking" vazam o raciocínio interno dentro do próprio
# `content` (em vez de `reasoning_content`) quando a extração de raciocínio
# está desativada no LM Studio, separado da resposta final por um marcador
# de canal (ex.: "<channel|>"). Descartamos tudo antes do último marcador.
_CHANNEL_MARKER_RE = re.compile(r"<\|?channel\|?>", re.IGNORECASE)


class LMStudioClient:
    """Cliente para o servidor local do LM Studio (API compatível com OpenAI).

    Requer que o LM Studio esteja aberto, com um modelo carregado e o
    servidor local ativo (Developer -> Start Server), expondo por padrão
    http://localhost:1234/v1.
    """

    def __init__(self) -> None:
        self._client = OpenAI(
            base_url=settings.lm_studio_base_url,
            api_key=settings.lm_studio_api_key,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=settings.lm_studio_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        if not content.strip() and choice.finish_reason == "length":
            raise RuntimeError(
                "O modelo esgotou o limite de tokens raciocinando (finish_reason=length) "
                "antes de gerar a resposta. Aumente LLM_MAX_TOKENS no .env."
            )

        parts = _CHANNEL_MARKER_RE.split(content)
        if len(parts) > 1:
            content = parts[-1]

        return content.strip()
