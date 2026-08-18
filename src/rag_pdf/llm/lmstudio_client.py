from openai import OpenAI

from rag_pdf.config import settings
from rag_pdf.llm.response_utils import raise_if_truncated_by_reasoning, strip_leaked_reasoning


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
        raise_if_truncated_by_reasoning(content, choice.finish_reason, "LLM_MAX_TOKENS")
        return strip_leaked_reasoning(content).strip()
