import base64
from pathlib import Path

from openai import OpenAI

from rag_pdf.config import settings
from rag_pdf.llm.response_utils import raise_if_truncated_by_reasoning, strip_leaked_reasoning
from rag_pdf.prompts.vision_prompt import SYSTEM_PROMPT, USER_PROMPT

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class VisionClient:
    """Cliente para o modelo com suporte a visão carregado no LM Studio.

    Usa o mesmo servidor local do `LMStudioClient` (API compatível com OpenAI),
    mas com um modelo diferente, capaz de receber imagens via `image_url`.
    """

    def __init__(self) -> None:
        self._client = OpenAI(
            base_url=settings.lm_studio_base_url,
            api_key=settings.lm_studio_api_key,
        )

    def describe_image(self, image_path: Path) -> str:
        mime_type = _MIME_TYPES.get(image_path.suffix.lower(), "image/png")
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"

        response = self._client.chat.completions.create(
            model=settings.lm_studio_vision_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            max_tokens=settings.vision_max_tokens,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        raise_if_truncated_by_reasoning(content, choice.finish_reason, "VISION_MAX_TOKENS")
        return strip_leaked_reasoning(content).strip()


def describe_image(image_path: Path) -> str:
    return VisionClient().describe_image(image_path)
