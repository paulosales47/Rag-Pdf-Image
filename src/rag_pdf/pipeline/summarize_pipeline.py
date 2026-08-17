from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from loguru import logger

from rag_pdf.config import settings
from rag_pdf.llm.lmstudio_client import LMStudioClient
from rag_pdf.prompts.summary_prompt import (
    MAP_SYSTEM_PROMPT,
    REDUCE_SYSTEM_PROMPT,
    build_map_prompt,
    build_reduce_prompt,
)
from rag_pdf.vectorstore.chroma_store import ChromaVectorStore

_MAX_REDUCE_LEVELS = 4


@dataclass
class SummarizeProgress:
    stage: str  # "map" ou "reduce"
    level: int  # 0 no map, 1+ a cada nível de redução
    step: int  # chamada atual (1-indexada) dentro da etapa
    total: int  # total de chamadas nesta etapa
    elapsed: float  # segundos decorridos nesta etapa
    avg_seconds_per_call: float  # média de segundos por chamada nesta etapa


ProgressCallback = Callable[[SummarizeProgress], None]


def list_sources() -> list[str]:
    return ChromaVectorStore().list_sources()


def _batch_by_chars(texts: list[str], max_chars: int) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for text in texts:
        if current and current_len + len(text) > max_chars:
            batches.append(current)
            current, current_len = [], 0
        current.append(text)
        current_len += len(text)
    if current:
        batches.append(current)
    return batches


def run_summarize(source: str, on_progress: ProgressCallback | None = None) -> str:
    store = ChromaVectorStore()
    chunks = store.get_by_source(source)
    if not chunks:
        raise ValueError(f"Nenhum chunk indexado para '{source}'. Rode o ingest primeiro.")

    llm = LMStudioClient()

    def run_stage(stage: str, level: int, prompts: list[tuple[str, str]]) -> list[str]:
        results: list[str] = []
        start = monotonic()
        for i, (system_prompt, user_prompt) in enumerate(prompts, start=1):
            results.append(llm.generate(system_prompt, user_prompt))
            elapsed = monotonic() - start
            if on_progress:
                on_progress(
                    SummarizeProgress(
                        stage=stage,
                        level=level,
                        step=i,
                        total=len(prompts),
                        elapsed=elapsed,
                        avg_seconds_per_call=elapsed / i,
                    )
                )
        return results

    batches = _batch_by_chars([c["text"] for c in chunks], settings.summarize_batch_chars)
    logger.info(f"Resumindo '{source}': {len(chunks)} chunks em {len(batches)} bloco(s) (map)")
    map_prompts = [
        (MAP_SYSTEM_PROMPT, build_map_prompt(source, "\n\n".join(batch))) for batch in batches
    ]
    summaries = run_stage("map", 0, map_prompts)

    level = 0
    while len(summaries) > 1 and level < _MAX_REDUCE_LEVELS:
        groups = _batch_by_chars(summaries, settings.summarize_batch_chars)
        if len(groups) == len(summaries):
            break  # não está mais reduzindo, evita loop sem fim
        level += 1
        logger.info(
            f"Resumindo '{source}': reduzindo {len(summaries)} resumos em {len(groups)} (nível {level})"
        )
        reduce_prompts = [
            (REDUCE_SYSTEM_PROMPT, build_reduce_prompt(source, "\n\n".join(group)))
            for group in groups
        ]
        summaries = run_stage("reduce", level, reduce_prompts)

    if len(summaries) > 1:
        final_prompt = [
            (REDUCE_SYSTEM_PROMPT, build_reduce_prompt(source, "\n\n".join(summaries)))
        ]
        summaries = run_stage("reduce", level + 1, final_prompt)

    return summaries[0]
