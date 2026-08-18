import pytest

from rag_pdf.llm.response_utils import raise_if_truncated_by_reasoning, strip_leaked_reasoning


def test_strip_leaked_reasoning_keeps_content_after_last_marker():
    content = "blah blah pensando...<|channel|>Resposta final aqui."
    assert strip_leaked_reasoning(content) == "Resposta final aqui."


def test_strip_leaked_reasoning_returns_unchanged_when_no_marker():
    content = "Resposta direta, sem raciocínio vazado."
    assert strip_leaked_reasoning(content) == content


def test_raise_if_truncated_by_reasoning_raises_on_empty_content_and_length():
    with pytest.raises(RuntimeError, match="VISION_MAX_TOKENS"):
        raise_if_truncated_by_reasoning("", "length", "VISION_MAX_TOKENS")


def test_raise_if_truncated_by_reasoning_ignores_non_length_finish_reason():
    raise_if_truncated_by_reasoning("", "stop", "VISION_MAX_TOKENS")


def test_raise_if_truncated_by_reasoning_ignores_nonempty_content():
    raise_if_truncated_by_reasoning("resposta parcial", "length", "VISION_MAX_TOKENS")
