"""Behavior tests for grounded generation and the paired A/B setup."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.contracts import validate_generation_result


def chunk(index: int, method: str = "hybrid") -> dict:
    return {
        "id": f"policy::chunk-{index}",
        "content": f"Evidence from policy section {index}.",
        "score": 1.0 - index / 10,
        "metadata": {
            "source": "policy.md",
            "title": "Policy",
            "doc_type": "legal",
            "url": None,
            "chunk_index": index,
        },
        "retrieval_method": method,
    }


def test_generation_citations_match_reordered_sources(monkeypatch):
    import src.task10_generation as generation

    chunks = [chunk(index) for index in range(5)]
    monkeypatch.setattr(generation, "retrieve", lambda query, top_k: chunks)

    def fake_llm(system_prompt, user_message):
        assert "[1] Policy | Nguồn: policy.md | Chunk ID: policy::chunk-0" in user_message
        assert "[2] Policy | Nguồn: policy.md | Chunk ID: policy::chunk-1" in user_message
        assert user_message.index("policy::chunk-2") < user_message.index("policy::chunk-1")
        return "The policy says so [1]. Another detail is here [2]."

    monkeypatch.setattr(generation, "call_llm", fake_llm)
    response = generation.generate_with_citation("What does the policy say?")

    validate_generation_result(response)
    assert response["sources"][0]["id"] == "policy::chunk-0"
    assert response["sources"][1]["id"] == "policy::chunk-1"
    assert [item["id"] for item in chunks] == [f"policy::chunk-{i}" for i in range(5)]


def test_generation_refuses_missing_or_invalid_citations(monkeypatch):
    import src.task10_generation as generation

    for answer in ("Unsupported answer", "Unsupported [2]", generation.REFUSAL_TOKEN):
        monkeypatch.setattr(generation, "call_llm", lambda *args: answer)
        response = generation.generate_from_sources("question", [chunk(0)])
        validate_generation_result(response)
        assert response["answer"] == generation.SAFE_REFUSAL
        assert response["sources"] == []


def test_generation_provider_failure_is_safe_in_chat_and_visible_in_evaluation(monkeypatch):
    import src.task10_generation as generation

    def failed(*args):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(generation, "call_llm", failed)
    assert generation.generate_from_sources("question", [chunk(0)])["answer"] == generation.SAFE_REFUSAL
    try:
        generation.generate_from_sources("question", [chunk(0)], raise_on_error=True)
    except RuntimeError as exc:
        assert "provider unavailable" in str(exc)
    else:
        raise AssertionError("Evaluation must not silently score a provider outage")


@pytest.mark.parametrize("provider,key_name", [
    ("openai", "OPENAI_API_KEY"),
    ("gemini", "GEMINI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
])
def test_provider_dispatch_extracts_plain_text(monkeypatch, provider, key_name):
    import src.task10_generation as generation

    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv(key_name, "test-key")

    if provider == "openai":
        import openai

        def fake_openai(**kwargs):
            assert kwargs["api_key"] == "test-key"

            def create(**request):
                assert request["model"] == "test-model"
                assert request["messages"][0]["content"] == "system"
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OpenAI answer"))])

            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        monkeypatch.setattr(openai, "OpenAI", fake_openai)
        expected = "OpenAI answer"
    elif provider == "gemini":
        from google import genai

        def fake_gemini(**kwargs):
            assert kwargs["api_key"] == "test-key"

            def generate_content(**request):
                assert request["model"] == "test-model"
                assert request["config"].system_instruction == "system"
                return SimpleNamespace(text="Gemini answer")

            return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

        monkeypatch.setattr(genai, "Client", fake_gemini)
        expected = "Gemini answer"
    else:
        import anthropic

        def fake_anthropic(**kwargs):
            assert kwargs["api_key"] == "test-key"

            def create(**request):
                assert request["model"] == "test-model"
                assert request["system"] == "system"
                return SimpleNamespace(content=[SimpleNamespace(type="text", text="Anthropic answer")])

            return SimpleNamespace(messages=SimpleNamespace(create=create))

        monkeypatch.setattr(anthropic, "Anthropic", fake_anthropic)
        expected = "Anthropic answer"

    assert generation.call_llm("system", "question") == expected


def test_evaluation_changes_only_retrieval_strategy(monkeypatch):
    import group_project.evaluation.run_ab_evaluation as evaluation

    calls = []

    def fake_semantic(query, top_k):
        calls.append(("dense", query, top_k))
        return [chunk(0, "dense")]

    def fake_retrieve(query, top_k, score_threshold, use_reranking):
        calls.append(("hybrid", query, top_k, score_threshold, use_reranking))
        return [chunk(0, "hybrid")]

    async def fake_score(metrics, case, answer, contexts):
        return {name: 0.5 for name in evaluation.METRICS}

    monkeypatch.setattr(evaluation, "retrieve", fake_retrieve)
    monkeypatch.setattr(evaluation, "semantic_search", fake_semantic)
    monkeypatch.setattr(evaluation, "generate_from_sources", lambda query, chunks, **kwargs: {"answer": "Answer [1]"})
    monkeypatch.setattr(evaluation, "score_case", fake_score)

    rows = asyncio.run(evaluation.evaluate([{
        "id": "case_01", "question": "Question", "expected_answer": "Answer",
        "expected_context": "Evidence from policy section 0.",
        "source": "data/standardized/legal/policy.md",
    }], {}))
    assert [row["config"] for row in rows] == ["A", "B"]
    assert calls == [
        ("dense", "Question", 5),
        ("hybrid", "Question", 5, -1.0, True),
    ]
    assert rows[0]["source_ids"] == rows[1]["source_ids"]


def test_streamlit_replays_cited_sources():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(Path(__file__).parent.parent / "app.py").run(timeout=20)
    at.session_state["messages"] = [
        {"role": "user", "content": "Question"},
        {
            "id": 1,
            "role": "assistant",
            "content": "Answer [1]",
            "sources": [chunk(0)],
            "retrieval_source": "hybrid",
        },
    ]
    at.run(timeout=20)
    assert not at.exception
    assert any("Answer [1]" in item.value for item in at.markdown)
    assert any("[1] Policy" in item.value for item in at.markdown)
