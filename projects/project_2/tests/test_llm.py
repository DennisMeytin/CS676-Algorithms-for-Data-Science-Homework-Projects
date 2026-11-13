"""
Tests for the LLM stubs.

`DemoLLM` powers `--offline`, which is the first thing anyone runs. If it
regresses, a new student's first impression is a broken app — so it gets tests.
"""

import pytest

from personaforge import PersonaSpec, generate_persona
from personaforge.llm import DemoLLM, ScriptedLLM, default_llm
from personaforge.orchestrator import _STAGE_MANAGER_SYSTEM


def test_scripted_llm_cycles_replies():
    llm = ScriptedLLM(replies=["one", "two"])
    got = [llm.complete("s", [{"role": "user", "content": "x"}]) for _ in range(4)]
    assert got == ["one", "two", "one", "two"]


def test_scripted_llm_records_calls():
    llm = ScriptedLLM()
    llm.complete("SYSTEM_MARKER", [{"role": "user", "content": "hello"}])

    assert len(llm.calls) == 1
    assert llm.calls[0]["system"] == "SYSTEM_MARKER"


def test_demo_llm_returns_persona_markdown_for_the_author_prompt():
    """Offline persona creation must yield a real name and role, not 'Unnamed'."""
    llm = DemoLLM()
    spec = generate_persona("doctor who is direct", llm)

    assert spec.name != "Unnamed"
    assert spec.role == "doctor"
    assert isinstance(spec, PersonaSpec)


def test_demo_llm_returns_dialogue_for_a_persona_prompt():
    llm = DemoLLM()
    reply = llm.complete("You are role-playing a character...", [{"role": "user", "content": "hi"}])

    assert "---" not in reply  # dialogue, not a persona document
    assert len(reply) > 0


def test_demo_llm_answers_the_stage_manager():
    llm = DemoLLM()
    reply = llm.complete(_STAGE_MANAGER_SYSTEM, [{"role": "user", "content": "what is this?"}])

    assert "offline" in reply.lower()


def test_default_llm_offline_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm = default_llm(offline=True)

    assert isinstance(llm, DemoLLM)
    assert llm.complete("anything", [{"role": "user", "content": "x"}])


def test_claude_llm_construction_does_not_need_a_key(monkeypatch):
    """Importing and constructing must be free; only calling requires a key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm = default_llm(offline=False)

    assert llm.model  # constructed fine


# --- usage tracking -----------------------------------------------------------


def test_usage_accumulates_and_prices_known_models():
    from types import SimpleNamespace

    from personaforge.llm import Usage

    usage = Usage()
    usage.add(SimpleNamespace(input_tokens=1_000_000, output_tokens=0))
    usage.add(SimpleNamespace(input_tokens=0, output_tokens=1_000_000))

    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (2, 1_000_000, 1_000_000)
    assert usage.cost("claude-haiku-5-5") == pytest.approx(0.10 + 0.50)
    assert usage.cost("some-future-model") is None
    assert "2 calls" in usage.summary("some-future-model")


def test_claude_llm_records_usage_from_each_response():
    from types import SimpleNamespace

    from personaforge.llm import ClaudeLLM

    block = SimpleNamespace(type="text", text="hi")
    response = SimpleNamespace(
        stop_reason="end_turn", content=[block], usage=SimpleNamespace(input_tokens=10, output_tokens=5)
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))
    llm = ClaudeLLM(_client=client)

    assert llm.complete("sys", [{"role": "user", "content": "x"}]) == "hi"
    assert (llm.usage.calls, llm.usage.input_tokens, llm.usage.output_tokens) == (1, 10, 5)
