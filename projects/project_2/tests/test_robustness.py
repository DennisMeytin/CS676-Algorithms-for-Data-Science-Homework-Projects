"""Failure handling: API errors, truncated replies, Ctrl-C, partial conversations, startup."""

from types import SimpleNamespace

import pytest

from personaforge import PersonaSpec, cli, save_persona, ui
from personaforge.conversation import run_conversation
from personaforge.llm import ClaudeLLM, DemoLLM, ScriptedToolLLM, friendly_error, text_response, tool_response
from personaforge.orchestrator import Orchestrator, Result


class FakeAPIError(Exception):
    """Stands in for the SDK's status errors: all friendly_error needs is a status_code."""

    def __init__(self, status_code, message="boom"):
        super().__init__(message)
        self.status_code = status_code


class APIConnectionError(Exception):
    pass


class APITimeoutError(APIConnectionError):
    pass


def squash(text):
    """Collapse the whitespace rich adds when it wraps lines."""
    return " ".join(text.split())


# --- friendly_error ------------------------------------------------------------


@pytest.mark.parametrize(
    "status, expected",
    [
        (401, "rejected your key"),
        (403, "permission"),
        (404, "--model"),
        (429, "Rate limited"),
        (500, "having trouble"),
        (529, "having trouble"),
        (400, "rejected the request"),
    ],
)
def test_friendly_error_by_status(status, expected):
    assert expected in friendly_error(FakeAPIError(status))


def test_friendly_error_by_exception_name():
    assert "timed out" in friendly_error(APITimeoutError())
    assert "internet connection" in friendly_error(APIConnectionError())


def test_friendly_error_passes_our_own_messages_through():
    assert friendly_error(RuntimeError("ANTHROPIC_API_KEY is not set.")) == "ANTHROPIC_API_KEY is not set."
    assert friendly_error(ValueError("bad")) == "ValueError: bad"


# --- truncated replies ---------------------------------------------------------


def client_returning(content, stop_reason):
    response = SimpleNamespace(
        stop_reason=stop_reason, content=content, usage=SimpleNamespace(input_tokens=1, output_tokens=1)
    )
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))


def test_empty_reply_that_ran_out_of_tokens_raises():
    llm = ClaudeLLM(_client=client_returning([], "max_tokens"))
    with pytest.raises(RuntimeError, match="output tokens"):
        llm.complete("sys", [{"role": "user", "content": "x"}])


def test_partial_text_that_ran_out_of_tokens_is_kept():
    block = SimpleNamespace(type="text", text="a cut-off sent")
    llm = ClaudeLLM(_client=client_returning([block], "max_tokens"))
    assert llm.complete("sys", [{"role": "user", "content": "x"}]) == "a cut-off sent"


def test_tool_reply_that_ran_out_of_tokens_raises():
    llm = ClaudeLLM(_client=client_returning([], "max_tokens"))
    with pytest.raises(RuntimeError, match="output tokens"):
        llm.complete_with_tools("sys", [{"role": "user", "content": "x"}], [])


def test_tool_use_blocks_are_converted():
    block = SimpleNamespace(type="tool_use", id="t1", name="list_personas", input={})
    llm = ClaudeLLM(_client=client_returning([block], "tool_use"))
    reply = llm.complete_with_tools("sys", [{"role": "user", "content": "x"}], [])
    assert reply.tool_calls[0].name == "list_personas"


# --- conversations that fail partway -------------------------------------------


class FakeAgent:
    def __init__(self, name, fail_with=None, fail_on_call=1):
        self.name, self.fail_with, self.fail_on_call, self.calls = name, fail_with, fail_on_call, 0

    def respond(self, history):
        self.calls += 1
        if self.fail_with is not None and self.calls == self.fail_on_call:
            raise self.fail_with
        return f"{self.name} line {self.calls}"


def test_failure_midway_keeps_what_was_said():
    conversation = run_conversation([FakeAgent("A"), FakeAgent("B", FakeAPIError(429))], topic="t", turns=6)

    assert [t.speaker for t in conversation.turns] == ["A"]
    assert "Rate limited" in conversation.ended_early


def test_failure_on_first_turn_returns_an_empty_conversation():
    conversation = run_conversation([FakeAgent("A", FakeAPIError(401)), FakeAgent("B")], topic="t", turns=6)

    assert conversation.turns == []
    assert "rejected your key" in conversation.ended_early


def test_ctrl_c_stops_the_conversation_but_keeps_turns():
    conversation = run_conversation([FakeAgent("A"), FakeAgent("B", KeyboardInterrupt())], topic="t", turns=6)

    assert len(conversation.turns) == 1
    assert conversation.ended_early == "interrupted by the user"


def test_transcript_says_when_it_stopped_early():
    conversation = run_conversation([FakeAgent("A"), FakeAgent("B", FakeAPIError(429))], topic="t", turns=6)
    assert "Stopped early after 1 turn(s)" in conversation.transcript()


def test_finished_conversation_has_no_early_stop():
    conversation = run_conversation([FakeAgent("A"), FakeAgent("B")], topic="t", turns=4)
    assert conversation.ended_early is None
    assert "Stopped early" not in conversation.transcript()


# --- orchestrator under failure ------------------------------------------------


class FlakyModel(DemoLLM):
    """A scripted persona model that starts failing after `ok_calls` replies."""

    def __init__(self, ok_calls, error):
        super().__init__()
        self.ok_calls, self.error, self.n = ok_calls, error, 0

    def complete(self, system, messages, max_tokens=2048):
        self.n += 1
        if self.n > self.ok_calls:
            raise self.error
        return super().complete(system, messages, max_tokens)


def two_personas(tmp_path):
    for name, role in (("Ana Ruiz", "doctor"), ("Bo Chen", "patient")):
        save_persona(PersonaSpec(name=name, role=role, summary="", body="b"), tmp_path)


def converse_call():
    return tool_response("start_conversation", {"participants": ["Ana Ruiz", "Bo Chen"], "topic": "x", "turns": 6})


def test_partial_conversation_is_saved_and_reported(tmp_path):
    two_personas(tmp_path)
    llm = ScriptedToolLLM(script=[converse_call(), text_response("done")], delegate=FlakyModel(2, FakeAPIError(429)))
    result = Orchestrator(llm=llm, persona_dir=tmp_path).handle("talk")

    assert len(result.conversation.turns) == 2
    assert "Stopped early" in (tmp_path / "conversation.md").read_text(encoding="utf-8")
    tool_result = llm.tool_calls_seen[1]["messages"][-1]["content"][0]
    assert "Stopped after 2 of 6 turns" in tool_result["content"]
    assert "is_error" not in tool_result


def test_conversation_that_cannot_start_is_an_error_and_saves_nothing(tmp_path):
    two_personas(tmp_path)
    llm = ScriptedToolLLM(script=[converse_call(), text_response("sorry")], delegate=FlakyModel(0, FakeAPIError(401)))
    result = Orchestrator(llm=llm, persona_dir=tmp_path).handle("talk")

    tool_result = llm.tool_calls_seen[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "could not start" in tool_result["content"]
    assert result.conversation is None
    assert not (tmp_path / "conversation.md").exists()


def test_ctrl_c_during_a_tool_leaves_history_the_api_will_accept(tmp_path):
    llm = ScriptedToolLLM(
        script=[tool_response("create_persona", {"description": "x"}, call_id="a"), text_response("ok")],
        delegate=FlakyModel(0, KeyboardInterrupt()),
    )
    orch = Orchestrator(llm=llm, persona_dir=tmp_path)

    with pytest.raises(KeyboardInterrupt):
        orch.handle("make someone")

    last = orch.history[-1]
    assert last["role"] == "user"
    assert last["content"][0]["tool_use_id"] == "a" and last["content"][0]["is_error"] is True

    # The next request carries a tool_use followed by its tool_result, so it is valid.
    orch.handle("never mind")
    roles = [m["role"] for m in llm.tool_calls_seen[1]["messages"]]
    assert roles[:3] == ["user", "assistant", "user"]


# --- the CLI -------------------------------------------------------------------


def feed(monkeypatch, lines):
    remaining = iter(lines)

    def prompt():
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(cli.ui, "user_prompt", prompt)


@pytest.fixture
def cli_env(monkeypatch):
    """Keep the real .env and real argv out of the tests."""
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_live_mode_without_a_key_refuses_to_start(monkeypatch, capsys, cli_env):
    monkeypatch.setattr("sys.argv", ["personaforge"])
    assert cli.main() == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_offline_mode_needs_no_key(monkeypatch, cli_env):
    monkeypatch.setattr("sys.argv", ["personaforge", "--offline"])
    feed(monkeypatch, [])
    assert cli.main() == 0


class FakeOrchestrator:
    """Raises the given errors on successive calls, then answers."""

    def __init__(self, errors):
        self.errors = list(errors)

    def handle(self, text, on_turn=None, on_start=None):
        if self.errors:
            raise self.errors.pop(0)
        return Result(reply="all good")


def run_cli(monkeypatch, errors, lines):
    monkeypatch.setattr("sys.argv", ["personaforge", "--offline"])
    monkeypatch.setattr(cli, "Orchestrator", lambda **kw: FakeOrchestrator(errors))
    feed(monkeypatch, lines)
    return cli.main()


def test_ctrl_c_during_a_request_does_not_end_the_session(monkeypatch, capsys, cli_env):
    code = run_cli(monkeypatch, [KeyboardInterrupt()], ["first", "second", "/quit"])

    out = squash(capsys.readouterr().out)
    assert code == 0
    assert "interrupted" in out and "all good" in out


def test_api_errors_are_shown_in_plain_language_and_the_session_continues(monkeypatch, capsys, cli_env):
    code = run_cli(monkeypatch, [FakeAPIError(401)], ["first", "second", "/quit"])

    out = squash(capsys.readouterr().out)
    assert code == 0
    assert "rejected your key" in out and "all good" in out


def test_error_text_with_square_brackets_is_printed_literally(capsys):
    ui.error("model said [bold]no[/bold] [x]")
    assert "[x]" in squash(capsys.readouterr().out)
