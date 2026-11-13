"""Edge cases from the final audit: odd inputs, odd files, model quirks, untested branches."""

from types import SimpleNamespace

import pytest

from personaforge import PersonaSpec, Usage, cli, edit_persona, find_persona, list_personas, save_persona, ui
from personaforge.conversation import Turn
from personaforge.llm import ClaudeLLM, DemoLLM, ScriptedLLM, ScriptedToolLLM, text_response, tool_response
from personaforge.orchestrator import Orchestrator, Result
from personaforge.persona import PersonaEditError, _diff

BODY = "# X\n\n## Background\n" + "some background text " * 10


def squash(text):
    return " ".join(text.split())


def save(tmp_path, *people):
    for name, role in people:
        save_persona(PersonaSpec(name=name, role=role, summary="s", body=BODY), tmp_path)


def client_returning(content, stop_reason="end_turn"):
    response = SimpleNamespace(
        stop_reason=stop_reason, content=content, usage=SimpleNamespace(input_tokens=1, output_tokens=1)
    )
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))


def tool_result_of(llm, index=1):
    return llm.tool_calls_seen[index]["messages"][-1]["content"][0]


def run_tool(tmp_path, args, tool="start_conversation"):
    llm = ScriptedToolLLM(script=[tool_response(tool, args), text_response("done")], delegate=DemoLLM())
    result = Orchestrator(llm=llm, persona_dir=tmp_path).handle("go")
    return result, tool_result_of(llm)


# --- the model client -----------------------------------------------------------


def test_missing_api_key_fails_with_instructions(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ClaudeLLM().complete("sys", [{"role": "user", "content": "x"}])


def test_refusals_come_back_as_a_placeholder_not_an_error():
    llm = ClaudeLLM(_client=client_returning([], "refusal"))

    assert llm.complete("s", [{"role": "user", "content": "x"}]) == "[declined to respond]"
    assert llm.complete_with_tools("s", [{"role": "user", "content": "x"}], []).text == "[declined to respond]"


def test_blank_text_blocks_are_not_sent_back_to_the_model():
    blank = SimpleNamespace(type="text", text="   ")
    use = SimpleNamespace(type="tool_use", id="t1", name="list_personas", input={})
    reply = ClaudeLLM(_client=client_returning([blank, use], "tool_use")).complete_with_tools("s", [], [])

    assert [b["type"] for b in reply.content] == ["tool_use"]


def make_thinking_block():
    data = {"type": "thinking", "thinking": "", "signature": "sig123"}
    return SimpleNamespace(type="thinking", model_dump=lambda **kw: dict(data))


def test_thinking_blocks_are_kept_with_their_signature():
    use = SimpleNamespace(type="tool_use", id="t1", name="list_personas", input={})
    llm = ClaudeLLM(_client=client_returning([make_thinking_block(), use], "tool_use"))
    reply = llm.complete_with_tools("s", [], [])

    assert [b["type"] for b in reply.content] == ["thinking", "tool_use"]
    assert reply.content[0]["signature"] == "sig123"
    assert [c.name for c in reply.tool_calls] == ["list_personas"]
    assert reply.text == ""


def test_thinking_only_reply_that_ran_out_of_tokens_raises():
    llm = ClaudeLLM(_client=client_returning([make_thinking_block()], "max_tokens"))
    with pytest.raises(RuntimeError, match="output tokens"):
        llm.complete_with_tools("s", [], [])


def test_a_reply_with_no_text_and_no_tools_is_reported_not_blank(tmp_path):
    thinking_only = SimpleNamespace(content=[{"type": "thinking", "thinking": "", "signature": "s"}])
    thinking_only.text = ""
    thinking_only.tool_calls = []

    class OnlyThinking(ScriptedToolLLM):
        def complete_with_tools(self, *a, **k):
            return thinking_only

    result = Orchestrator(llm=OnlyThinking(), persona_dir=tmp_path).handle("hi")
    assert result.reply == "(no response)"


# --- conversation requests ------------------------------------------------------


@pytest.mark.parametrize("given", ["Ana Ruiz and Bo Chen", "Ana Ruiz, Bo Chen", "Ana Ruiz & Bo Chen"])
def test_participants_sent_as_one_string_are_split(tmp_path, given):
    save(tmp_path, ("Ana Ruiz", "doctor"), ("Bo Chen", "patient"))
    result, _ = run_tool(tmp_path, {"participants": given, "topic": "x", "turns": 2})

    assert result.conversation.participants == ["Ana Ruiz", "Bo Chen"]


@pytest.mark.parametrize("names", [["Ana Ruiz", "Ana Ruiz"], ["doctor", "Ana Ruiz"]])
def test_a_persona_cannot_talk_to_itself(tmp_path, names):
    save(tmp_path, ("Ana Ruiz", "doctor"), ("Bo Chen", "patient"))
    result, tool_result = run_tool(tmp_path, {"participants": names, "topic": "x", "turns": 2})

    assert tool_result["is_error"] is True
    assert "talk to itself" in tool_result["content"]
    assert result.conversation is None
    assert not (tmp_path / "conversation.md").exists()


def test_offline_router_also_rejects_a_persona_talking_to_itself(tmp_path):
    save(tmp_path, ("Ana Ruiz", "doctor"), ("Bo Chen", "patient"))
    result = Orchestrator(llm=DemoLLM(), persona_dir=tmp_path).handle("have them talk between ana and ana about x")

    assert result.is_error and "talk to itself" in result.reply


def test_offline_router_reports_an_ambiguous_name(tmp_path):
    save(tmp_path, ("Ana Ruiz", "doctor"), ("Bo Chen", "patient"), ("Cy Diaz", "patient"))
    result = Orchestrator(llm=DemoLLM(), persona_dir=tmp_path).handle(
        "have them talk between patient and ana about x"
    )

    assert result.is_error and "matches several personas" in result.reply


def test_offline_router_asks_for_a_description(tmp_path):
    result = Orchestrator(llm=DemoLLM(), persona_dir=tmp_path).handle("create a persona")
    assert "what kind of persona" in result.reply


def test_the_start_callback_receives_the_topic(tmp_path):
    save(tmp_path, ("Dana Ruiz", "doctor"), ("Pat Chen", "patient"))
    seen = []
    Orchestrator(llm=DemoLLM(), persona_dir=tmp_path).handle("have them talk about the plan", on_start=seen.append)

    assert seen == ["the plan"]


# --- persona files --------------------------------------------------------------


def test_an_unreadable_file_in_the_folder_is_skipped(tmp_path):
    save(tmp_path, ("Ana Ruiz", "doctor"))
    (tmp_path / "broken.md").write_bytes(b"\xff\xfe\x00 not text \x80\x81")

    assert [s.name for s in list_personas(tmp_path)] == ["Ana Ruiz"]
    assert find_persona("ana", tmp_path).name == "Ana Ruiz"


def test_editing_still_works_next_to_an_unreadable_file(tmp_path):
    save(tmp_path, ("Ana Ruiz", "doctor"))
    (tmp_path / "broken.md").write_bytes(b"\xff\xfe\x00")
    spec = list_personas(tmp_path)[0]
    rewrite = f"---\nname: Ana Ruiz\nrole: doctor\nsummary: s\n---\n\n{BODY}\nExtra line."

    edit_persona(spec, "more detail", ScriptedLLM(replies=[rewrite]), tmp_path)

    assert "Extra line." in (tmp_path / "ana-ruiz.md").read_text(encoding="utf-8")


def test_names_that_try_to_escape_the_folder_stay_inside_it(tmp_path):
    path = save_persona(PersonaSpec(name="../../evil", role="x", summary="", body="b"), tmp_path)
    assert path.parent == tmp_path


def test_editing_a_persona_with_no_file_on_disk_fails_cleanly(tmp_path):
    ghost = PersonaSpec(name="Ghost", role="x", summary="", body=BODY)
    with pytest.raises(PersonaEditError, match="could not find the file"):
        edit_persona(ghost, "anything", ScriptedLLM(replies=["x"]), tmp_path)


def test_an_edit_that_is_too_short_is_rejected_and_changes_nothing(tmp_path):
    save(tmp_path, ("Ana Ruiz", "doctor"))
    spec = list_personas(tmp_path)[0]
    before = (tmp_path / "ana-ruiz.md").read_text(encoding="utf-8")
    short = "---\nname: Ana Ruiz\nrole: doctor\nsummary: s\n---\n\n## Background\nshort"

    with pytest.raises(PersonaEditError, match="too short"):
        edit_persona(spec, "x", ScriptedLLM(replies=[short]), tmp_path)

    assert (tmp_path / "ana-ruiz.md").read_text(encoding="utf-8") == before


def test_a_long_diff_is_cut_off_with_a_count():
    old = PersonaSpec(name="A", role="r", summary="s", body="\n".join(f"old line {i}" for i in range(20)))
    new = PersonaSpec(name="A", role="r", summary="s", body="\n".join(f"new line {i}" for i in range(20)))

    assert "more diff lines" in _diff(old, new, max_lines=5)


# --- what the user sees ---------------------------------------------------------


def test_help_mentions_editing_and_undo(capsys):
    ui.help_text()
    out = squash(capsys.readouterr().out)

    assert "undo that edit" in out and "live mode only" in out


class LiveFake:
    """Stands in for a live model: has a name and a usage tally, like ClaudeLLM."""

    model = "claude-haiku-5-5"

    def __init__(self):
        self.usage = Usage()
        self.usage.add(SimpleNamespace(input_tokens=1000, output_tokens=500))
        self.usage.add(SimpleNamespace(input_tokens=1000, output_tokens=500))


class SpeakingOrchestrator:
    def handle(self, text, on_turn=None, on_start=None):
        on_turn(Turn(speaker="Ana", text="hello there"))
        return Result(reply="all good")


def test_a_live_session_prints_its_usage_and_streams_dialogue(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setattr("sys.argv", ["personaforge"])
    monkeypatch.setattr(cli, "default_llm", lambda **kw: LiveFake())
    monkeypatch.setattr(cli, "Orchestrator", lambda **kw: SpeakingOrchestrator())
    lines = iter(["talk", "/quit"])
    monkeypatch.setattr(cli.ui, "user_prompt", lambda: next(lines))

    assert cli.main() == 0
    out = squash(capsys.readouterr().out)

    assert "Ana: hello there" in out
    assert "usage: 2 calls, 2000 in / 1000 out tokens" in out


# --- tool inputs the model might get wrong ---------------------------------------


def test_create_persona_without_a_description_is_an_error(tmp_path):
    _, tool_result = run_tool(tmp_path, {"description": "   "}, tool="create_persona")

    assert tool_result["is_error"] is True and "description is required" in tool_result["content"]
    assert list(tmp_path.glob("*.md")) == []


@pytest.mark.parametrize(
    "given, expected",
    [("abc", 6), (None, 6), ("3", 3), (1, 2), (2.9, 2), (999, 20)],
)
def test_turns_that_are_missing_or_odd_fall_back_or_are_clamped(tmp_path, given, expected):
    save(tmp_path, ("Dana Ruiz", "doctor"), ("Pat Chen", "patient"))
    args = {"participants": ["Dana Ruiz", "Pat Chen"], "topic": "x", "turns": given}
    result, _ = run_tool(tmp_path, args)

    assert len(result.conversation.turns) == expected


def test_an_empty_model_reply_is_reported(tmp_path):
    class Silent(ScriptedToolLLM):
        def complete_with_tools(self, *a, **k):
            return SimpleNamespace(content=[], text="", tool_calls=[])

    assert Orchestrator(llm=Silent(), persona_dir=tmp_path).handle("hi").reply == "(no response)"


# --- the REPL's smaller branches ---------------------------------------------------


class EchoOrchestrator:
    def handle(self, text, on_turn=None, on_start=None):
        return Result(reply=f"heard: {text}", events=["wrote somewhere"])


def run_repl(monkeypatch, lines):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["personaforge", "--offline"])
    monkeypatch.setattr(cli, "Orchestrator", lambda **kw: EchoOrchestrator())
    remaining = iter(lines)
    monkeypatch.setattr(cli.ui, "user_prompt", lambda: next(remaining))
    return cli.main()


def test_repl_ignores_blank_lines_and_runs_slash_commands(monkeypatch, capsys):
    code = run_repl(monkeypatch, ["", "/help", "/clear", "hello", "/q"])
    out = squash(capsys.readouterr().out)

    assert code == 0
    assert "Slash commands" in out  # /help printed
    assert "heard: hello" in out and "wrote somewhere" in out  # reply and its event line
    assert "heard: " not in out.replace("heard: hello", "")  # the blank line was not sent


# --- the last small gaps -----------------------------------------------------------


def test_text_blocks_in_a_tool_reply_are_kept():
    block = SimpleNamespace(type="text", text="hello there")
    reply = ClaudeLLM(_client=client_returning([block])).complete_with_tools("s", [], [])

    assert reply.text == "hello there"


def test_the_scripted_tool_model_says_when_its_script_has_run_out():
    reply = ScriptedToolLLM(script=[]).complete_with_tools("s", [], [])
    assert reply.text == "(script exhausted)"


def test_the_api_client_is_built_once_from_the_key(monkeypatch):
    import anthropic

    built = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setattr(anthropic, "Anthropic", lambda: built.append(1) or object())
    llm = ClaudeLLM()

    assert llm._get_client() is llm._get_client()
    assert built == [1]
