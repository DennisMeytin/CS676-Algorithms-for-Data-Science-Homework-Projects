"""Tests for the tool-use router. No API key: the model is a ScriptedToolLLM."""

from personaforge.llm import ScriptedToolLLM, text_response, tool_response
from personaforge.orchestrator import MAX_HISTORY, MAX_TOOL_STEPS, TOOLS, Orchestrator


def make(tmp_path, *script):
    llm = ScriptedToolLLM(script=list(script))
    return Orchestrator(llm=llm, persona_dir=tmp_path), llm


def test_tools_are_declared_with_schemas():
    names = {t["name"] for t in TOOLS}
    assert names == {"create_persona", "list_personas", "edit_persona", "undo_edit", "start_conversation"}
    assert all("input_schema" in t for t in TOOLS)


def test_model_choice_creates_persona_and_replies(tmp_path):
    orch, llm = make(
        tmp_path,
        tool_response("create_persona", {"description": "a busy doctor"}),
        text_response("Made your doctor."),
    )
    result = orch.handle("I need someone to play a doctor")

    assert result.reply == "Made your doctor."
    assert len(list(tmp_path.glob("*.md"))) == 1
    assert any(e.startswith("wrote") for e in result.events)


def test_tool_result_is_sent_back_to_the_model(tmp_path):
    orch, llm = make(
        tmp_path,
        tool_response("list_personas", {}, call_id="toolu_1"),
        text_response("None yet."),
    )
    orch.handle("who do I have?")

    second_call_messages = llm.tool_calls_seen[1]["messages"]
    last = second_call_messages[-1]
    assert last["role"] == "user"
    assert last["content"][0]["type"] == "tool_result"
    assert last["content"][0]["tool_use_id"] == "toolu_1"


def test_clarifying_question_needs_no_tool(tmp_path):
    orch, _ = make(tmp_path, text_response("What kind of persona?"))
    result = orch.handle("create a persona")

    assert result.reply == "What kind of persona?"
    assert list(tmp_path.glob("*.md")) == []


def test_chained_tools_create_two_then_converse(tmp_path):
    orch, _ = make(
        tmp_path,
        tool_response("create_persona", {"description": "doctor"}, call_id="a"),
        tool_response("create_persona", {"description": "patient"}, call_id="b"),
        tool_response(
            "start_conversation",
            {"participants": ["doctor", "patient"], "topic": "the results", "turns": 4},
            call_id="c",
        ),
        text_response("They talked."),
    )
    turns = []
    result = orch.handle("two people discuss results", on_turn=turns.append)

    assert result.conversation is not None
    assert len(result.conversation.turns) == 4
    assert len(turns) == 4  # the UI callback still fires per turn
    assert result.reply == "They talked."


def test_conversation_with_unknown_persona_reports_error_to_model(tmp_path):
    orch, llm = make(
        tmp_path,
        tool_response("start_conversation", {"participants": ["ghost", "nobody"], "topic": "x"}),
        text_response("Those personas don't exist."),
    )
    result = orch.handle("make them talk")

    tool_result = llm.tool_calls_seen[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "couldn't find" in tool_result["content"]
    assert result.conversation is None


def test_conversation_needs_two_participants(tmp_path):
    orch, llm = make(
        tmp_path,
        tool_response("start_conversation", {"participants": ["doctor"], "topic": "x"}),
        text_response("Who else?"),
    )
    orch.handle("have the doctor talk")

    tool_result = llm.tool_calls_seen[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True


def test_turns_are_clamped(tmp_path):
    orch, _ = make(
        tmp_path,
        tool_response("create_persona", {"description": "doctor"}, call_id="a"),
        tool_response("create_persona", {"description": "patient"}, call_id="b"),
        tool_response(
            "start_conversation",
            {"participants": ["doctor", "patient"], "topic": "x", "turns": 500},
            call_id="c",
        ),
        text_response("done"),
    )
    result = orch.handle("go")
    assert len(result.conversation.turns) == 20


def test_unknown_tool_is_reported_not_raised(tmp_path):
    orch, llm = make(tmp_path, tool_response("delete_everything", {}), text_response("ok"))
    result = orch.handle("hi")

    assert llm.tool_calls_seen[1]["messages"][-1]["content"][0]["is_error"] is True
    assert result.reply == "ok"


def test_runaway_tool_loop_is_stopped(tmp_path):
    orch, llm = make(tmp_path, *[tool_response("list_personas", {}, call_id=f"t{i}") for i in range(20)])
    result = orch.handle("loop forever")

    assert len(llm.tool_calls_seen) == MAX_TOOL_STEPS
    assert "too many steps" in result.reply


def test_memory_persists_across_messages(tmp_path):
    orch, llm = make(tmp_path, text_response("one"), text_response("two"))
    orch.handle("first")
    orch.handle("second")

    roles = [m["role"] for m in llm.tool_calls_seen[1]["messages"]]
    assert roles == ["user", "assistant", "user"]


def test_history_trim_never_orphans_a_tool_result(tmp_path):
    orch, _ = make(tmp_path)
    for i in range(MAX_HISTORY):
        orch.history.append({"role": "user", "content": f"q{i}"})
        orch.history.append({"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "n", "input": {}}]})
        orch.history.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "r"}]})
    orch._trim_history()

    assert len(orch.history) <= MAX_HISTORY
    first = orch.history[0]
    assert first["role"] == "user" and isinstance(first["content"], str)


def test_offline_llm_still_uses_regex_router(tmp_path):
    from personaforge.llm import DemoLLM

    orch = Orchestrator(llm=DemoLLM(), persona_dir=tmp_path)
    result = orch.handle("create a persona doctor")
    assert "Created" in result.reply
