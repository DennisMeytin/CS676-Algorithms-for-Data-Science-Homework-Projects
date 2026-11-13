"""Editing and undoing personas. No key: the 'model' replays a prepared rewrite."""

import pytest

from personaforge import PersonaSpec, save_persona
from personaforge.llm import ScriptedLLM, ScriptedToolLLM, text_response, tool_response
from personaforge.orchestrator import Orchestrator
from personaforge.persona import (
    HISTORY_DIRNAME,
    PersonaEditError,
    edit_persona,
    list_personas,
    load_persona,
    undo_edit,
)

BODY = (
    "# Ana Ruiz\n\n## Background\nA busy ER doctor with fifteen years on the floor.\n\n"
    "## Personality\n- Direct\n- Tired\n\n## How they speak\n- Short sentences\n"
)
EDITED_BODY = BODY.replace("- Direct", "- Direct\n- Deeply skeptical of every new treatment")


def doc(name="Ana Ruiz", role="doctor", summary="A busy ER doctor.", body=EDITED_BODY):
    return f"---\nname: {name}\nrole: {role}\nsummary: {summary}\n---\n\n{body}"


@pytest.fixture
def saved(tmp_path):
    spec = PersonaSpec(name="Ana Ruiz", role="doctor", summary="A busy ER doctor.", body=BODY)
    return spec, save_persona(spec, tmp_path), tmp_path


def llm_returning(text):
    return ScriptedLLM(replies=[text])


def test_edit_rewrites_file_and_keeps_backup(saved):
    spec, path, tmp = saved
    result = edit_persona(spec, "more skeptical", llm_returning(doc()), tmp)

    assert "skeptical" in path.read_text(encoding="utf-8")
    assert result.backup.exists() and "skeptical" not in result.backup.read_text(encoding="utf-8")
    assert result.backup.parent.name == HISTORY_DIRNAME
    assert "+- Deeply skeptical" in result.diff


def test_backups_are_not_listed_as_personas(saved):
    spec, _, tmp = saved
    edit_persona(spec, "more skeptical", llm_returning(doc()), tmp)
    assert len(list_personas(tmp)) == 1


def test_name_and_role_are_locked(saved):
    spec, path, tmp = saved
    edit_persona(spec, "x", llm_returning(doc(name="Someone Else", role="skeptic")), tmp)

    loaded = load_persona(path)
    assert (loaded.name, loaded.role) == ("Ana Ruiz", "doctor")


def test_prose_reply_changes_nothing(saved):
    spec, path, tmp = saved
    before = path.read_text(encoding="utf-8")
    with pytest.raises(PersonaEditError):
        edit_persona(spec, "x", llm_returning("Sorry, I cannot help with that."), tmp)

    assert path.read_text(encoding="utf-8") == before
    assert not (tmp / HISTORY_DIRNAME).exists()


def test_dropped_sections_are_rejected(saved):
    spec, path, tmp = saved
    before = path.read_text(encoding="utf-8")
    truncated = doc(body="# Ana Ruiz\n\n## Background\n" + "A long enough background. " * 6)
    with pytest.raises(PersonaEditError, match="dropped sections"):
        edit_persona(spec, "x", llm_returning(truncated), tmp)

    assert path.read_text(encoding="utf-8") == before


def test_empty_change_is_rejected(saved):
    spec, _, tmp = saved
    with pytest.raises(PersonaEditError):
        edit_persona(spec, "   ", llm_returning(doc()), tmp)


def test_undo_restores_previous_version(saved):
    spec, path, tmp = saved
    original = path.read_text(encoding="utf-8")
    edit_persona(spec, "more skeptical", llm_returning(doc()), tmp)

    undo_edit(spec, tmp)
    assert path.read_text(encoding="utf-8") == original
    assert list((tmp / HISTORY_DIRNAME).glob("*.md")) == []


def test_undo_steps_back_multiple_edits(saved):
    spec, path, tmp = saved
    first = doc(body=EDITED_BODY)
    second = doc(body=EDITED_BODY.replace("- Tired", "- Tired\n- Secretly kind"))
    edit_persona(spec, "one", llm_returning(first), tmp)
    edit_persona(spec, "two", llm_returning(second), tmp)

    undo_edit(spec, tmp)
    assert "Secretly kind" not in path.read_text(encoding="utf-8")
    assert "skeptical" in path.read_text(encoding="utf-8")
    undo_edit(spec, tmp)
    assert "skeptical" not in path.read_text(encoding="utf-8")


def test_undo_with_no_history_raises(saved):
    spec, _, tmp = saved
    with pytest.raises(PersonaEditError, match="nothing to undo"):
        undo_edit(spec, tmp)


def test_undo_does_not_touch_a_persona_with_a_prefix_name(tmp_path):
    a = PersonaSpec(name="Ana", role="nurse", summary="", body=BODY)
    b = PersonaSpec(name="Ana Ruiz", role="doctor", summary="", body=BODY)
    save_persona(a, tmp_path)
    save_persona(b, tmp_path)
    edit_persona(b, "x", llm_returning(doc()), tmp_path)

    with pytest.raises(PersonaEditError, match="nothing to undo"):
        undo_edit(a, tmp_path)


# --- through the orchestrator ------------------------------------------------


def make_orch(tmp, *script, rewrite=None):
    llm = ScriptedToolLLM(script=list(script), delegate=llm_returning(rewrite or doc()))
    return Orchestrator(llm=llm, persona_dir=tmp), llm


def test_model_can_edit_a_persona(saved):
    _, path, tmp = saved
    orch, _ = make_orch(
        tmp,
        tool_response("edit_persona", {"persona": "doctor", "change": "more skeptical"}),
        text_response("Done."),
    )
    result = orch.handle("make the doctor more skeptical")

    assert result.reply == "Done."
    assert "skeptical" in path.read_text(encoding="utf-8")
    assert any(e.startswith("backup") for e in result.events)


def test_edit_diff_is_sent_back_to_the_model(saved):
    _, _, tmp = saved
    orch, llm = make_orch(
        tmp,
        tool_response("edit_persona", {"persona": "Ana Ruiz", "change": "more skeptical"}),
        text_response("ok"),
    )
    orch.handle("edit")

    tool_result = llm.tool_calls_seen[1]["messages"][-1]["content"][0]
    assert "Deeply skeptical" in tool_result["content"]
    assert "is_error" not in tool_result


def test_failed_edit_reports_error_and_preserves_file(saved):
    _, path, tmp = saved
    before = path.read_text(encoding="utf-8")
    orch, llm = make_orch(
        tmp,
        tool_response("edit_persona", {"persona": "doctor", "change": "x"}),
        text_response("That edit failed."),
        rewrite="Sorry, I cannot do that.",
    )
    orch.handle("edit the doctor")

    assert llm.tool_calls_seen[1]["messages"][-1]["content"][0]["is_error"] is True
    assert path.read_text(encoding="utf-8") == before


def test_edit_unknown_persona_is_an_error(tmp_path):
    orch, llm = make_orch(
        tmp_path,
        tool_response("edit_persona", {"persona": "ghost", "change": "x"}),
        text_response("No such persona."),
    )
    orch.handle("edit ghost")
    assert llm.tool_calls_seen[1]["messages"][-1]["content"][0]["is_error"] is True


def test_edit_ambiguous_role_asks_instead_of_guessing(saved):
    spec, _, tmp = saved
    save_persona(PersonaSpec(name="Raj Patel", role="doctor", summary="", body=BODY), tmp)
    orch, llm = make_orch(
        tmp,
        tool_response("edit_persona", {"persona": "doctor", "change": "x"}),
        text_response("Which doctor?"),
    )
    orch.handle("edit the doctor")

    result_text = llm.tool_calls_seen[1]["messages"][-1]["content"][0]["content"]
    assert "matches several personas" in result_text


def test_model_can_undo(saved):
    _, path, tmp = saved
    original = path.read_text(encoding="utf-8")
    orch, _ = make_orch(
        tmp,
        tool_response("edit_persona", {"persona": "doctor", "change": "more skeptical"}, call_id="a"),
        text_response("Edited."),
        tool_response("undo_edit", {"persona": "doctor"}, call_id="b"),
        text_response("Undone."),
    )
    orch.handle("make the doctor more skeptical")
    orch.handle("actually undo that")

    assert path.read_text(encoding="utf-8") == original


def test_offline_mode_explains_editing_needs_a_live_model(tmp_path):
    from personaforge.llm import DemoLLM

    result = Orchestrator(llm=DemoLLM(), persona_dir=tmp_path).handle("edit the doctor")
    assert "live model" in result.reply
