"""
Tests for persona serialization and the on-disk format.

None of these touch the network. A persona is a file, so these are file tests.
"""

import pytest

from personaforge import PersonaSpec, find_persona, generate_persona, list_personas, load_persona, save_persona
from personaforge.llm import ScriptedLLM
from personaforge.persona import AmbiguousPersona

SAMPLE_MARKDOWN = """---
name: Maria Delgado
role: patient
summary: 58-year-old with poorly controlled type 2 diabetes
---

# Maria Delgado

## Background
Retired schoolteacher. Diagnosed eight years ago.

## How they speak
- Long sentences, lots of context before the point
"""


def test_slug_is_filesystem_safe():
    spec = PersonaSpec(name="Dr. Anne O'Hara-Smith", role="doctor", summary="x", body="y")
    assert spec.slug() == "dr-anne-o-hara-smith"


def test_slug_never_empty():
    spec = PersonaSpec(name="!!!", role="x", summary="", body="")
    assert spec.slug() == "unnamed"


def test_roundtrip_preserves_fields():
    spec = PersonaSpec(name="Maria Delgado", role="patient", summary="a summary", body="# Body\n\ntext")
    parsed = PersonaSpec.from_markdown(spec.to_markdown())
    assert parsed.name == spec.name
    assert parsed.role == spec.role
    assert parsed.summary == spec.summary
    assert parsed.body == spec.body


def test_parses_frontmatter():
    spec = PersonaSpec.from_markdown(SAMPLE_MARKDOWN)
    assert spec.name == "Maria Delgado"
    assert spec.role == "patient"
    assert "Retired schoolteacher" in spec.body
    # The frontmatter block itself must not leak into the body — it would end up
    # in the system prompt and confuse the model about who it is.
    assert "---" not in spec.body.split("\n")[0]


def test_missing_frontmatter_still_loads():
    spec = PersonaSpec.from_markdown("# Just a heading\n\nSome prose.")
    assert spec.name == "Unnamed"
    assert "Just a heading" in spec.body


# -----------------------------------------------------------------------------
# Regression tests: things the real model actually does
#
# The two cases below are not hypothetical edge cases someone imagined. They are
# shapes the model returned during live testing, each of which broke the parser
# until it was taught to cope. They are pinned here because the offline tests
# cannot see the real API — so without these, a refactor of from_markdown()
# would pass every test and still break the app the next time a real call came
# back slightly malformed.
#
# Hardening the parser further is YOUR TASK item 3 in persona.py. These tests
# only hold the ground already taken.
# -----------------------------------------------------------------------------

MISSING_CLOSING_DELIMITER = """---
name: Maria Delgado
role: patient
summary: 58-year-old with poorly controlled type 2 diabetes

# Maria Delgado

## Background
Retired schoolteacher.
"""


def test_missing_closing_delimiter_still_parses():
    """
    The model omits the closing '---' roughly one time in three.

    Before this was handled, the whole document was treated as frontmatter: the
    persona loaded with an empty body, so the agent got a system prompt
    describing nobody and answered as a generic assistant. Nothing raised.
    """
    spec = PersonaSpec.from_markdown(MISSING_CLOSING_DELIMITER)
    assert spec.name == "Maria Delgado"
    assert spec.role == "patient"
    assert "Retired schoolteacher" in spec.body
    # The header keys must not bleed into the body, or they end up in the
    # system prompt and the agent starts narrating its own metadata.
    assert "role: patient" not in spec.body


def test_code_fenced_document_still_parses():
    """The model sometimes wraps the whole persona in a markdown code fence."""
    spec = PersonaSpec.from_markdown("```markdown\n" + SAMPLE_MARKDOWN + "```")
    assert spec.name == "Maria Delgado"
    assert "Retired schoolteacher" in spec.body
    assert "```" not in spec.body


def test_save_and_load(tmp_path):
    spec = PersonaSpec(name="Maria Delgado", role="patient", summary="s", body="# M")
    path = save_persona(spec, tmp_path)

    assert path.exists()
    assert path.name == "maria-delgado.md"
    assert load_persona(path).name == "Maria Delgado"


def test_list_personas_empty_dir(tmp_path):
    assert list_personas(tmp_path / "does-not-exist") == []


def test_list_personas_sorted(tmp_path):
    for name in ["Zoe Chen", "Adam Blake"]:
        save_persona(PersonaSpec(name=name, role="x", summary="", body="b"), tmp_path)

    names = [s.name for s in list_personas(tmp_path)]
    assert names == ["Adam Blake", "Zoe Chen"]


def test_find_persona_by_name_slug_and_role(tmp_path):
    save_persona(PersonaSpec(name="Maria Delgado", role="patient", summary="", body="b"), tmp_path)

    assert find_persona("Maria Delgado", tmp_path).role == "patient"
    assert find_persona("maria-delgado", tmp_path).role == "patient"
    assert find_persona("patient", tmp_path).name == "Maria Delgado"
    assert find_persona("nobody at all", tmp_path) is None


def test_generate_persona_parses_model_output():
    llm = ScriptedLLM(replies=[SAMPLE_MARKDOWN])
    spec = generate_persona("a diabetic patient", llm)

    assert spec.name == "Maria Delgado"
    assert spec.role == "patient"
    assert len(llm.calls) == 1


def test_generate_persona_survives_unstructured_output():
    """The model ignoring the format must not produce a nameless persona."""
    llm = ScriptedLLM(replies=["Sure! Here is a persona for you."])
    spec = generate_persona("a nurse", llm)

    assert spec.name != "Unnamed"
    assert spec.summary == "a nurse"


# --- robustness fixes -------------------------------------------------------


def _save(tmp_path, name, role):
    return save_persona(PersonaSpec(name=name, role=role, summary="", body="b"), tmp_path)


def test_save_never_overwrites_existing_persona(tmp_path):
    first = _save(tmp_path, "Maria Delgado", "patient")
    second = _save(tmp_path, "Maria Delgado", "patient")

    assert first != second
    assert second.name == "maria-delgado-2.md"
    assert len(list(tmp_path.glob("*.md"))) == 2


def test_save_overwrite_is_opt_in(tmp_path):
    first = _save(tmp_path, "Maria Delgado", "patient")
    again = save_persona(PersonaSpec(name="Maria Delgado", role="nurse", summary="", body="b"), tmp_path, overwrite=True)

    assert first == again
    assert len(list(tmp_path.glob("*.md"))) == 1


def test_find_does_not_match_partial_words(tmp_path):
    _save(tmp_path, "Dorothy Hale", "teacher")
    assert find_persona("doc", tmp_path) is None


def test_find_matches_whole_word_in_name(tmp_path):
    _save(tmp_path, "Maria Delgado", "patient")
    assert find_persona("maria", tmp_path).name == "Maria Delgado"


def test_find_raises_when_role_is_shared(tmp_path):
    _save(tmp_path, "Maria Delgado", "patient")
    _save(tmp_path, "Sam Okafor", "patient")

    with pytest.raises(AmbiguousPersona) as err:
        find_persona("patient", tmp_path)
    assert len(err.value.candidates) == 2


def test_exact_name_beats_shared_role(tmp_path):
    _save(tmp_path, "Maria Delgado", "patient")
    _save(tmp_path, "Sam Okafor", "patient")
    assert find_persona("Sam Okafor", tmp_path).name == "Sam Okafor"


def test_generate_persona_normalizes_multiword_role():
    llm = ScriptedLLM(replies=["---\nname: Ana Ruiz\nrole: Primary Care Doctor\nsummary: s\n---\n\n# Ana Ruiz\n"])
    assert generate_persona("a doctor", llm).role == "doctor"


def test_parser_recovers_header_after_stray_fence():
    """Seen live with Haiku: an empty '---' header, then the real one in the body."""
    text = "---\n---\nname: Dolores Pruitt\nrole: patient\nsummary: Retired bookkeeper.\n---\n\n# Dolores Pruitt\n\n## Background\nShe gardens.\n"
    spec = PersonaSpec.from_markdown(text)

    assert spec.name == "Dolores Pruitt"
    assert spec.role == "patient"
    assert spec.summary == "Retired bookkeeper."
    assert spec.body.startswith("# Dolores Pruitt")
    assert "name:" not in spec.body


def test_parser_still_falls_back_on_plain_prose():
    spec = PersonaSpec.from_markdown("I'm sorry, I can't help with that.")
    assert spec.name == "Unnamed"


# --- transcripts live next to personas but are not personas ----------------------


def test_saved_transcript_is_not_listed_as_a_persona(tmp_path):
    save_persona(PersonaSpec(name="Ana Ruiz", role="doctor", summary="", body="b"), tmp_path)
    (tmp_path / "conversation.md").write_text("# Conversation: x\n\n**Ana:** hi", encoding="utf-8")

    assert [s.name for s in list_personas(tmp_path)] == ["Ana Ruiz"]
    assert find_persona("unnamed", tmp_path) is None


def test_a_persona_called_conversation_does_not_collide_with_the_transcript(tmp_path):
    path = save_persona(PersonaSpec(name="Conversation", role="host", summary="", body="b"), tmp_path)

    assert path.name == "conversation-persona.md"
    assert [s.name for s in list_personas(tmp_path)] == ["Conversation"]


def test_a_second_conversation_does_not_use_the_first_transcript_as_a_character(tmp_path):
    from personaforge.llm import DemoLLM
    from personaforge.orchestrator import Orchestrator

    # Names chosen to sort AFTER "conversation.md", as "doctor..." and "patient..." do:
    # the offline router takes the first two files, so order decides who is picked.
    for name, role in (("Dana Ruiz", "doctor"), ("Pat Chen", "patient")):
        save_persona(PersonaSpec(name=name, role=role, summary="", body="b"), tmp_path)
    orchestrator = Orchestrator(llm=DemoLLM(), persona_dir=tmp_path)

    orchestrator.handle("have them talk about the results")
    second = orchestrator.handle("have them talk about the plan")

    assert second.conversation.participants == ["Dana Ruiz", "Pat Chen"]
