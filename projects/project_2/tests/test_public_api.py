"""The package's public surface: what `import personaforge` promises."""

import importlib.metadata

import personaforge
from personaforge import (
    AmbiguousPersona,
    Orchestrator,
    PersonaEditError,
    default_llm,
)


def test_every_name_in_all_is_importable():
    missing = [name for name in personaforge.__all__ if not hasattr(personaforge, name)]
    assert missing == []


def test_all_has_no_duplicates():
    assert len(personaforge.__all__) == len(set(personaforge.__all__))


def test_features_added_in_this_submission_are_public():
    expected = {
        "edit_persona",
        "undo_edit",
        "EditResult",
        "AmbiguousPersona",
        "PersonaEditError",
        "friendly_error",
        "Usage",
        "ToolCall",
        "ToolResponse",
        "ScriptedToolLLM",
        "DemoLLM",
    }
    assert expected <= set(personaforge.__all__)


def test_version_matches_the_installed_package():
    """Keeps __version__ and pyproject.toml from drifting apart."""
    assert personaforge.__version__ == importlib.metadata.version("personaforge")


def test_documented_errors_are_value_errors():
    """Callers can catch ValueError without importing our types."""
    assert issubclass(AmbiguousPersona, ValueError)
    assert issubclass(PersonaEditError, ValueError)


def test_the_docstring_example_works_offline(tmp_path):
    orchestrator = Orchestrator(llm=default_llm(offline=True), persona_dir=tmp_path)

    created = orchestrator.handle("create a persona doctor")
    edited = orchestrator.handle("edit the doctor")

    assert "Created" in created.reply
    assert "live model" in edited.reply
