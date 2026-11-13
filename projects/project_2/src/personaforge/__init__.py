"""
PersonaForge — an agent-to-agent persona simulation package.

The idea in four lines:

    spec  = generate_persona("a patient with chronic back pain", llm)
    path  = save_persona(spec)              # writes temp/<name>.md
    agent = agent_from_spec(spec, llm)      # markdown + model = a mind
    run_conversation([doctor, patient], topic="the test results")

A persona is a markdown file. An agent is that file plus a model. A conversation
is agents taking turns. Everything else is plumbing.

To drive it in plain language instead, hand a model to the orchestrator. With a
tool-capable model (`default_llm()`) Claude decides which action to take; with
`default_llm(offline=True)` a regex router stands in:

    orchestrator = Orchestrator(llm=default_llm(), persona_dir="temp")
    orchestrator.handle("create a persona: a direct, busy ER doctor")
    orchestrator.handle("make the doctor more skeptical")   # backed up, validated
    orchestrator.handle("undo that edit")

Failures you can expect to catch: `AmbiguousPersona` when a name matches several
personas, and `PersonaEditError` when an edit or undo cannot be applied (nothing
on disk has changed). `friendly_error` turns an API failure into one readable line.
"""

from .agent import PersonaAgent, agent_from_spec
from .conversation import Conversation, Turn, run_conversation, save_transcript
from .llm import (
    LLM,
    ClaudeLLM,
    DemoLLM,
    ScriptedLLM,
    ScriptedToolLLM,
    ToolCall,
    ToolResponse,
    Usage,
    default_llm,
    friendly_error,
)
from .orchestrator import Orchestrator, Result
from .persona import (
    AmbiguousPersona,
    EditResult,
    PersonaEditError,
    PersonaSpec,
    edit_persona,
    find_persona,
    generate_persona,
    list_personas,
    load_persona,
    save_persona,
    undo_edit,
)

__version__ = "0.2.0"

__all__ = [
    # agents and conversations
    "PersonaAgent",
    "agent_from_spec",
    "Conversation",
    "Turn",
    "run_conversation",
    "save_transcript",
    # models
    "LLM",
    "ClaudeLLM",
    "DemoLLM",
    "ScriptedLLM",
    "ScriptedToolLLM",
    "ToolCall",
    "ToolResponse",
    "Usage",
    "default_llm",
    "friendly_error",
    # the orchestrator
    "Orchestrator",
    "Result",
    # personas
    "PersonaSpec",
    "find_persona",
    "generate_persona",
    "list_personas",
    "load_persona",
    "save_persona",
    "edit_persona",
    "undo_edit",
    "EditResult",
    "AmbiguousPersona",
    "PersonaEditError",
]
