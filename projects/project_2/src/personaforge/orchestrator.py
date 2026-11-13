"""
orchestrator.py — the agent you talk to.

You type "I want to create a persona patient" and something has to decide that
you meant `create_persona("patient")`. That decision is intent routing, and it
happens here.

There are two routers. With a model that supports tools, `_handle_with_tools`
hands Claude the tool definitions in `TOOLS`, runs whichever tool it calls, feeds
the result back, and repeats until the model answers in plain text. That is the
real path, and it can ask clarifying questions and chain actions.

The original regular-expression router is kept as the offline fallback. The
scripted models used by `--offline` and by most tests cannot call tools, so
`handle()` falls back to patterns for them. It is free and deterministic, but
pattern order decides meaning, so it only understands phrasings close to the
ones its author anticipated.

Anything the regex router does *not* recognize is passed to the model as ordinary
conversation, so the orchestrator still feels like something you are talking to
rather than a command prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .agent import agent_from_spec
from .conversation import Conversation, Turn, run_conversation, save_transcript
from .llm import LLM
from .persona import (
    DEFAULT_PERSONA_DIR,
    TRANSCRIPT_FILENAME,
    AmbiguousPersona,
    PersonaEditError,
    PersonaSpec,
    edit_persona,
    find_persona,
    generate_persona,
    list_personas,
    save_persona,
    undo_edit,
)

MIN_TURNS, MAX_TURNS, DEFAULT_TURNS = 2, 20, 6
MAX_TOOL_STEPS = 6      # tool round-trips per user message, so a confused model cannot loop forever
MAX_HISTORY = 40        # messages kept between user messages

_STAGE_MANAGER_SYSTEM = """You are the stage manager of a persona simulation workshop.

The user builds fictional characters and has them talk to each other. You help them
decide who to create and what those characters should discuss.

Be brief — two or three sentences. When the user seems ready to act, remind them they
can say things like "create a persona doctor", "list personas", or "have them talk
about the test results"."""


_TOOL_SYSTEM = """You are the stage manager of a persona simulation workshop. The user builds
fictional characters and has them talk to each other. You act through tools.

Rules:
- To make a character, call create_persona. To make several, call it once per character.
- If the user wants a persona but gives no description, or wants a conversation but
  gives no topic or does not say who should talk, ask one short clarifying question
  instead of guessing.
- Never invent persona names. If you are unsure who exists, call list_personas first.
- To change a persona the user already has, call edit_persona; to take back the last
  change, call undo_edit. Editing cannot rename a persona or change its role.
- To start a conversation, call start_conversation with the exact persona names.
- You may chain tools in one turn, e.g. list_personas and then start_conversation.
- After tools run, answer in two or three sentences. Do not repeat the dialogue; the
  user has already seen it."""

TOOLS = [
    {
        "name": "create_persona",
        "description": "Create and save one new persona character from a short description.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What kind of person to create, e.g. 'a skeptical doctor who is always busy'",
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "list_personas",
        "description": "List every saved persona with its name, role and summary.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "edit_persona",
        "description": (
            "Change an existing persona in place, e.g. 'make the doctor more skeptical'. "
            "Name and role cannot be changed. The previous version is kept so it can be undone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "persona": {"type": "string", "description": "Name or role of the persona to change"},
                "change": {"type": "string", "description": "What to change, in plain language"},
            },
            "required": ["persona", "change"],
        },
    },
    {
        "name": "undo_edit",
        "description": "Undo the most recent edit_persona on a persona. Can be repeated to step further back.",
        "input_schema": {
            "type": "object",
            "properties": {"persona": {"type": "string", "description": "Name or role of the persona"}},
            "required": ["persona"],
        },
    },
    {
        "name": "start_conversation",
        "description": "Have two or more saved personas talk to each other about a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names (or roles) of at least two existing personas",
                },
                "topic": {"type": "string", "description": "What they should talk about"},
                "turns": {
                    "type": "integer",
                    "description": f"Total lines of dialogue, {MIN_TURNS}-{MAX_TURNS}. Default {DEFAULT_TURNS}.",
                },
            },
            "required": ["participants", "topic"],
        },
    },
]


@dataclass
class Result:
    """What the orchestrator did, in a form the UI can render."""

    reply: str
    events: List[str] = field(default_factory=list)
    conversation: Optional[Conversation] = None
    is_error: bool = False


class Orchestrator:
    """Routes what the user typed to the right part of the package."""

    def __init__(self, llm: LLM, persona_dir: Path = DEFAULT_PERSONA_DIR) -> None:
        self.llm = llm
        self.persona_dir = Path(persona_dir)
        self.history: List[dict] = []

    # -- intents ------------------------------------------------------------

    def _create_persona(self, description: str) -> Result:
        """Author a persona, write it to disk, and report where it landed."""
        if not description.strip():
            return Result(reply="Tell me what kind of persona — for example, 'create a persona doctor'.")

        spec = generate_persona(description, self.llm)
        path = save_persona(spec, self.persona_dir)
        return Result(
            reply=f"Created **{spec.name}** ({spec.role}) — {spec.summary}",
            events=[f"wrote {path}"],
        )

    def _list_personas(self) -> Result:
        """Show what is currently in the persona folder."""
        specs = list_personas(self.persona_dir)
        if not specs:
            return Result(reply="No personas yet. Try: create a persona patient")

        lines = [f"- **{s.name}** ({s.role}) — {s.summary}" for s in specs]
        return Result(reply=f"{len(specs)} persona(s) in `{self.persona_dir}`:\n" + "\n".join(lines))

    def _resolve(self, name: str) -> PersonaSpec:
        """Look up one persona or raise a message the model can act on."""
        found = find_persona(name, self.persona_dir)  # may raise AmbiguousPersona
        if found is None:
            raise PersonaEditError(f"I couldn't find a persona matching '{name}'.")
        return found

    def _edit_persona(self, name: str, change: str) -> Result:
        """Rewrite a saved persona, keeping the old version for undo."""
        spec = self._resolve(name)
        edit = edit_persona(spec, change, self.llm, self.persona_dir)
        return Result(
            reply=f"Updated {edit.spec.name}. Changes:\n{edit.diff or '(no visible change)'}",
            events=[f"wrote {edit.path}", f"backup {edit.backup}"],
        )

    def _undo_edit(self, name: str) -> Result:
        """Restore the previous version of a persona."""
        spec = self._resolve(name)
        path = undo_edit(spec, self.persona_dir)
        return Result(reply=f"Restored the previous version of {spec.name}.", events=[f"wrote {path}"])

    def _converse(
        self,
        topic: str,
        who: Optional[List[str]] = None,
        turns: int = DEFAULT_TURNS,
        on_turn: Optional[Callable[[Turn], None]] = None,
        on_start: Optional[Callable[[str], None]] = None,
    ) -> Result:
        """Wake up two personas and let them talk."""
        specs: List[PersonaSpec] = []

        if who:
            for name in who:
                try:
                    found = find_persona(name, self.persona_dir)
                except AmbiguousPersona as exc:
                    return Result(reply=str(exc), is_error=True)
                if found is None:
                    return Result(reply=f"I couldn't find a persona matching '{name}'.")
                specs.append(found)
        else:
            specs = list_personas(self.persona_dir)[:2]

        if len({s.name for s in specs}) < len(specs):
            return Result(
                reply="A persona cannot talk to itself: two of those names point to the same persona.",
                is_error=True,
            )

        if len(specs) < 2:
            return Result(reply="I need at least two personas before they can talk. Create another one first.")

        agents = [agent_from_spec(s, self.llm) for s in specs]
        if on_start is not None:
            on_start(topic)
        conversation = run_conversation(agents, topic=topic, turns=turns, on_turn=on_turn)

        if conversation.ended_early and not conversation.turns:
            return Result(reply=f"The conversation could not start: {conversation.ended_early}", is_error=True)

        path = save_transcript(conversation, self.persona_dir, filename=TRANSCRIPT_FILENAME)
        names = " and ".join(s.name for s in specs)
        if conversation.ended_early:
            reply = f"Stopped after {len(conversation.turns)} of {turns} turns between {names}: {conversation.ended_early}"
        else:
            reply = f"{len(conversation.turns)} turns between {names}."

        return Result(reply=reply, events=[f"wrote {path}"], conversation=conversation)

    # -- tool use -----------------------------------------------------------

    def _trim_history(self) -> None:
        """
        Keep the last MAX_HISTORY messages, starting at a plain user message.

        Cutting anywhere else could separate a tool_use block from its
        tool_result, which the API rejects.
        """
        if len(self.history) <= MAX_HISTORY:
            return
        trimmed = self.history[-MAX_HISTORY:]
        while trimmed and not (trimmed[0]["role"] == "user" and isinstance(trimmed[0]["content"], str)):
            trimmed.pop(0)
        self.history = trimmed

    def _run_tool(self, call, on_turn, on_start) -> Result:
        """Execute one tool call. Never raises: errors go back to the model as text."""
        args = call.input or {}
        try:
            if call.name == "create_persona":
                description = str(args.get("description", "")).strip()
                if not description:
                    return Result(reply="description is required", is_error=True)
                return self._create_persona(description)

            if call.name == "list_personas":
                return self._list_personas()

            if call.name == "edit_persona":
                return self._edit_persona(str(args.get("persona", "")), str(args.get("change", "")))

            if call.name == "undo_edit":
                return self._undo_edit(str(args.get("persona", "")))

            if call.name == "start_conversation":
                # The schema asks for a list, but a model can send "Ana and Bo" as one
                # string, which would otherwise be iterated letter by letter.
                names = args.get("participants") or []
                if isinstance(names, str):
                    names = re.split(r"\s*(?:,|&|\band\b)\s*", names)
                who = [str(p).strip() for p in names if str(p).strip()]
                if len(who) < 2:
                    return Result(reply="participants must name at least two personas", is_error=True)
                topic = str(args.get("topic", "")).strip() or "an initial consultation"
                try:
                    turns = int(args.get("turns", DEFAULT_TURNS))
                except (TypeError, ValueError):
                    turns = DEFAULT_TURNS
                turns = max(MIN_TURNS, min(MAX_TURNS, turns))
                result = self._converse(topic, who=who, turns=turns, on_turn=on_turn, on_start=on_start)
                if result.conversation is None:
                    result.is_error = True
                else:
                    result.reply += "\n\n" + result.conversation.transcript()
                return result

            return Result(reply=f"unknown tool: {call.name}", is_error=True)
        except Exception as exc:  # noqa: BLE001 — report to the model, let it recover
            return Result(reply=f"{type(exc).__name__}: {exc}", is_error=True)

    def _handle_with_tools(self, text, on_turn, on_start) -> Result:
        """Let the model pick tools, run them, and feed results back until it answers."""
        self._trim_history()
        self.history.append({"role": "user", "content": text})

        events: List[str] = []
        conversation: Optional[Conversation] = None

        for _ in range(MAX_TOOL_STEPS):
            response = self.llm.complete_with_tools(_TOOL_SYSTEM, self.history, TOOLS, max_tokens=4096)
            if not response.content:
                return Result(reply="(no response)", events=events, conversation=conversation)

            self.history.append({"role": "assistant", "content": response.content})
            calls = response.tool_calls
            if not calls:
                return Result(reply=response.text or "(no response)", events=events, conversation=conversation)

            tool_results = []
            try:
                for call in calls:
                    outcome = self._run_tool(call, on_turn, on_start)
                    events.extend(outcome.events)
                    conversation = outcome.conversation or conversation
                    block = {"type": "tool_result", "tool_use_id": call.id, "content": outcome.reply}
                    if outcome.is_error:
                        block["is_error"] = True
                    tool_results.append(block)
            except KeyboardInterrupt:
                # Every tool_use needs a matching tool_result or the API rejects the
                # next request, so close out the unfinished calls before re-raising.
                answered = {block["tool_use_id"] for block in tool_results}
                tool_results += [
                    {"type": "tool_result", "tool_use_id": c.id, "content": "interrupted by the user", "is_error": True}
                    for c in calls
                    if c.id not in answered
                ]
                self.history.append({"role": "user", "content": tool_results})
                raise
            self.history.append({"role": "user", "content": tool_results})

        return Result(
            reply="I took too many steps on that and stopped. Could you say it more simply?",
            events=events,
            conversation=conversation,
        )

    def _chat(self, text: str) -> Result:
        """Nothing matched, so just talk to the user."""
        self._trim_history()
        self.history.append({"role": "user", "content": text})
        reply = self.llm.complete(_STAGE_MANAGER_SYSTEM, self.history, max_tokens=2048)
        self.history.append({"role": "assistant", "content": reply})
        return Result(reply=reply)

    # -- routing ------------------------------------------------------------

    def handle(
        self,
        text: str,
        on_turn: Optional[Callable[[Turn], None]] = None,
        on_start: Optional[Callable[[str], None]] = None,
    ) -> Result:
        """
        Work out what the user meant and do it.

        With a tool-capable model, the model decides (`_handle_with_tools`).
        Otherwise the patterns below are tried in order and the first match wins,
        so their ordering is load-bearing — which is why that path is only the
        offline fallback.
        """
        # A model that supports tools routes the request itself. Everything below
        # is the offline fallback (DemoLLM / ScriptedLLM have no tool support).
        if hasattr(self.llm, "complete_with_tools"):
            return self._handle_with_tools(text, on_turn, on_start)

        lowered = text.strip().lower()

        # Rewriting a persona needs a real model; the offline stub can only fake it.
        if re.match(r"(?:edit|undo)\b", lowered):
            return Result(
                reply="Editing and undo need a live model. Run without --offline to use them.",
                is_error=True,
            )

        # "create a persona doctor" / "make a persona who is a nurse"
        create = re.search(
            r"(?:create|make|add|build|generate)\s+(?:a\s+|an\s+|new\s+)*persona\s*(?:for|of|who is|that is|:)?\s*(.*)",
            lowered,
        )
        if create:
            return self._create_persona(create.group(1).strip())

        # "list personas" / "who do I have"
        if re.search(r"\b(list|show|what|who)\b.*\bpersona", lowered) or lowered in {"list", "ls"}:
            return self._list_personas()

        # "have them talk about X" / "start a conversation between A and B"
        converse = re.search(
            r"\b(?:talk|conversation|converse|chat|discuss|speak)\b",
            lowered,
        )
        if converse:
            topic_match = re.search(r"\babout\s+(.*)$", lowered)
            topic = topic_match.group(1).strip() if topic_match else "an initial consultation"

            who_match = re.search(r"\bbetween\s+(.*?)(?:\s+about\b|$)", lowered)
            who = None
            if who_match:
                who = [w.strip() for w in re.split(r"\s+and\s+|,", who_match.group(1)) if w.strip()]

            return self._converse(topic=topic, who=who, on_turn=on_turn, on_start=on_start)

        return self._chat(text)


# =============================================================================
# KNOWN LIMITS — status of the weaknesses the starter listed here
# =============================================================================
#
#  1. DONE. The regex router is replaced by real tool use (`TOOLS`,
#     `_handle_with_tools`). The loop runs at most MAX_TOOL_STEPS round-trips
#     per message so a confused model cannot spin forever.
#
#  2. DONE on the tool path: the model asks "what kind of persona?" instead of
#     guessing. The offline regex router still answers with a canned string.
#
#  3. DONE on the tool path: the model can chain "list the personas, then have
#     them talk". The offline regex router is still first-match-wins.
#
#  4. DONE on the tool path: `start_conversation` requires at least two named
#     participants and returns an error otherwise, so the model asks. The offline
#     path still takes the first two personas when none are named.
#
#  5. DONE on the tool path: the model sets `turns`, clamped to MIN_TURNS..
#     MAX_TURNS. The offline path is fixed at DEFAULT_TURNS.
#
#  6. PARTLY DONE. History is trimmed to MAX_HISTORY messages, and tool results
#     carry what was created, so the model remembers within a session. Nothing
#     is saved between sessions.
#
#  Not addressed: a conversation runs to completion once started and cannot be
#  interrupted.
