"""
llm.py — the thin layer between a persona and a model.

Everything in this package that needs to "think" goes through the `LLM`
protocol below. The implementations:

    ClaudeLLM        real Claude calls. Needs ANTHROPIC_API_KEY. Also supports
                     tool use (`complete_with_tools`) and counts its own usage.
    ScriptedLLM      returns canned strings. No key, no network, no cost.
    DemoLLM          the offline stub behind `--offline`; answers in the shape
                     the caller expects (persona markdown, dialogue, chat).
    ScriptedToolLLM  replays a fixed sequence of tool calls, to test the tool loop.

That split is deliberate and it is the reason this project is testable. The
unit tests in `tests/` inject a ScriptedLLM, so they run in milliseconds for
free, and `main.py --offline` uses one too so you can see the whole app work
before you have configured anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

# Claude Opus 5 is the default. While you are iterating you may want
# "claude-haiku-5-5" — a persona conversation makes one call per turn, and those
# add up (about $0.01 for a full test session, against about $0.46 on Opus 5.5).
# Say which model produced your submitted transcripts.
DEFAULT_MODEL = "claude-opus-5"


class LLM(Protocol):
    """Anything that can turn a system prompt plus a transcript into a reply."""

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        ...


# =============================================================================
# ⚠️  NEEDS YOUR OWN API KEY
# =============================================================================
# REQUIRES A KEY. ClaudeLLM is every real model call in this package. Without
# ANTHROPIC_API_KEY it raises with instructions rather than failing quietly. Get
# a key at https://console.anthropic.com/ and put it in `.env` (copy
# `.env.example`). Calls are billed to you. A conversation costs one call per
# turn, plus one per persona created and one or more routing calls per message
# you type (the model decides which tool to use). The usage line printed when a
# session ends shows the real count.
#
# You do not need a key to work on this project: `uv run pytest` and
# `uv run main.py --offline` exercise the whole package for free.
#
# VERIFIED LIVE. `complete()` and `complete_with_tools()` have been run against
# the real API end to end on Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5: personas
# are authored, edited and saved, and hold character across a multi-turn
# conversation. Two bugs surfaced in live runs, both outside this file: the model
# sometimes writes a malformed frontmatter header (handled in persona.py), and
# piped output crashed on Windows (handled in cli.py).
# =============================================================================


# Anthropic's published rates in dollars per million tokens: (input, output).
# Only used to print an estimate; unknown models report tokens without a price.
PRICES_PER_MTOK = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


_OUT_OF_TOKENS = (
    "The model used all of its output tokens before replying (it may have been "
    "thinking). Try again, or raise max_tokens."
)


def friendly_error(exc: BaseException) -> str:
    """
    One readable line for an API or network failure, saying what to do about it.

    Matches on the HTTP status and the exception's class name instead of importing
    the SDK's exception types, so it also works for any error carrying a
    `status_code`, and it can be tested without the SDK.
    """
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    detail = str(exc) or name

    if status == 401:
        return "The API rejected your key (401). Check ANTHROPIC_API_KEY in .env."
    if status == 403:
        return "Your key does not have permission for that request (403)."
    if status == 404:
        return "The API could not find that model (404). Check the --model name."
    if status == 429:
        return "Rate limited (429). Wait a moment and try again."
    if isinstance(status, int) and status >= 500:
        return f"Anthropic's servers are having trouble ({status}). Try again shortly."
    if status == 400:
        return f"The API rejected the request (400): {detail}"
    if "Timeout" in name:
        return "The request timed out. Try again."
    if "Connection" in name:
        return "Could not reach the API. Check your internet connection."
    if name == "RuntimeError":
        return detail  # ours, already written for the user
    return f"{name}: {detail}"


@dataclass
class Usage:
    """Running token count for one session, so cost is measured rather than guessed."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, api_usage: Any) -> None:
        self.calls += 1
        self.input_tokens += getattr(api_usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(api_usage, "output_tokens", 0) or 0

    def cost(self, model: str) -> Optional[float]:
        """Estimated dollars, or None if we do not know this model's price."""
        if model not in PRICES_PER_MTOK:
            return None
        price_in, price_out = PRICES_PER_MTOK[model]
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000

    def summary(self, model: str) -> str:
        cost = self.cost(model)
        price = f", ~${cost:.4f}" if cost is not None else ""
        return f"{self.calls} calls, {self.input_tokens} in / {self.output_tokens} out tokens{price} ({model})"


@dataclass
class ClaudeLLM:
    """
    Real Claude. One method, deliberately — this package only ever needs to say
    "here is who you are, here is the conversation so far, what do you say next".

    The client is created lazily so importing this module never requires a key.
    """

    model: str = DEFAULT_MODEL
    _client: Any = field(default=None, repr=False)
    usage: Usage = field(default_factory=Usage)

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                    "or run in offline mode: uv run main.py --offline"
                )
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        """Send one request and return the concatenated text of the reply."""
        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        self.usage.add(response.usage)

        # Claude can decline a request; content is empty or partial when it does.
        if response.stop_reason == "refusal":
            return "[declined to respond]"

        text = "".join(block.text for block in response.content if block.type == "text").strip()

        # Thinking models can spend the whole limit before writing anything. An empty
        # reply would otherwise pass for a real (blank) line of dialogue or persona.
        if not text and response.stop_reason == "max_tokens":
            raise RuntimeError(_OUT_OF_TOKENS)
        return text

    def complete_with_tools(
        self, system: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], max_tokens: int = 4096
    ) -> "ToolResponse":
        """One request with tools attached; returns text and tool_use blocks as dicts."""
        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )
        self.usage.add(response.usage)

        if response.stop_reason == "refusal":
            return ToolResponse([{"type": "text", "text": "[declined to respond]"}])

        blocks: List[Dict[str, Any]] = []
        for block in response.content:
            if block.type == "text" and block.text.strip():
                blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
            elif block.type in ("thinking", "redacted_thinking"):
                # Thinking models require their own thinking blocks, signature and
                # all, to be sent back unchanged on the next request in a tool loop.
                blocks.append(block.model_dump(exclude_none=True))

        visible = [b for b in blocks if b["type"] in ("text", "tool_use")]
        if not visible and response.stop_reason == "max_tokens":
            raise RuntimeError(_OUT_OF_TOKENS)
        return ToolResponse(blocks)


@dataclass
class ToolCall:
    """One tool the model asked to run."""

    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ToolResponse:
    """
    A model reply that may contain tool calls, as plain dicts.

    `content` is already in the shape the Messages API wants echoed back in the
    next request, so the orchestrator never touches SDK objects and can be tested
    without the SDK.
    """

    content: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(b["text"] for b in self.content if b["type"] == "text").strip()

    @property
    def tool_calls(self) -> List[ToolCall]:
        return [ToolCall(b["id"], b["name"], b["input"]) for b in self.content if b["type"] == "tool_use"]


def text_response(text: str) -> ToolResponse:
    """A reply with no tool calls."""
    return ToolResponse([{"type": "text", "text": text}])


def tool_response(name: str, input: Dict[str, Any], call_id: str = "toolu_test") -> ToolResponse:
    """A reply that asks to run one tool."""
    return ToolResponse([{"type": "tool_use", "id": call_id, "name": name, "input": input}])


@dataclass
class ScriptedToolLLM:
    """
    Plays back a fixed list of ToolResponses for `complete_with_tools`, and
    delegates plain `complete` to a DemoLLM so persona authoring and dialogue
    still work. Used to test the tool-use loop with no key and no network.
    """

    script: List[ToolResponse] = field(default_factory=list)
    tool_calls_seen: List[Dict[str, Any]] = field(default_factory=list)
    delegate: Any = None

    def __post_init__(self) -> None:
        if self.delegate is None:
            self.delegate = DemoLLM()

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        return self.delegate.complete(system, messages, max_tokens)

    def complete_with_tools(
        self, system: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], max_tokens: int = 4096
    ) -> ToolResponse:
        self.tool_calls_seen.append({"system": system, "messages": list(messages), "tools": tools})
        index = len(self.tool_calls_seen) - 1
        if index >= len(self.script):
            return text_response("(script exhausted)")
        return self.script[index]


@dataclass
class ScriptedLLM:
    """
    A fake model that replays a fixed list of replies, then loops.

    Used by the tests and by `--offline`. It records every call it received in
    `.calls`, which makes it easy to assert that a persona's markdown actually
    reached the system prompt.
    """

    replies: List[str] = field(default_factory=lambda: ["(scripted reply)"])
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        self.calls.append({"system": system, "messages": list(messages)})
        return self.replies[(len(self.calls) - 1) % len(self.replies)]


_DEMO_DIALOGUE = [
    "I understand. Could you tell me more about when the symptoms started?",
    "That has been going on for about three weeks now, mostly in the mornings.",
    "Have you noticed anything that makes it better or worse?",
    "Coffee seems to make it worse, and lying down helps a little.",
    "Thank you, that is useful. I would like to run a couple of tests.",
    "Whatever you think is best, doctor. I just want to feel normal again.",
]

_DEMO_PERSONA = """---
name: {name}
role: {role}
summary: a demo persona generated in offline mode
---

# {name}

## Background
This persona was produced by the offline stub, not by a model. Run without
`--offline` to get a real character written by Claude.

## Personality
- Placeholder
- Placeholder

## How they speak
- In whatever the scripted stub happens to return
"""


@dataclass
class DemoLLM:
    """
    The offline stub used by `--offline`.

    A single ScriptedLLM cannot serve both jobs: asked to author a persona it
    would return a line of dialogue, and you would get a character named
    "Unnamed". So this one looks at the system prompt and answers in the shape
    the caller expects — persona markdown for the author prompt, dialogue for
    everything else.
    """

    calls: List[Dict[str, Any]] = field(default_factory=list)

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        self.calls.append({"system": system, "messages": list(messages)})
        request = messages[-1]["content"] if messages else ""

        # The persona-author prompt is the only one that asks for frontmatter.
        if "role: <one lowercase word" in system:
            description = request.replace("Write a persona for:", "").strip()
            words = [w for w in description.split() if w.isalpha()]
            role = words[0].lower() if words else "person"
            return _DEMO_PERSONA.format(name=description[:40].title() or "Demo Persona", role=role)

        if "stage manager" in system.lower():
            return "Offline mode — I'm a stub. Try: create a persona doctor"

        return _DEMO_DIALOGUE[(len(self.calls) - 1) % len(_DEMO_DIALOGUE)]


def default_llm(offline: bool = False, model: Optional[str] = None) -> LLM:
    """Pick an implementation. Central so you only change this in one place."""
    if offline:
        return DemoLLM()
    return ClaudeLLM(model=model or DEFAULT_MODEL)
