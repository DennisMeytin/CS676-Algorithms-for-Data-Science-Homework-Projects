"""
cli.py — the REPL loop.

`main.py` at the project root is a two-line wrapper around `main()` here, so the
app can be started either way:

    uv run main.py
    uv run personaforge
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import ui
from .llm import default_llm, friendly_error
from .orchestrator import Orchestrator
from .persona import DEFAULT_PERSONA_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PersonaForge — agent-to-agent persona simulation.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use scripted replies instead of calling Claude (no API key needed)",
    )
    parser.add_argument(
        "--persona-dir",
        default=str(DEFAULT_PERSONA_DIR),
        help=f"where persona .md files are stored (default: {DEFAULT_PERSONA_DIR})",
    )
    parser.add_argument("--model", default=None, help="override the Claude model id")
    return parser


def _print_usage(llm) -> None:
    """Show what a live session cost. Offline models have no `usage`, so print nothing."""
    usage = getattr(llm, "usage", None)
    if usage is not None and usage.calls:
        ui.console.print(f"[dim]usage: {usage.summary(llm.model)}[/dim]")


def _goodbye(llm) -> int:
    """Leave the REPL: say bye, report the session's cost, return the exit code."""
    ui.console.print("[dim]bye[/dim]")
    _print_usage(llm)
    return 0


def main() -> int:
    """Run the terminal app until the user quits. Returns a process exit code."""
    args = build_parser().parse_args()
    load_dotenv()

    # The UI prints non-ASCII glyphs (arrows, em dashes). On Windows the console
    # or a pipe can default to cp1252, which cannot encode them and crashed the
    # app mid-conversation. Force UTF-8 and degrade to '?' rather than raising.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    # Fail once, up front, instead of once per message.
    if not args.offline and not os.getenv("ANTHROPIC_API_KEY"):
        ui.error(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or run: uv run main.py --offline"
        )
        return 1

    llm = default_llm(offline=args.offline, model=args.model)
    orchestrator = Orchestrator(llm=llm, persona_dir=Path(args.persona_dir))

    ui.banner(offline=args.offline, persona_dir=args.persona_dir)

    # Dialogue is printed as each turn arrives rather than in one block at
    # the end, so a six-turn conversation feels like a conversation.
    def on_turn(turn):
        ui.dialogue_line(turn.speaker, turn.text)

    while True:
        try:
            text = ui.user_prompt()
        except (EOFError, KeyboardInterrupt):
            ui.console.print()
            return _goodbye(llm)

        if not text:
            continue

        # Slash commands are handled here rather than in the orchestrator: they
        # are about the session, not about personas.
        if text in {"/quit", "/exit", "/q"}:
            return _goodbye(llm)
        if text == "/help":
            ui.help_text()
            continue
        if text == "/clear":
            ui.console.clear()
            continue

        try:
            result = orchestrator.handle(text, on_turn=on_turn, on_start=ui.conversation_header)
        except KeyboardInterrupt:
            # Ctrl-C while a request is running cancels that request, not the session.
            ui.console.print("\n[dim]interrupted. Type /quit to leave.[/dim]")
            continue
        except Exception as exc:  # noqa: BLE001 — a crashed turn shouldn't kill the session
            ui.error(friendly_error(exc))
            continue

        ui.agent_says(result.reply)
        for message in result.events:
            ui.event(message)
