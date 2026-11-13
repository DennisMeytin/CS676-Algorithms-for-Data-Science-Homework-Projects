# PersonaForge: Submission Notes

CS676 Project 2 · Dennis Meytin

This file summarizes what was changed in the professor's PersonaForge skeleton and what
came out of it. The professor's original `README.md` is unchanged (apart from one pointer
line to this file). The formal write-up is [`REPORT.md`](REPORT.md), and the detailed
measurements are in [`RESULTS.md`](RESULTS.md).

## Scope

**In scope.** Replace the regular-expression router with real tool use, and make the
persona files safe to manage from a model that can change them.

**Out of scope (not attempted).** Agent memory, smarter turn-taking, an observer agent,
self-updating personas, and the Gradio/Streamlit deployment bonus. `REPORT.md` section 5 names the most useful next steps (an observer agent that scores
whether characters hold their positions, and a limit on reply length).

## What changed

| Area | Change |
|---|---|
| Routing (`orchestrator.py`, `llm.py`) | With a live model, Claude chooses from five tools (`create_persona`, `list_personas`, `edit_persona`, `undo_edit`, `start_conversation`). It asks clarifying questions and can chain actions. Safeguards: 6 tool round-trips per message, turns clamped to 2–20, errors returned to the model as text, history trimmed to 40 messages without separating a tool call from its result. The regex router remains as the `--offline` fallback. |
| Persona files (`persona.py`) | Saving never overwrites (`name-2.md`); lookup matches whole words and raises an error listing the candidates when several match; roles are normalized to one word; the parser recovers a header that follows a stray `---`. |
| Edit and undo (`persona.py`) | The model's rewrite is validated before anything is written (real persona, not too short, all section headings kept). The old file is backed up to `.history/` first. Name and role cannot change. `undo_edit` restores the newest backup. |
| Cost report (`llm.py`, `cli.py`) | Token counts per session and an estimated cost printed on exit. |
| Windows fix (`cli.py`) | Output forced to UTF-8, which fixed a crash on piped output. |
| Robustness (`llm.py`, `conversation.py`, `orchestrator.py`, `cli.py`) | API failures print one plain-language line (bad key, wrong model, rate limit, no connection) instead of a traceback. A conversation that fails or is interrupted with Ctrl-C keeps the turns so far and saves the partial transcript. Ctrl-C during a request cancels that request, not the session, and keeps the tool history valid. A live session without a key refuses to start. An empty reply that ran out of tokens raises instead of passing for a blank line. |
| Public API (`__init__.py`) | The new functions and types are exported (`edit_persona`, `undo_edit`, `EditResult`, `AmbiguousPersona`, `PersonaEditError`, `friendly_error`, `Usage`, `ToolCall`, `ToolResponse`, `ScriptedToolLLM`, `DemoLLM`); the module docstring shows the orchestrator in use; version bumped from 0.1.0 to 0.2.0 (`pyproject.toml` and the lock updated to match). |
| Transcript bug (`persona.py`) | A saved `conversation.md` was listed as a persona called "Unnamed", and offline it could be picked as a character in the next conversation. It is now skipped, and a persona named "Conversation" cannot collide with it. Found by the clean-copy test. |
| Final-audit fixes (`orchestrator.py`, `persona.py`, `llm.py`, `ui.py`) | A persona can no longer be asked to talk to itself. Participants sent as one string ("Ana and Bo") are split instead of read letter by letter. An unreadable file in the persona folder is skipped instead of crashing every listing. Thinking blocks from thinking models are sent back unchanged in the tool loop. `/help` now mentions edit and undo. |
| Tests | 152 offline tests (the skeleton had 41), 99% line coverage. |

## Run it

```bash
uv sync
uv run pytest                              # 152 tests, no key needed
uv run main.py --offline                   # whole app with a scripted model
uv run main.py --model claude-haiku-5-5    # live; needs ANTHROPIC_API_KEY in .env
```

In the live app, try: `create a persona` (it will ask what kind), then
`make the doctor more skeptical`, `undo that edit`, and
`have them discuss the MRI results for 10 turns`. Editing and undo need a live model.

## Results

**Cost of one session** (the same nine-line script, one run per model):

| | Haiku 5.5 | Sonnet 5.5 | Opus 5.5 |
|---|---|---|---|
| API calls | 28 | 28 | 28 |
| Estimated cost | $0.010 | $0.22 | $0.46 |

The submitted transcript is from Opus 5.5 (`report_materials/final_opus-5-5_conversation.md`,
24 calls, about $0.44).

**Agreement drift.** Asked to argue, characters tend to end up agreeing. A prompt rule
telling them to hold their position was tested on 10 conversations (5 with, 5 without,
scored blind by a separate model) and made no measurable difference: the patient gave in
4 of 5 times either way. The rule was removed. Details in `REPORT.md` section 4.5.

**Failures found and fixed:** a model-written persona header that broke the parser, and
a Windows encoding crash. Open weaknesses (validation is structural, undo can overwrite a
hand edit, the ambiguity error was not seen firing in a live run) are listed in
`REPORT.md` section 5.

## Files

| Path | What it is |
|---|---|
| `src/personaforge/` | The package |
| `tests/` | 152 offline tests (`test_orchestrator_tools.py`, `test_edit.py`, `test_robustness.py`, `test_public_api.py` and `test_edge_cases.py` are new) |
| `evals/` | The agreement-drift experiment (`hold_position_eval.py`) and its results |
| `report_materials/` | Raw live sessions, transcripts, and personas before and after an edit |
| `REPORT.md` | Technical report |
| `Project2_Report.pdf` | The same report as a PDF |
| `RESULTS.md` | Detailed measurements and failure notes |
