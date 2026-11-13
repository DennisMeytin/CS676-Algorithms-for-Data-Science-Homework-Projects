# PersonaForge results: measurements, transcripts and failure notes

The detailed engineering record behind `REPORT.md`. Numbers come from the saved runs in
`report_materials/` and `evals/results/`; where a claim is an inference rather than
something measured, the text says so. The paper-format report is `REPORT.md`.

---

## 1. Scope and rationale

The starter README lists five directions and says depth beats breadth. I chose **one
headline change and one supporting theme**:

1. **Replace the regex router with real tool use** (README item 1). The model now
   decides what the user meant and calls tools.
2. **Make the persona lifecycle safe**: no silent overwrites, honest lookup, validated
   model output, and a new edit-and-undo feature. This was not on the README list. I
   added it because tool use lets the model *change* things, and a router that can
   overwrite files needs a safety net before it is trusted.

Deliberately **not done**: agent memory, smarter turn-taking, an observer agent, and
self-updating personas. They are listed in section 6 with how they would build on this
work.

I chose depth here because the tool loop is the part of the system that every other
feature depends on. A weak router makes every later feature feel broken.

---

## 2. Design: what changed and why

### 2.1 Tool-use routing (`orchestrator.py`, `llm.py`)

**Before.** `handle()` was a stack of regular expressions. It could not ask a clarifying
question, pattern order decided meaning ("list the personas then have them talk" only
listed), and `turns` was hardcoded to 6.

**After.** If the model supports tools, `handle()` runs a loop:

1. Send the conversation plus five tool schemas to the model.
2. If the reply contains tool calls, run them and send the results back.
3. Repeat until the model replies with plain text, which is shown to the user.

| Tool | Purpose |
|---|---|
| `create_persona(description)` | Author and save one persona |
| `list_personas()` | Show what exists |
| `edit_persona(persona, change)` | Rewrite a persona in place |
| `undo_edit(persona)` | Restore the previous version |
| `start_conversation(participants, topic, turns)` | Run a conversation |

`ClaudeLLM` gained `complete_with_tools()`, which returns text and tool calls as plain
dictionaries, so the orchestrator never touches SDK objects and can be tested with a
scripted model.

**Safeguards**, each added because a tool loop can fail in ways a command parser cannot:

| Risk | Safeguard |
|---|---|
| Model loops on tools forever | Capped at 6 round-trips per user message |
| Absurd arguments (`turns: 500`) | Turns clamped to 2–20 |
| A tool raises | Error returned to the model as text; it can recover or tell the user |
| History grows without limit | Trimmed to 40 messages |
| Trimming splits a tool call from its result (the API rejects this) | Trim only at a plain user message |
| Model invents persona names | Prompt requires `list_personas` first when unsure; unknown names return an error |

**The regex router is kept** as the fallback for `--offline`, because the scripted
offline model cannot call tools. This keeps `uv run main.py --offline` and the whole test
suite free to run.

Starter-file issues this resolves: clarifying questions, chained requests, settable turn
count, silent "first two personas" selection, and an orchestrator with no memory of what
it created.

### 2.2 Persona robustness (`persona.py`)

| Problem in the starter | Change |
|---|---|
| `save_persona` silently overwrote a same-name file | Saves as `name-2.md`, `name-3.md`; overwrite is opt-in |
| `find_persona` matched substrings ("doc" found "Dorothy") and returned the first hit | Exact match first, then whole-word match; several matches raise `AmbiguousPersona` listing the candidates |
| Models write roles like "Primary Care Doctor" despite the one-word rule | Role normalized to one lowercase word |
| Model output with a stray `---` before the real header produced a persona named after the raw description | Parser lifts `name`/`role`/`summary` out of the top of the body (see 4.1) |

### 2.3 Edit and undo (`persona.py`)

Editing is the first operation that destroys data, so it is built around three rules:

1. **Validate before writing.** The model's rewrite is parsed and checked: it must be a
   real persona (not prose), must not be implausibly short (under 100 characters), and
   must keep every `##` section heading the original had. On failure nothing is written.
2. **Back up before replacing.** The old file is copied to
   `<persona_dir>/.history/<name>--<timestamp>.md`. `undo_edit` restores the newest
   backup and deletes it, so repeating it steps back further.
3. **Identity is locked.** Name and role cannot change. The filename and
   `find_persona` both key on them, and a model that "helpfully" renames a character has
   still done the edit we wanted, so name and role are forced back rather than the whole
   edit rejected.

Alternatives considered:

| Alternative | Why not |
|---|---|
| Let edits rename a persona | Breaks filenames, history, and lookup by role ("the doctor") |
| A database for history | A folder of dated markdown files matches the project's idea that a persona is a file, and a person can open it |
| Fake editing in `--offline` mode | A canned stub cannot rewrite text meaningfully; `--offline` says editing needs a live model |

### 2.4 Smaller changes

- **Cost tracking.** `ClaudeLLM` counts calls and tokens and prints a one-line estimate
  when a session ends. The figures in this report come from it.
- **UTF-8 output.** `cli.py` forces UTF-8 stdout (see 4.2).
- **Higher `max_tokens`** on tool routing, chat, persona writing and replies, because
  the newer models think before answering and thinking counts against the limit.

---

## 3. Saved transcript and commentary

**Model:** Claude Opus 5.5 (`claude-opus-5-5`). **Run:** `report_materials/final_opus-5-5_session.txt`.
**Cost:** 24 API calls, 56,432 input and 10,578 output tokens, about **$0.44**.

The session script (`report_materials/final_script.txt`):

```
create a persona
a skeptical, overworked ER doctor who respects data but hates being second-guessed
create a patient who is sure her chronic back pain is being dismissed and who keeps her own symptom log
list personas
make the doctor a little more patient with questions
have them discuss whether she needs an MRI for 10 turns
```

The full 10-turn conversation is in `report_materials/final_opus-5-5_conversation.md`.
Two short excerpts:

> **Marguerite:** What brought me in is the numbness in my left leg. Outer calf, down toward
> the ankle. I've logged it for fourteen months, and it tracks with cold mornings. [...]
> I'm not asking for a scan for the sake of it. I'm asking whether that pattern means
> anything to you.
>
> **Renata:** That's the best history anyone's handed me this month. And yes, the pattern
> means something to me. [...] A disc pressing on a root usually tracks with position,
> sitting, bending. It doesn't usually care about the thermometer.

### What worked

- **The whole feature chain ran in one session with no errors.** The model asked what
  kind of persona to create when the first message gave none (the clarifying question
  the regex router could not ask), created two, listed them, edited one, and ran the
  conversation. Its summary of the edit was accurate and it offered to undo it.
- **The two voices are clearly different.** The patient speaks in dates, readings and
  notebook entries ("volume eleven"); the doctor asks structured clinical questions and
  explains her reasoning.
- **The patient's detail changes the plot.** The numbness-against-temperature log moves the
  doctor's suspicion from the spine to a nerve at the knee. The characters are using
  what is written in their persona files, which is the point of the architecture.

### What did not work

- **There was no argument.** The doctor was written to hate being second-guessed, but the
  patient never second-guessed her, and the edit I requested made her warmer. The system's
  own summary said "they didn't argue at all". This is the agreement drift analysed in 4.3.
- **Replies are far longer than the prompt asks for.** The agent prompt says "a few
  sentences", but many turns run several paragraphs, which also raises cost.
- **The doctor performs the physical exam in dialogue** ("Walk to the door and back on
  your heels"), which is close to the stage directions the prompt forbids.
- **The clinical content is plausible fiction and was not verified.** Nothing in this
  report or transcript is medical advice.

---

## 4. Failure analysis

### 4.1 Structured output: the model breaks the file format

Personas are markdown files with a frontmatter header, and the model writes them. The
starter README already documents one failure (a missing closing `---`).

**New failure found in live testing.** In my first live run (Haiku 5.5), the patient was
saved as "A Patient With Chronic Back Pain" with role `unknown`, which is the parser's
fallback. The saved file (`report_materials/live_run1_malformed_persona.md`) shows a real header
(`name: Dolores Pruitt`, `role: patient`) sitting *inside the body*. My inference is that
the model emitted a stray `---` before its header, so the parser closed an empty header
and left the real one in the body. I did not capture the model's raw output, so the
regression test reconstructs that shape rather than replaying it. The parser now lifts
the three known keys from the top of the body when the normal header is empty.

This failed silently: no error, a wrong name, and the stage manager only mentioned it
because it noticed. The lesson matches the README's: be strict about what matters (a
real persona, the right sections) and generous about what doesn't (where a fence sits).

**Frequency.** Across the 18 persona creations in the later live runs (three model
comparisons, two repeat runs, the final run and the edit/undo run), none had the fallback
name or `role: unknown`. So this failure is intermittent and I cannot give a rate from
these numbers.

### 4.2 Platform failure: Windows console encoding

Piped output on Windows defaults to cp1252, which cannot encode the arrows and dashes the
UI prints, so the app crashed with `UnicodeEncodeError` while printing a reply.
`cli.py` now forces UTF-8 and replaces unprintable characters rather
than raising. This only appeared because I ran the app with piped input, which is how a
scripted demo would run it.

### 4.3 Agents drift toward agreement

In every comparison run (Haiku, Sonnet, Opus) I asked for an *argument*, and the
conversations ended in agreement or rapport. The models' own summaries said so: Opus
"didn't turn into much of an argument", Sonnet "less like a fight and more like Dale
slowly giving in".

**Attempted fix.** I added a rule to the agent prompt: hold your position, concede only for
a genuinely good reason, and only partway. Two Haiku runs with the rule were inconclusive,
so I built a proper test.

**Evaluation** (`evals/hold_position_eval.py`, results in `evals/results/hold_position.json`):

- **Scenario with opposed goals:** a patient insists on antibiotics for a cold; the doctor
  refuses without evidence. Ten turns each.
- **Arms:** 5 conversations without the rule, 5 with it. Characters played by Haiku 5.5.
- **Scoring:** each transcript scored blind by a separate model (Sonnet 5.5) that was not
  told which arm produced it: how firmly each side held its position at the end (1–5), and the
  outcome.

| Arm | Patient gave in | Mean "patient held" | Mean "doctor held" |
|---|---|---|---|
| Baseline (no rule) | 4 of 5 | 2.4 | 5.0 |
| With hold rule | 4 of 5 | 2.4 | 4.4 |

**Result: the rule had no measurable effect**, so I reverted it. The patient gave up the
demand in 8 of 10 conversations either way.

My interpretation, which is a hypothesis and not something this test establishes: the
patient's position is medically wrong, and the models are reluctant to hold it to the
end however they are prompted. If so, "agents cave" is partly the model's values, not just
a prompt gap.

**Limits of this test:** one scenario, one character model, one judge, five runs per arm,
and a judge that is itself a model (I spot-checked three of its scores against the
transcript endings and they matched).

### 4.4 Cost

Same 9-line script on three models (`report_materials/live_script.txt`), one run each:

| | Haiku 5.5 | Sonnet 5.5 | Opus 5.5 |
|---|---|---|---|
| API calls | 28 | 28 | 28 |
| Input tokens | 65,776 | 69,442 | 69,675 |
| Output tokens | 6,972 | 8,357 | 9,047 |
| **Estimated cost** | **$0.010** | **$0.22** | **$0.46** |

- Usage is almost identical across models, so cost differences come from price, not
  behaviour. Opus costs about 46 times Haiku for the same session.
- **Cost drivers.** Every conversation turn re-sends the whole transcript, so a conversation costs
  more than linearly in its length. Verbose replies (4.3 above) multiply this.
- **Estimates, not invoices.** Costs are computed from published per-million-token rates in
  `llm.py` and the token counts the API returned. The eval's 10 conversations cost about
  $0.023; judge calls are not included in that figure.
- **One run per model** is a comparison, not a benchmark.

### 4.5 Known gaps (not fixed, not hidden)

- **Edit validation is structural.** It checks headings and length, so a rewrite that
  keeps the headings but guts the content would pass.
- **Undo can overwrite a hand edit.** If a persona file is edited by hand after an
  `edit_persona` call, `undo_edit` restores the pre-edit backup and the hand edit is lost.
- **No protection against concurrent edits** to the same file.
- **The ambiguity error was not shown firing live.** In all runs where two doctors existed,
  the model asked "which doctor?" on its own, having just created both. The error path is
  unit-tested but I cannot say the live model ever triggered it.
- **Agents have no memory** between conversations, and turn-taking is still strict
  round-robin.

---

## 5. Testing

**152 tests, all offline, about two seconds.** The starter had 41 (the README
says 39; the repo has 41). I added 41.

| File | Tests | Covers |
|---|---|---|
| `test_agent.py` | 6 | starter |
| `test_conversation.py` | 15 | starter |
| `test_llm.py` | 9 | starter 7, plus usage counting and pricing |
| `test_persona.py` | 25 | starter 13, plus no-overwrite, lookup, ambiguity, role normalization, parser recovery, transcripts not listed as personas |
| `test_orchestrator_tools.py` | 13 | tool loop, error handling, step cap, clamping, history trimming, offline fallback |
| `test_edit.py` | 17 | edit, validation failures, locked identity, backup, undo (including multi-step), routing through the model |
| `test_robustness.py` | 26 | friendly API errors, truncated replies, partial conversations, Ctrl-C, startup check |
| `test_public_api.py` | 6 | every exported name imports, new features are public, version matches the installed package, errors are catchable as `ValueError` |
| `test_edge_cases.py` | 35 | final-audit gaps: odd tool inputs (string participants, a persona talking to itself, bad `turns`), unreadable files, refusals, thinking blocks, missing key, REPL branches |

**How the model is faked.** `ScriptedToolLLM` replays a fixed sequence of tool calls and
text replies and records what it was sent, so tests can assert on exactly what the
orchestrator returned to the model (for example, that an error carries `is_error`).

**What the tests do not cover.** They prove the logic is right given a model that behaves
as scripted; they cannot show a real model will follow the schemas. That is what the live
runs are for. Behaviour that only appears with a real model (4.1, 4.3) is covered by saved
runs, not by tests. The live tool path was verified on three models in one scripted
session each, which is a small sample.

---

## 6. Limitations and future work

| Idea | How it builds on this work |
|---|---|
| **Observer agent** that scores conversations | The hold-position eval already uses a judge; an in-app observer would reuse that prompt and flag agreement drift as it happens |
| **Agent memory** | Appending "what I learned" to the persona file would go through `edit_persona`, so it inherits validation and backups |
| **Self-updating personas** | Same path, with a reviewable proposed change instead of an automatic write |
| **Smarter turn-taking** | A `start_conversation` option the model sets (who speaks next, when to stop) |
| **Verbosity control** | Enforce the "few sentences" rule, which would cut cost and sharpen voices |

---

## Appendix: reproducing this

```bash
uv sync
uv run pytest                              # 152 tests, no key needed
uv run main.py --offline                   # whole app, scripted model
uv run main.py --model claude-haiku-5-5    # live; needs ANTHROPIC_API_KEY in .env
uv run python evals/hold_position_eval.py  # the agreement-drift eval (billed)
```

- **Models used:** Haiku 5.5, Sonnet 5.5 and Opus 5.5 for characters; Sonnet 5.5 as the
  judge. The submitted transcript is from Opus 5.5.
- **Evidence:** `report_materials/` (raw sessions, transcripts, personas before and after
  an edit) and `evals/results/` (eval data and log).
