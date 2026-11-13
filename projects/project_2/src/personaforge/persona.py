"""
persona.py — a persona is a markdown file on disk.

That is the whole idea of this package. There is no persona database and no
in-memory registry. A persona is a `.md` file in `temp/`, and anything that can
read that file can wake the persona up. You can open one in a text editor,
change a sentence, and the agent behaves differently on the next run.

File format — frontmatter for the machine, markdown for the model:

    ---
    name: Maria Delgado
    role: patient
    summary: 58-year-old with poorly controlled type 2 diabetes
    ---

    # Maria Delgado

    ## Background
    ...

The frontmatter parser handles exactly the three keys above, plus recovery for
the two malformed headers models actually produce: a missing closing fence and a
stray fence before the real header. See KNOWN LIMITS at the bottom.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .llm import LLM

# Where persona files live. Relative to wherever you launched the app.
DEFAULT_PERSONA_DIR = Path("temp")

# Conversation transcripts are saved next to the personas, so they match the same
# `*.md` pattern. This name is skipped when listing personas, otherwise the last
# transcript shows up as a character called "Unnamed".
TRANSCRIPT_FILENAME = "conversation.md"


@dataclass
class PersonaSpec:
    """One persona: the metadata we index on, plus the markdown that is its mind."""

    name: str
    role: str
    summary: str
    body: str  # markdown; becomes the agent's system prompt

    def slug(self) -> str:
        """Filesystem-safe stem, e.g. 'Maria Delgado' -> 'maria-delgado'."""
        s = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-") or "unnamed"
        if f"{s}.md" == TRANSCRIPT_FILENAME:  # a persona called "Conversation" must not collide
            s += "-persona"
        return s

    def to_markdown(self) -> str:
        """Serialize to the frontmatter + body format shown above."""
        return (
            "---\n"
            f"name: {self.name}\n"
            f"role: {self.role}\n"
            f"summary: {self.summary}\n"
            "---\n\n"
            f"{self.body.strip()}\n"
        )

    @classmethod
    def from_markdown(cls, text: str) -> "PersonaSpec":
        """
        Parse a persona file back into a spec.

        Missing frontmatter is tolerated: we fall back to sensible defaults and
        treat the whole file as the body, so a hand-written persona still loads.
        """
        meta = {"name": "Unnamed", "role": "unknown", "summary": ""}
        text = text.strip()

        # Models sometimes wrap the whole document in a code fence.
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
            text = text.strip()

        body = text

        if text.startswith("---"):
            rest = text[3:].lstrip("\n")
            closing = re.search(r"^---\s*$", rest, re.MULTILINE)
            if closing:
                # Well-formed: frontmatter is everything up to the closing fence.
                front, body = rest[:closing.start()], rest[closing.end():]
            else:
                # The closing '---' is missing, which the model does drop from
                # time to time. Rather than throw the whole document away, read
                # `key: value` lines from the top and treat the first line that
                # is not one as the start of the body.
                front_lines, body_lines, in_front = [], [], True
                for line in rest.splitlines():
                    if in_front and re.match(r"^[A-Za-z_][A-Za-z0-9_ -]*:", line):
                        front_lines.append(line)
                    elif in_front and not line.strip():
                        continue            # blank line inside the header block
                    else:
                        in_front = False
                        body_lines.append(line)
                front, body = "\n".join(front_lines), "\n".join(body_lines)

            for line in front.splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip().lower()
                    if key in meta:
                        meta[key] = value.strip()

        # Seen live: the model emitted a stray '---' (or a blank header) before the
        # real frontmatter, so the first fence closed an empty header and the real
        # `name:` block ended up in the body. If we found no name, look for the
        # three known keys at the top of the body and lift them out.
        if meta["name"] == "Unnamed":
            lines = body.strip().splitlines()
            found, consumed = {}, 0
            for i, line in enumerate(lines[:12]):
                m = re.match(r"^(name|role|summary):\s*(.+?)\s*$", line.strip(), re.IGNORECASE)
                if m:
                    found[m.group(1).lower()] = m.group(2)
                    consumed = i + 1
                elif line.strip() in ("", "---"):
                    consumed = max(consumed, i + 1) if found else consumed
                elif found:
                    break
            if "name" in found:
                meta.update(found)
                body = "\n".join(lines[consumed:])

        return cls(name=meta["name"], role=meta["role"], summary=meta["summary"], body=body.strip())


def save_persona(spec: PersonaSpec, directory: Path = DEFAULT_PERSONA_DIR, overwrite: bool = False) -> Path:
    """
    Write the persona to `<directory>/<slug>.md`, creating the folder if needed.

    An existing file is never replaced unless `overwrite=True`: a second
    "Maria Delgado" is saved as `maria-delgado-2.md` instead of destroying the first.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec.slug()}.md"
    if not overwrite:
        n = 2
        while path.exists():
            path = directory / f"{spec.slug()}-{n}.md"
            n += 1
    path.write_text(spec.to_markdown(), encoding="utf-8")
    return path


def load_persona(path: Path) -> PersonaSpec:
    """Read one persona file from disk."""
    return PersonaSpec.from_markdown(Path(path).read_text(encoding="utf-8"))


def _persona_files(directory: Path) -> List[Path]:
    """Persona files in a folder, sorted by name. Transcripts are not personas."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return [p for p in sorted(directory.glob("*.md")) if p.name != TRANSCRIPT_FILENAME]


def _try_load(path: Path) -> Optional[PersonaSpec]:
    """Load one persona file, or None if it cannot be read as text (a stray binary file)."""
    try:
        return load_persona(path)
    except (UnicodeDecodeError, OSError):
        return None


def list_personas(directory: Path = DEFAULT_PERSONA_DIR) -> List[PersonaSpec]:
    """
    Every persona currently saved, sorted by filename. Empty if none yet.

    A file in the folder that is not readable text is skipped instead of making
    every listing and conversation fail.
    """
    loaded = (_try_load(p) for p in _persona_files(directory))
    return [spec for spec in loaded if spec is not None]


class AmbiguousPersona(ValueError):
    """More than one persona matches what the user typed."""

    def __init__(self, needle: str, candidates: List[PersonaSpec]) -> None:
        self.candidates = candidates
        names = ", ".join(f"{c.name} ({c.role})" for c in candidates)
        super().__init__(f"'{needle}' matches several personas: {names}. Use a full name.")


def find_persona(name: str, directory: Path = DEFAULT_PERSONA_DIR) -> Optional[PersonaSpec]:
    """
    Look a persona up by name, slug, or role — case-insensitively.

    Users type "the doctor" or "maria", not exact filenames, so loose matches
    are allowed, but never a guess:
      1. an exact name or slug match wins outright;
      2. otherwise an exact role, then a whole-word match inside a name;
      3. if that still leaves more than one persona, raise AmbiguousPersona
         instead of silently picking the first;
      4. no match returns None.
    Whole-word matching means "doc" no longer finds "Dorothy".
    """
    needle = name.strip().lower()
    specs = list_personas(directory)

    exact = [s for s in specs if needle in (s.name.lower(), s.slug())]
    if len(exact) == 1:
        return exact[0]

    def words(s: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", s.lower())

    needle_words = words(needle)
    candidates = exact or (
        [s for s in specs if needle == s.role.lower()]
        or [s for s in specs if needle_words and all(w in words(s.name) for w in needle_words)]
    )

    if len(candidates) > 1:
        raise AmbiguousPersona(name, candidates)
    return candidates[0] if candidates else None


# -----------------------------------------------------------------------------
# Generating a persona with the model
# -----------------------------------------------------------------------------

_AUTHOR_SYSTEM = """You write character specifications for a multi-agent simulation.

Given a short description, write a persona as a markdown document. Output ONLY the
markdown — no preamble, no code fences.

Use exactly this structure:

---
name: <a specific, realistic full name>
role: <one lowercase word, e.g. patient, doctor, teacher, customer>
summary: <one line, under 15 words>
---

# <name>

## Background
Two or three sentences. Be concrete and specific — age, situation, history.

## Personality
Three or four bullet points about temperament and outlook.

## How they speak
Two or three bullets on voice: vocabulary, sentence length, verbal habits.

## What they want
One or two sentences on their goal in a conversation.

## What they avoid
One or two sentences on what they will not say or do.

Write a specific individual, not a type. Give them a quirk. Avoid stereotypes,
and do not make them a spokesperson for a demographic."""


def generate_persona(description: str, llm: LLM) -> PersonaSpec:
    """
    Ask the model to author a persona from a one-line description.

    :param description: e.g. "a patient with type 2 diabetes who distrusts doctors"
    :param llm:         any LLM implementation (real or scripted)
    :return:            the parsed PersonaSpec — not yet saved to disk
    """
    markdown = llm.complete(
        system=_AUTHOR_SYSTEM,
        messages=[{"role": "user", "content": f"Write a persona for: {description}"}],
        max_tokens=4096,
    )

    spec = PersonaSpec.from_markdown(markdown)

    # The prompt asks for a one-word lowercase role; models sometimes write
    # "Primary Care Doctor". Normalize rather than trust.
    role_words = re.findall(r"[a-z0-9]+", spec.role.lower())
    spec.role = role_words[-1] if role_words else "unknown"

    # If the model ignored the frontmatter format we still want something usable,
    # so fall back to the raw description rather than failing the whole command.
    if spec.name == "Unnamed":
        spec.name = description[:40].strip().title() or "Unnamed"
        spec.summary = description.strip()

    return spec


# -----------------------------------------------------------------------------
# Editing a persona, and undoing it
# -----------------------------------------------------------------------------
#
# Editing is the first operation that destroys data, so it is built around three
# rules: the model's rewrite is validated BEFORE anything is written; the old
# version is copied to `<dir>/.history/` before it is replaced; and name and role
# are locked, because the filename and `find_persona` both key on them.

HISTORY_DIRNAME = ".history"
_MIN_EDITED_BODY_CHARS = 100

_EDIT_SYSTEM = """You revise character specifications for a multi-agent simulation.

You will be given a persona document and a requested change. Return the COMPLETE
updated document, in exactly the same format: the frontmatter block (name, role,
summary) followed by the same markdown sections.

Rules:
- Change only what the request implies. Keep everything else word-for-word.
- Never change the name or the role.
- Update the summary line only if the change makes it inaccurate.
- Keep every section heading.
- Output ONLY the document. No preamble, no commentary, no code fences."""


class PersonaEditError(ValueError):
    """An edit or undo could not be completed. Nothing was changed on disk."""


@dataclass
class EditResult:
    """What an edit did, for the UI and for the model's tool result."""

    spec: PersonaSpec
    path: Path
    backup: Path
    diff: str


def persona_path(spec: PersonaSpec, directory: Path = DEFAULT_PERSONA_DIR) -> Path:
    """The file a loaded persona came from. Raises PersonaEditError if it is gone."""
    for path in _persona_files(directory):
        candidate = _try_load(path)
        if candidate and candidate.name == spec.name and candidate.role == spec.role:
            return path
    raise PersonaEditError(f"could not find the file for {spec.name}")


def _headings(body: str) -> List[str]:
    return [h.strip().lower() for h in re.findall(r"^##\s+(.+)$", body, re.MULTILINE)]


def _validate_edit(old: PersonaSpec, new: PersonaSpec) -> PersonaSpec:
    """
    Accept the model's rewrite or raise. Name and role are forced back to the
    originals rather than rejected: a model that "helpfully" renames someone has
    still done the edit we wanted, but the identity must not move.
    """
    if new.name == "Unnamed":
        raise PersonaEditError("the model did not return a valid persona document; nothing was changed")
    if len(new.body) < _MIN_EDITED_BODY_CHARS:
        raise PersonaEditError("the rewritten persona was too short to be a real edit; nothing was changed")

    missing = [h for h in _headings(old.body) if h not in _headings(new.body)]
    if missing:
        raise PersonaEditError(f"the rewrite dropped sections ({', '.join(missing)}); nothing was changed")

    new.name, new.role = old.name, old.role
    return new


def _diff(old: PersonaSpec, new: PersonaSpec, max_lines: int = 40) -> str:
    lines = list(
        difflib.unified_diff(
            old.to_markdown().splitlines(),
            new.to_markdown().splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
            n=0,
        )
    )
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more diff lines)"]
    return "\n".join(lines)


def edit_persona(
    spec: PersonaSpec,
    change: str,
    llm: LLM,
    directory: Path = DEFAULT_PERSONA_DIR,
) -> EditResult:
    """
    Apply a plain-language change ("make her more skeptical") to a saved persona.

    Order of operations matters: ask the model, parse, validate, back up the old
    file, and only then overwrite. Any failure before the last step leaves the
    persona exactly as it was.
    """
    change = change.strip()
    if not change:
        raise PersonaEditError("tell me what to change about the persona")

    directory = Path(directory)
    path = persona_path(spec, directory)

    raw = llm.complete(
        system=_EDIT_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Persona document:\n\n{path.read_text(encoding='utf-8')}\n\nRequested change: {change}",
            }
        ],
        max_tokens=4096,
    )
    new = _validate_edit(spec, PersonaSpec.from_markdown(raw))

    history = directory / HISTORY_DIRNAME
    history.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = history / f"{path.stem}--{stamp}.md"
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    path.write_text(new.to_markdown(), encoding="utf-8")
    return EditResult(spec=new, path=path, backup=backup, diff=_diff(spec, new))


def undo_edit(spec: PersonaSpec, directory: Path = DEFAULT_PERSONA_DIR) -> Path:
    """
    Restore the most recent backup of a persona, then remove that backup so
    calling this again steps back one more edit.
    """
    directory = Path(directory)
    path = persona_path(spec, directory)

    # Stems share a prefix ("maria" vs "maria-delgado"), so match on the separator.
    backups = sorted((directory / HISTORY_DIRNAME).glob(f"{path.stem}--*.md"))
    if not backups:
        raise PersonaEditError(f"there is nothing to undo for {spec.name}")

    latest = backups[-1]
    path.write_text(latest.read_text(encoding="utf-8"), encoding="utf-8")
    latest.unlink()
    return path


# =============================================================================
# KNOWN LIMITS — status of the weaknesses the starter listed here
# =============================================================================
#
#  1. DONE. `find_persona` prefers an exact name or slug, then matches whole
#     words ("doc" no longer finds "Dorothy"), and raises AmbiguousPersona
#     listing the candidates when several personas match.
#
#  2. DONE. `save_persona` never overwrites: a second "Maria Delgado" is saved
#     as `maria-delgado-2.md`. Overwriting is opt-in with `overwrite=True`.
#
#  3. PARTLY DONE. The parser still handles three fixed keys with no quoting.
#     It now recovers a missing closing fence and a stray fence before the real
#     header, which is a list of failure modes handled, not a robust parser.
#
#  4. PARTLY DONE. `role` is normalized to one lowercase word, and edits are
#     validated (a real persona, not too short, every section heading kept).
#     `generate_persona` still does not check that a new persona's name looks
#     like a name or that its body has the sections the prompt asked for; it falls
#     back to the raw description instead.
#
#  5. DONE. `edit_persona` rewrites a persona in place, backing the old file up
#     to `.history/` first; `undo_edit` restores it. Name and role are locked.
#
#  6. NOT DONE. A persona does not change during a conversation.
#
#  Not addressed: an edit validated only on structure can still gut the content
#  under intact headings, `undo_edit` overwrites a hand edit made after the last
#  model edit, and two simultaneous edits to one file are not guarded.
