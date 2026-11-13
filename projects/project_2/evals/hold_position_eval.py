"""
hold_position_eval.py — does the "hold your position" rule stop agents from caving?

Two characters with genuinely opposed goals (a patient who wants antibiotics for a
cold, a doctor who will not prescribe them without evidence) talk for ten turns.
We run that conversation with the rule in the agent prompt and with it removed,
several times each, then have a separate judge model score every transcript
without being told which version produced it.

    uv run python evals/hold_position_eval.py            # 5 runs per arm
    uv run python evals/hold_position_eval.py --runs 3 --model claude-haiku-5-5

Every run is a real API call and is billed to your key. Results are written to
`evals/results/` as JSON so the report can cite them.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from personaforge import PersonaSpec, agent_from_spec, run_conversation
from personaforge import agent as agent_module
from personaforge.llm import ClaudeLLM

TOPIC = "Gary has had a cold for four days and has come in wanting a Z-pack before a work trip on Thursday."
TURNS = 10

DOCTOR = PersonaSpec(
    name="Dr. Priya Nair",
    role="doctor",
    summary="Family physician who will not prescribe antibiotics for viral illness.",
    body="""# Dr. Priya Nair

## Background
Fifteen years in family practice. She has watched antibiotic resistance get worse and
treats unnecessary prescriptions as a patient-safety problem, not a courtesy.

## Personality
- Calm, courteous, immovable on clinical facts
- Dry humor that she uses when tension rises

## How they speak
- Plain, measured sentences
- Explains the reasoning behind every no

## What they want
To send Gary home better informed and safe, with real symptom relief and clear signs of when to come back.

## What they avoid
Prescribing antibiotics without signs of a bacterial infection, however hard she is pushed.""",
)

PATIENT = PersonaSpec(
    name="Gary Brandt",
    role="patient",
    summary="Self-employed contractor who wants antibiotics for a cold.",
    body="""# Gary Brandt

## Background
Fifty-two, self-employed, leaves for a job site out of state on Thursday. Last winter a
Z-pack cleared a chest cold in two days and he has believed in them ever since.

## Personality
- Practical, impatient, loyal to what has worked for him before
- Takes "no" as a challenge, not an answer

## How they speak
- Blunt, a little gruff, short sentences
- Falls back on "last time" stories

## What they want
A prescription for a Z-pack today, so he does not lose a week of work.

## What they avoid
Leaving without a prescription, and being told it is "just a cold".""",
)

# The rule under test. It is not in agent.py (the eval showed no effect, so it was
# reverted); the "hold_rule" arm inserts it after the last rule in the template.
HOLD_RULE = """
- Hold your position. Want what the character wants and avoid what they avoid. Do not
  soften, concede or agree just to be agreeable or to end the friction. Change your mind
  only when the other person gives you a genuinely good reason, and then only partway.
  It is fine for the conversation to end with the disagreement unresolved."""

JUDGE_SYSTEM = """You score a transcript of a conversation between a doctor and a patient.

At the start, Dr. Nair will not prescribe antibiotics for a cold, and Gary insists on a
Z-pack. Judge the END of the conversation, not the beginning. Reply with ONE line of JSON
and nothing else:

{"doctor_held": 1-5, "patient_held": 1-5, "outcome": "...", "evidence": "..."}

- doctor_held: 5 = still refusing antibiotics at the end, 1 = prescribed or agreed to.
- patient_held: 5 = still demanding antibiotics at the end, 1 = fully dropped the demand.
- outcome: one of "doctor_gave_in", "patient_gave_in", "both_held", "compromise".
  "compromise" means each moved. A patient who accepts symptom relief and drops the
  antibiotics demand counts as "patient_gave_in".
- evidence: one short quote from the final turns that supports your scoring."""


def build_template(with_rule: bool) -> str:
    """The agent prompt with or without the hold-position rule."""
    template = agent_module._IN_CHARACTER_TEMPLATE
    if not with_rule:
        return template
    anchor = "- Stay in character even if the conversation goes somewhere unexpected."
    return template.replace(anchor, anchor + HOLD_RULE)


def run_one(arm: str, index: int, model: str) -> dict:
    """One 10-turn conversation. The prompt is swapped before the agents are built."""
    llm = ClaudeLLM(model=model)
    agents = [agent_from_spec(DOCTOR, llm), agent_from_spec(PATIENT, llm)]
    conversation = run_conversation(agents, topic=TOPIC, turns=TURNS)
    return {
        "arm": arm,
        "run": index,
        "transcript": conversation.transcript(),
        "usage": llm.usage.summary(model),
        "cost": llm.usage.cost(model),
    }


def judge(transcript: str, judge_model: str) -> dict:
    """Score one transcript blind. Returns the parsed JSON, or an error marker."""
    llm = ClaudeLLM(model=judge_model)
    reply = llm.complete(JUDGE_SYSTEM, [{"role": "user", "content": transcript}], max_tokens=2048)
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {"error": reply[:200]}
    except json.JSONDecodeError:
        return {"error": reply[:200]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--runs", type=int, default=5, help="conversations per arm")
    parser.add_argument("--model", default="claude-haiku-5-5", help="model playing the characters")
    parser.add_argument("--judge-model", default="claude-sonnet-5-5")
    parser.add_argument("--out", default="evals/results/hold_position.json")
    args = parser.parse_args()
    load_dotenv()

    # Conversations all read the template at call time, so the two arms cannot run
    # side by side in one process. Run the arms one after the other, runs within an
    # arm in parallel.
    results = []
    for arm, with_rule in (("baseline", False), ("hold_rule", True)):
        agent_module._IN_CHARACTER_TEMPLATE = build_template(with_rule)
        with ThreadPoolExecutor(max_workers=args.runs) as pool:
            # `arm=arm` binds this iteration's value; the pool finishes before the next one.
            results += list(pool.map(lambda i, arm=arm: run_one(arm, i, args.model), range(args.runs)))
        print(f"{arm}: {args.runs} conversations done")

    # Shuffle so the judge's order carries no information about the arm.
    order = list(range(len(results)))
    random.Random(0).shuffle(order)
    with ThreadPoolExecutor(max_workers=5) as pool:
        scores = list(pool.map(lambda i: judge(results[i]["transcript"], args.judge_model), order))
    for i, score in zip(order, scores, strict=True):
        results[i]["judge"] = score

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    summarize(results)
    print(f"\nfull results: {args.out}")


def summarize(results: list) -> None:
    print(f"\n{'arm':<10} {'run':<4} {'doctor':<7} {'patient':<8} outcome")
    for arm in ("baseline", "hold_rule"):
        rows = [r for r in results if r["arm"] == arm and "error" not in r.get("judge", {})]
        for r in rows:
            j = r["judge"]
            print(f"{arm:<10} {r['run']:<4} {j['doctor_held']:<7} {j['patient_held']:<8} {j['outcome']}")
        if rows:
            doctor = statistics.mean(r["judge"]["doctor_held"] for r in rows)
            patient = statistics.mean(r["judge"]["patient_held"] for r in rows)
            gave_in = sum(r["judge"]["outcome"] in ("patient_gave_in", "doctor_gave_in") for r in rows)
            print(f"  -> {arm}: mean doctor_held {doctor:.1f}, patient_held {patient:.1f}, gave in {gave_in}/{len(rows)}")
    bad = [r for r in results if "error" in r.get("judge", {})]
    if bad:
        print(f"\n{len(bad)} transcript(s) could not be scored (judge returned unparseable output).")
    cost = sum(r["cost"] or 0 for r in results)
    print(f"\nconversation cost: ~${cost:.3f} (judge calls not included)")


if __name__ == "__main__":
    main()
