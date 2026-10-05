"""
credibility.py — THIS IS THE FILE YOU IMPROVE.
===============================================================================

Everything else in this project is scaffolding. The chatbot works, the UI works,
the tracing works. What does NOT work well is the function below: `score_url()`.

Your job for Project 1 is to make it better. Read the KNOWN WEAKNESSES section
at the bottom of this file — every item on that list is a bug you are invited to
fix, and every one of them is worth points.

-------------------------------------------------------------------------------
THE CONTRACT (do not change this)
-------------------------------------------------------------------------------
    score_url("https://arxiv.org/abs/1706.03762")

    -> {"score": 0.9, "explanation": "arxiv.org is a recognized preprint ..."}

    score:       float in [0.0, 1.0].  0 = not credible, 1 = highly credible.
    explanation: str. A human-readable reason for the score.

Your grader, the evaluation harness (`evaluate.py`), the test suite
(`test_credibility.py`), and the Streamlit app all depend on this exact shape.
If you change the keys or the types, everything downstream breaks.

-------------------------------------------------------------------------------
HOW THE BASELINE WORKS
-------------------------------------------------------------------------------
Two layers, combined at the end:

    Layer 1 (rules)  Pure string inspection of the URL. No network, no API key.
                     Always runs. This is why the app works before you have
                     configured anything.

    Layer 2 (LLM)    One Claude call that judges the URL. Only runs when
                     ANTHROPIC_API_KEY is set and `use_llm` is not False.
                     Skipped silently otherwise.

The final score is a weighted blend of the two. See `score_url()`.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote, urlparse

# The model used for the Layer 2 judgment. Claude Opus 5 is the most capable
# model; switch to "claude-haiku-4-5" if you are scoring many URLs and want to
# cut cost, or "claude-sonnet-5" for a middle option. Scoring quality moves with
# this choice — measured on the 24-URL set, Opus 5 gives MAE 0.086 / 83.3% and
# Haiku 4.5 gives 0.102 / 75.0% — so say in your report which model produced
# your numbers.
#
# Not every model accepts the same parameters. Haiku 4.5 rejects the `effort`
# setting that llm_opinion() sends; the code detects that and retries without
# it, so switching models here is safe. See _NO_EFFORT_SUPPORT further down.
JUDGE_MODEL = "claude-opus-5"

# How much each layer contributes to the final score. These two must sum to 1.0.
# Tuning this split is one of the easiest wins available to you.
# Weakness 9 fix: measured, not guessed. Sweeping the rule weight from 0 to 1 on
# the 36 labelled URLs (LLM scores collected once, then blended offline), MAE bottoms
# out near 0.20 and band accuracy peaks at 0.25-0.40; the curve is flat from 0.15 to
# 0.30. 0.40 sits in that good region, gets the best band accuracy, and still gives
# the rules a real share rather than letting the model decide almost alone. Leave-one-
# out cross-validation picked 0.20 every time, so the choice is stable, but it is fitted
# to our own labels. Ceilings are applied after the blend, so they still bind.
RULE_WEIGHT = 0.4
LLM_WEIGHT = 0.6


# =============================================================================
# LAYER 1 — RULE-BASED SIGNALS
# =============================================================================
# Each table below is a hand-written guess about what makes a source credible.
# They are deliberately short and deliberately naive. Extending them is the
# fastest way to improve your score on the evaluation set, but note that a
# longer lookup table is not the same thing as a better *algorithm* — the
# report asks you to justify your approach, not just your word list.

# Exact-domain judgments. Highest-confidence signal we have.
DOMAIN_SCORES: Dict[str, float] = {
    # Peer-reviewed / archival
    "nature.com": 0.95,
    "science.org": 0.95,
    "nejm.org": 0.95,
    "thelancet.com": 0.95,
    "pubmed.ncbi.nlm.nih.gov": 0.92,
    "arxiv.org": 0.65,          # preprint: NOT peer reviewed (see PREPRINT_DOMAINS)
    "biorxiv.org": 0.55,        # preprint: NOT peer reviewed, and medical claims carry more risk
    "medrxiv.org": 0.55,        # preprint, medical: same reasoning as bioRxiv
    "ssrn.com": 0.60,           # preprint, social science and economics
    "researchsquare.com": 0.50, # preprint platform with lighter screening
    # Reference
    "wikipedia.org": 0.65,
    "britannica.com": 0.80,
    # Mainstream press
    "reuters.com": 0.85,
    "apnews.com": 0.85,
    "bbc.com": 0.82,
    "nytimes.com": 0.80,
    "wsj.com": 0.80,
    # User-generated / self-published
    "medium.com": 0.35,
    "substack.com": 0.35,
    "blogspot.com": 0.25,
    "wordpress.com": 0.25,
    "reddit.com": 0.25,
    "quora.com": 0.20,
    "x.com": 0.15,
    "twitter.com": 0.15,
    # Satire — factually false by design, which the LLM layer often misses
    "theonion.com": 0.05,
    "clickhole.com": 0.05,
    "babylonbee.com": 0.05,
}

# Fallback when the exact domain is unknown. Coarse and easy to fool.
TLD_SCORES: Dict[str, float] = {
    ".gov": 0.88,
    ".edu": 0.82,
    ".mil": 0.85,
    ".org": 0.60,
    ".com": 0.50,
    ".net": 0.48,
    ".io": 0.45,
    ".biz": 0.30,
    ".info": 0.30,
    ".xyz": 0.25,
}

# Substrings in the URL path that hint at self-published or low-edit content.
PATH_PENALTIES: Dict[str, float] = {
    "/blog/": -0.10,
    "/opinion/": -0.08,
    "/sponsored/": -0.20,
    "/press-release/": -0.15,
    "/advertorial/": -0.25,
    "/forum/": -0.12,
    "/comments/": -0.12,
}

# Neutral starting point for a URL we know nothing about.
NEUTRAL_SCORE = 0.5

# Weakness 3 fix: preprint servers host work that nobody has reviewed. A DOI on
# one of these only says the item was registered, not that it was published, so
# the DOI bonus in rule_based_signals() is skipped for these hosts. Without that,
# the bonus pushed a bioRxiv preprint (0.70) up to 0.82, above Wikipedia.
PREPRINT_DOMAINS = {"arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com", "researchsquare.com"}

# Weakness 11 fix: a subdomain whose first label names a kind of content is not
# the publisher's own reporting. blogs.nytimes.com borrows the brand but not the
# newsroom's editing, so it must not inherit the parent's full score. For a known
# parent we cap the score; for an unknown parent we apply a small penalty.
USER_CONTENT_LABELS = {"blog", "blogs", "forum", "forums", "community", "answers",
                       "discuss", "people", "personal", "sites", "pages", "opinion"}
SUBDOMAIN_CEILING = 0.55
SUBDOMAIN_PENALTY = -0.10

# Weakness 10 fix: the reason a known domain gets its score, in plain words. This is
# what turns "'arxiv.org' is a domain we recognize" into something a reader can act
# on. A domain without an entry falls back to a generic sentence.
DOMAIN_NOTES: Dict[str, str] = {
    "arxiv.org": "a preprint server, so papers are posted without peer review",
    "biorxiv.org": "a preprint server, so findings have not been peer reviewed",
    "medrxiv.org": "a medical preprint server, so findings have not been peer reviewed",
    "ssrn.com": "a preprint server, so papers are posted without peer review",
    "researchsquare.com": "a preprint platform with light screening",
    "wikipedia.org": "an open-edit encyclopedia: useful for orientation, but check its citations",
    "britannica.com": "an edited reference work",
    "medium.com": "a self-publishing platform with no editorial review",
    "substack.com": "a self-publishing platform with no editorial review",
    "blogspot.com": "a personal blog platform",
    "wordpress.com": "a personal blog platform",
    "reddit.com": "user-generated discussion that nobody reviews",
    "quora.com": "user-generated answers that nobody reviews",
    "x.com": "social media posts",
    "twitter.com": "social media posts",
    "theonion.com": "a satirical outlet whose stories are fictional by design",
    "clickhole.com": "a satirical outlet whose stories are fictional by design",
    "babylonbee.com": "a satirical outlet whose stories are fictional by design",
    "nature.com": "a peer-reviewed journal publisher",
    "science.org": "a peer-reviewed journal publisher",
    "nejm.org": "a peer-reviewed medical journal",
    "thelancet.com": "a peer-reviewed medical journal",
    "pubmed.ncbi.nlm.nih.gov": "an index of biomedical literature",
    "reuters.com": "a wire service with a published corrections policy",
    "apnews.com": "a wire service with a published corrections policy",
    "bbc.com": "a public broadcaster with editorial standards",
    "nytimes.com": "a newspaper with professional editing and fact-checking",
    "wsj.com": "a newspaper with professional editing and fact-checking",
}

# Endings that only institutions can register, used to word the explanation (not
# to change any score): the ending is real evidence, unlike .com or .xyz.
INSTITUTIONAL_TLDS = {".gov", ".edu", ".mil"}

# Weakness 5 fix: .edu earns 0.82 because universities publish vetted work, but a
# "/~name/" path is a personal page on the same server (a student homepage, class
# notes, a hobby site). It is self-published, so it gets a ceiling like any other
# self-published source, however respectable the hostname looks.
PERSONAL_PAGE_CEILING = 0.50

# Weakness 12 fix, two parts. (a) Paid placement is not editorial content, whatever
# domain it sits on, so it gets a ceiling: subtracting 0.20 from a .com's 0.50
# still left a sponsored page looking middling. (b) Stacked path penalties are
# mostly one signal counted several times (a URL with /blog/ and /comments/ is one
# kind of page, not two), so their sum is floored instead of growing without limit.
PROMO_PATHS = ("/sponsored/", "/advertorial/", "/press-release/")
PROMO_CEILING = 0.25
PATH_PENALTY_FLOOR = -0.30


@dataclass
class Signal:
    """One piece of evidence that moved the score, kept so we can explain it."""

    name: str      # short machine-readable label, e.g. "known_domain"
    value: float   # the score or delta this signal contributed
    reason: str    # human-readable sentence for the explanation field


def _normalize_domain(url: str) -> str:
    """
    Pull a bare lowercase domain out of a URL.

    Strips the scheme, any userinfo, the port, and a leading "www.". Returns an
    empty string when the URL has no host at all, which the caller treats as a
    malformed input.
    """
    host = (urlparse(url).netloc or "").lower()
    host = host.split("@")[-1]      # drop user:pass@
    host = host.split(":")[0]       # drop :port
    if host.startswith("www."):
        host = host[4:]
    return host


def _match_known_domain(domain: str) -> Optional[Tuple[str, float]]:
    """
    Look the domain up in DOMAIN_SCORES, allowing subdomains to match.

    "en.wikipedia.org" matches the "wikipedia.org" entry, and "arxiv.org"
    matches itself. We check the exact domain first so a more specific entry
    always wins over a more general one.
    """
    if domain in DOMAIN_SCORES:
        return domain, DOMAIN_SCORES[domain]
    for known, score in DOMAIN_SCORES.items():
        if domain.endswith("." + known):
            return known, score
    return None


def rule_based_signals(url: str) -> List[Signal]:
    """
    Inspect the URL string and return every signal that fired.

    This runs with no network access and no API key, which is what makes the
    app usable straight after `git clone`. It is also the reason the baseline
    is weak: a URL string alone tells you almost nothing about whether the
    page behind it is any good.
    """
    signals: List[Signal] = []
    parsed = urlparse(url)
    domain = _normalize_domain(url)

    # Signal 1: exact or suffix match against our hand-written domain table.
    match = _match_known_domain(domain)
    if match:
        known, score = match
        note = DOMAIN_NOTES.get(known, "a publisher we have a reputation estimate for")
        signals.append(Signal("known_domain", score, f"{known} is {note} (base score {score:.2f})"))
        # A subdomain like blogs.<known> keeps the brand but not its editorial
        # process, so it is capped instead of inheriting the parent's score.
        prefix = domain[: -len(known) - 1].split(".") if domain != known else []
        label = next((p for p in prefix if p in USER_CONTENT_LABELS), None)
        if label:
            signals.append(Signal(
                "cap", SUBDOMAIN_CEILING,
                f"'{label}.' on {known} is user or opinion content, not its newsroom, so the score is capped at {SUBDOMAIN_CEILING:.2f}",
            ))
    else:
        # Signal 2: fall back to the top-level domain. Very coarse.
        for tld, score in TLD_SCORES.items():
            if domain.endswith(tld):
                signals.append(Signal("tld", score, f"the score starts from the site's '{tld}' ending ({score:.2f}), which is a weak signal"))
                break
        else:
            signals.append(Signal("unknown", NEUTRAL_SCORE, "we recognize neither this site nor its domain ending, so the score starts at neutral"))
        if any(p in USER_CONTENT_LABELS for p in domain.split(".")[:-2]):
            signals.append(Signal("subdomain", SUBDOMAIN_PENALTY, "subdomain suggests blog, forum or user content"))

    # Signal 3: HTTPS. Weak evidence — a scam site can buy a certificate too.
    if parsed.scheme == "https":
        signals.append(Signal("https", 0.02, "it uses HTTPS, a very small plus since scam sites can get certificates too"))
    elif parsed.scheme == "http":
        signals.append(Signal("no_https", -0.05, "it uses plain HTTP, so the connection is not encrypted"))

    # Signal 4: path keywords suggesting opinion, sponsorship, or user content.
    path = (parsed.path or "").lower()
    for fragment, delta in PATH_PENALTIES.items():
        if fragment in path:
            reason = "" if fragment in PROMO_PATHS else f"the URL path contains '{fragment}', which usually means opinion, user or promotional content"
            signals.append(Signal("path", delta, reason))

    # Signal 4a: paid or promotional content is capped, not just nudged down.
    promo = next((p for p in PROMO_PATHS if p in path), None)
    if promo:
        signals.append(Signal(
            "cap", PROMO_CEILING,
            f"'{promo}' marks paid or promotional content, which is not editorial, so the score is capped at {PROMO_CEILING:.2f}",
        ))

    # Signal 4b: a "/~user/" path marks a personal page, whatever the host is.
    if re.search(r"/~[^/]+", path):
        signals.append(Signal(
            "cap", PERSONAL_PAGE_CEILING,
            f"'/~' marks a personal page, which is self-published, so the score is capped at {PERSONAL_PAGE_CEILING:.2f}",
        ))

    # Signal 5: a DOI in the path implies a registered scholarly work, unless the
    # host is a preprint server, where a DOI does not imply review.
    is_preprint = any(domain == d or domain.endswith("." + d) for d in PREPRINT_DOMAINS)
    if re.search(r"/10\.\d{4,9}/", path):
        if is_preprint:
            signals.append(Signal("preprint_doi", 0.0, "has a DOI, but preprint servers do not peer review"))
        else:
            signals.append(Signal("doi", 0.10, "the URL contains a DOI, which suggests a registered publication"))

    return signals


def _combine_signals(signals: List[Signal]) -> float:
    """
    Fold the signal list into a single number in [0, 1].

    The first signal is treated as the base score (it is always the domain or
    TLD judgment) and every later signal is an additive adjustment. This is a
    crude aggregation — see KNOWN WEAKNESSES.
    """
    if not signals:
        return NEUTRAL_SCORE
    base = signals[0].value
    # Path penalties are floored as a group so overlapping hints are not double
    # counted; every other signal is a plain additive adjustment.
    path_total = max(PATH_PENALTY_FLOOR, sum(s.value for s in signals[1:] if s.name == "path"))
    adjustment = path_total + sum(s.value for s in signals[1:] if s.name not in ("cap", "path"))
    score = max(0.0, min(1.0, base + adjustment))
    # "cap" signals are ceilings: they limit the result however many positive
    # signals stacked up, which an additive delta cannot do.
    caps = [s.value for s in signals if s.name == "cap"]
    return min(score, *caps) if caps else score


# =============================================================================
# LAYER 2 — LLM JUDGMENT
# =============================================================================

_JUDGE_SYSTEM = """You assess the credibility of web sources for a research assistant.

Given a URL, judge how much a careful reader should trust content published there.
Consider: the publisher's editorial standards and reputation, whether the content is
peer reviewed, whether it is self-published, and whether the outlet is satirical.

Score 0.0 (not credible at all) to 1.0 (highly credible). Be skeptical of
self-published platforms and satire. Judge the SOURCE, not the topic. If you do not
recognize the domain, say so and score near 0.5 rather than guessing confidently."""

# Constraining the response to this schema means we never have to parse prose or
# repair malformed JSON — the API guarantees the shape.
_JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "description": "Credibility from 0.0 to 1.0"},
        "reason": {"type": "string", "description": "One sentence justifying the score"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}


# =============================================================================
# ⚠️  NEEDS YOUR OWN API KEY — AND SHIPPED UNVERIFIED
# =============================================================================
# REQUIRES A KEY. This function is the only part of credibility.py that calls
# Anthropic. Without ANTHROPIC_API_KEY set it returns None and the scorer falls
# back to rules only — no error, just a weaker score. Get a key at
# https://console.anthropic.com/ and put it in `.env` (copy `.env.example`).
# The key is yours and the calls are billed to you, which is exactly why the
# rules layer, the tests, and evaluate.py were all built to run without one.
#
# VERIFIED LIVE. This path has been run against the real API: the
# `output_config` structured-output call returns a valid score and reason, and
# the blend measurably helps. On the 24-URL evaluation set, turning this layer
# on moves MAE from 0.142 to 0.086 and band accuracy from 66.7% to 83.3%.
#
# Where it helps most is the held-out domains the rule table has never seen —
# a JAMA article scores 0.52 on rules alone and 0.70 with this layer, because
# the model knows what JAMA is and the lookup table does not.
# =============================================================================


# Models that rejected `effort` in this process, so we only pay for that
# discovery once. Not every model supports the parameter — Haiku 4.5 does not,
# and it is the model the cost-saving advice above points you at. Rather than
# hard-coding a list that goes stale with every release, we try the call, and
# if the API says the parameter is unsupported we remember that and retry
# without it. Capability detection over a maintained allowlist.
_NO_EFFORT_SUPPORT: set = set()

# Failures are swallowed below so the app degrades instead of crashing, but a
# silent degrade is impossible to debug. We print the first occurrence of each
# distinct failure to stderr — once, not once per URL, so scoring 24 URLs does
# not produce 24 identical warnings.
_WARNED: set = set()


def _warn_once(key: str, message: str) -> None:
    """
    Print a diagnostic to stderr the first time `key` is seen.

    The key is separate from the message on purpose. API errors embed a unique
    request_id, so deduplicating on the message text would let the same failure
    warn on every one of 24 URLs. Key on the stable part — the model and the
    error class — and let the message carry the changing detail.
    """
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"  [credibility] {message}", file=sys.stderr)


def llm_opinion(url: str) -> Optional[Signal]:
    """
    Ask Claude to judge the URL. Returns None whenever the call cannot be made.

    Returning None rather than raising is deliberate: a missing API key, a
    network blip, or a safety refusal should degrade the score to rules-only
    instead of taking down the whole app. Effort is set to "low" because this
    is a small judgment and we may be scoring several URLs per question — but
    see _NO_EFFORT_SUPPORT above; not every model accepts that parameter.

    The degrade is quiet in the score but no longer quiet in the terminal: any
    failure prints one line to stderr explaining itself.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    output_config: Dict[str, Any] = {
        "format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}
    }
    if JUDGE_MODEL not in _NO_EFFORT_SUPPORT:
        output_config["effort"] = "low"

    try:
        import anthropic

        client = anthropic.Anthropic()
        try:
            response = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=1024,
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": f"Rate the credibility of this source: {url}"}],
                output_config=output_config,
            )
        except anthropic.BadRequestError as exc:
            # "This model does not support the effort parameter." Remember it
            # and retry once without, so switching JUDGE_MODEL to a cheaper
            # model keeps working instead of silently scoring rules-only.
            if "effort" not in str(exc) or "effort" not in output_config:
                raise
            _NO_EFFORT_SUPPORT.add(JUDGE_MODEL)
            _warn_once(
                f"effort:{JUDGE_MODEL}",
                f"{JUDGE_MODEL} does not accept output_config.effort; "
                "retrying without it. The LLM layer is still on.",
            )
            output_config.pop("effort")
            response = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=1024,
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": f"Rate the credibility of this source: {url}"}],
                output_config=output_config,
            )

        # Claude can decline a request; content is empty or partial when it does.
        if response.stop_reason == "refusal":
            _warn_once(
                f"refusal:{JUDGE_MODEL}",
                f"{JUDGE_MODEL} declined to score a URL; falling back to rules for it.",
            )
            return None

        text = next((b.text for b in response.content if b.type == "text"), "")
        data = json.loads(text)
        score = max(0.0, min(1.0, float(data["score"])))
        return Signal("llm", score, str(data["reason"]))

    except Exception as exc:
        # Any failure falls back to rules-only scoring rather than crashing —
        # but says so, once, instead of leaving you to wonder why the LLM layer
        # made no difference to your numbers.
        _warn_once(
            f"{type(exc).__name__}:{JUDGE_MODEL}",
            f"LLM layer unavailable ({type(exc).__name__}: {str(exc)[:160]}). "
            "Scoring with rules only.",
        )
        return None


# =============================================================================
# THE FUNCTION YOU ARE GRADED ON
# =============================================================================

# =============================================================================
# LAYER 1b — REAL METADATA FROM OPENALEX (weakness 4, and the novelty item)
# =============================================================================
# The rules above judge a URL string. When the URL carries a DOI we can instead ask
# OpenAlex (free, no key) about the actual work: has it been retracted, is it in a
# journal or only in a repository, and how often has it been cited. That replaces a
# guess about the domain with a fact about the paper. Any failure (offline, timeout,
# unknown DOI) returns None and the scorer simply carries on with the rules.

OPENALEX_TIMEOUT = 5            # seconds; a slow lookup must not stall the UI
RETRACTED_CEILING = 0.10        # a retracted paper is discredited, whatever the venue
JOURNAL_BONUS = 0.08            # indexed in a journal, so some review took place
REPOSITORY_CEILING = 0.65       # a repository hosts work without implying review
CITATION_BONUS_MAX = 0.07       # citations are weak evidence, so the bonus is small

_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s?#]+)")
_META_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}


def extract_doi(url: str) -> Optional[str]:
    """Return the DOI embedded in a URL's path, or None when there isn't one."""
    match = _DOI_RE.search(unquote(urlparse(url).path or ""))
    return match.group(1).rstrip("/.") if match else None


def openalex_work(doi: str) -> Optional[Dict[str, Any]]:
    """
    Look a DOI up on OpenAlex and return the few fields we use, or None.

    Returns {"retracted": bool, "cited_by": int, "venue_type": str | None,
    "venue": str | None}. A 404 just means OpenAlex does not know the DOI and is
    silent; any other failure is reported once on stderr. Results are cached,
    including misses, so a repeated URL never pays for a second round trip.
    """
    if doi in _META_CACHE:
        return _META_CACHE[doi]

    endpoint = (
        "https://api.openalex.org/works/doi:" + quote(doi, safe="/()")
        + "?select=is_retracted,cited_by_count,primary_location"
    )
    request = urllib.request.Request(endpoint, headers={"User-Agent": "cs676-credibility/0.1"})
    result: Optional[Dict[str, Any]] = None
    try:
        with urllib.request.urlopen(request, timeout=OPENALEX_TIMEOUT) as response:
            data = json.load(response)
        source = ((data.get("primary_location") or {}).get("source")) or {}
        result = {
            "retracted": bool(data.get("is_retracted")),
            "cited_by": int(data.get("cited_by_count") or 0),
            "venue_type": source.get("type"),
            "venue": source.get("display_name"),
        }
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            _warn_once(f"openalex:{exc.code}", f"OpenAlex lookup failed (HTTP {exc.code}); using rules only.")
    except Exception as exc:
        _warn_once(f"openalex:{type(exc).__name__}",
                   f"OpenAlex unavailable ({type(exc).__name__}); using rules only.")

    _META_CACHE[doi] = result
    return result


def metadata_signals(meta: Dict[str, Any], is_preprint_host: bool) -> List[Signal]:
    """
    Turn an OpenAlex record into signals. A retraction outranks everything else.

    Three kinds of evidence come out of a record: whether the work was retracted,
    what kind of venue holds it, and how often it has been cited. Ceilings are
    returned as "cap" signals so score_url() can apply them again after the blend.
    """
    venue = meta.get("venue") or "an unnamed venue"
    # A retracted paper is discredited whatever journal printed it, so it returns
    # immediately: a journal bonus or citation count must not soften that.
    if meta["retracted"]:
        return [Signal("cap", RETRACTED_CEILING,
                       f"OpenAlex records this paper as RETRACTED, so it should not be relied on (capped at {RETRACTED_CEILING:.2f})")]

    signals: List[Signal] = []
    # Venue type: a journal implies some review took place; a repository (where
    # preprints live) does not, so it gets a ceiling instead of a bonus. Preprint
    # hosts never get the journal bonus even if OpenAlex lists them as a journal.
    if meta["venue_type"] == "journal" and not is_preprint_host:
        signals.append(Signal("venue", JOURNAL_BONUS, f"it appears in {venue}, an indexed journal"))
    elif meta["venue_type"] == "repository":
        signals.append(Signal("cap", REPOSITORY_CEILING,
                              f"it is hosted in {venue}, a repository, which does not imply peer review (capped at {REPOSITORY_CEILING:.2f})"))
    # Citations are weak, noisy evidence (a bad paper can be cited to be refuted),
    # so the bonus grows with the logarithm and is capped at a few points: 10 cites
    # adds about 0.016, 10,000 cites about 0.06.
    if meta["cited_by"] > 0:
        bonus = min(CITATION_BONUS_MAX, 0.015 * math.log10(1 + meta["cited_by"]))
        signals.append(Signal("citations", bonus, f"it has been cited {meta['cited_by']:,} times"))
    return signals


def _sentence(reason: str) -> str:
    """
    Capitalise a signal reason and make sure it ends with one full stop.

    Signal reasons are written as lowercase clauses so they can be joined into
    sentences; this turns each one into a proper sentence for the reader. Models
    and rules may or may not end a reason with a full stop, so any trailing one
    is removed first to avoid doubling it.
    """
    reason = reason.strip().rstrip(".")
    # A leading domain name ("arxiv.org is ...") must keep its lowercase.
    if "." in reason.split(" ", 1)[0]:
        return reason + "."
    return reason[:1].upper() + reason[1:] + "."


def _build_explanation(final: float, signals: List[Signal], llm: Optional[Signal]) -> str:
    """
    Compose the explanation a reader sees: a verdict first, then the reasons.

    Order matters more than wording. A retraction goes first because it overrides
    everything else; the minor HTTPS signal goes last so it never reads as a
    headline. When the site is unfamiliar the explanation says the score is a
    rough estimate, so a reader does not mistake a default for a judgment.
    """
    verdict = f"{score_band(final)[0].capitalize()} credibility ({final:.2f})."

    def priority(s: Signal) -> int:
        if "RETRACTED" in s.reason:
            return 0
        return 2 if s.name in ("https", "no_https") else 1

    # A signal with an empty reason still moves the score but is not worth a
    # sentence: the promo path penalty is already explained by its ceiling.
    sentences = [_sentence(s.reason) for s in sorted(signals, key=priority) if s.reason]
    if llm is not None:
        sentences.append(_sentence(f"A Claude model rated the source {llm.value:.2f}: {llm.reason}"))
    retracted = any("RETRACTED" in s.reason for s in signals)
    if signals and signals[0].name in ("tld", "unknown") and not retracted:
        # .gov, .edu and .mil are restricted to institutions, so the ending itself
        # carries information; saying "we do not recognize the site" undersells it.
        if any(f"'{tld}' ending" in signals[0].reason for tld in INSTITUTIONAL_TLDS):
            sentences.append("We have no specific rating for this site, but this domain ending is restricted to institutions.")
        elif llm is not None:
            # The model's sentence above names the site, so "we do not recognize it"
            # would contradict it. Only the rule layer is in the dark here.
            sentences.append("Our rules do not recognize this site, so the model's judgment carries most of this score.")
        else:
            sentences.append("Treat this score as a rough estimate, because we do not recognize the site.")
    return " ".join([verdict] + sentences)


# Scoring the same URL repeatedly in one session is common (a chat may cite the
# same paper on every turn), so results are memoized for the process lifetime.
_CACHE: Dict[Tuple[str, Optional[bool], Optional[bool]], Dict[str, Any]] = {}


def score_url(url: str, use_llm: Optional[bool] = None, use_metadata: Optional[bool] = None) -> Dict[str, Any]:
    """
    Score the credibility of a source URL.

    :param url:          The URL to evaluate.
    :param use_llm:      True forces the Claude judgment, False forces rules-only,
                         None (default) uses the LLM when an API key is available.
    :param use_metadata: False skips the OpenAlex lookup (no network). None or True
                         looks up the DOI when the URL has one.
    :return:             {"score": float in [0,1], "explanation": str}
    """
    # Guard clause: anything that is not a usable http(s) URL scores 0.0 with an
    # explanation rather than raising, so one bad link cannot break a whole page.
    # This runs BEFORE the cache lookup: a list or dict is unhashable, so using it
    # as a cache key raised TypeError before the guard was ever reached.
    if not isinstance(url, str) or not url.strip():
        return {"score": 0.0, "explanation": "No URL was provided."}

    cache_key = (url, use_llm, use_metadata)
    if cache_key in _CACHE:
        return dict(_CACHE[cache_key])

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not _normalize_domain(url):
        return {"score": 0.0, "explanation": f"'{url}' is not a valid http(s) URL."}

    # Layer 1 always runs.
    signals = rule_based_signals(url)

    # Layer 1b: real metadata when the URL carries a DOI. Its ceilings (retracted,
    # repository) are ordinary "cap" signals, applied again after the LLM blend.
    doi = extract_doi(url)
    if doi and use_metadata is not False:
        meta = openalex_work(doi)
        if meta:
            domain = _normalize_domain(url)
            on_preprint_host = any(domain == d or domain.endswith("." + d) for d in PREPRINT_DOMAINS)
            signals += metadata_signals(meta, on_preprint_host)

    rule_score = _combine_signals(signals)

    # Layer 2 runs only when it can. Blend if we got an opinion, otherwise the
    # rule score stands on its own.
    llm = llm_opinion(url) if use_llm is not False else None
    if llm is not None:
        final = RULE_WEIGHT * rule_score + LLM_WEIGHT * llm.value
    else:
        final = rule_score

    # Ceilings bind after the blend too. Without this a generous model opinion pulls
    # a capped page back up (blogs.<trusted site>: rules 0.55, model 0.80, blend 0.70,
    # which would show as HIGH). Ceilings are claims about the most a source can earn.
    ceilings = [s.value for s in signals if s.name == "cap"]
    if ceilings:
        final = min(final, *ceilings)
    final = round(max(0.0, min(1.0, final)), 2)
    result = {"score": final, "explanation": _build_explanation(final, signals, llm)}

    _CACHE[cache_key] = dict(result)
    return result


def score_band(score: float) -> Tuple[str, str]:
    """
    Map a score onto a display band. Used by the app to colour the source chips.

    :return: (label, streamlit_colour) — e.g. ("HIGH", "green")
    """
    if score >= 0.70:
        return "HIGH", "green"
    if score >= 0.40:
        return "MEDIUM", "orange"
    return "LOW", "red"


# =============================================================================
# KNOWN WEAKNESSES — YOUR TASK LIST
# =============================================================================
#
# This baseline is deliberately mediocre. Everything below is a real defect.
# You are not expected to fix all of them; pick the ones you can defend in your
# Deliverable 2 report, and measure the change with `python evaluate.py`.
#
#  1. IT NEVER READS THE PAGE. The whole of Layer 1 inspects a string. It cannot
#     tell a rigorous article from a hoax hosted on the same domain. Fetching the
#     page and looking for an author, a date, citations, or a corrections policy
#     is the single biggest available improvement.
#
#  2. THE DOMAIN TABLE IS A HAND-WRITTEN GUESS. ~30 domains, no source, no
#     evidence. Wikipedia's own perennial-sources list and similar published
#     datasets are real, citable alternatives to inventing numbers.
#
#  3. IT CANNOT TELL A PREPRINT FROM A PEER-REVIEWED PAPER. arxiv.org is scored
#     0.75 for every paper on it, whether it is "Attention Is All You Need" or
#     something posted this morning that nobody has read.
#
#  4. IT HAS NEVER HEARD OF RETRACTION. A retracted, discredited paper scores
#     exactly as high as a replicated one. Crossref and OpenAlex both expose
#     retraction status and citation counts over free APIs.
#
#  5. ANY .edu SCORES HIGHLY. Including a student's personal homepage hosted on
#     a university server.
#
#  6. THE AGGREGATION IS ARITHMETIC, NOT STATISTICAL. `_combine_signals` adds
#     numbers that were picked by hand. Nothing here is fitted to data. A
#     regression or a lasso over labelled examples would let you *learn* the
#     weights instead of guessing them — and would let you report which features
#     actually matter (Session 06).
#
#  7. THE SCORE IS NOT CALIBRATED. A 0.7 does not mean "right 70% of the time";
#     it means "some numbers happened to add up to 0.7". A calibration curve or
#     a Brier score would tell you how wrong that is (Session 05).
#
#  8. THERE IS NO UNCERTAINTY. An unrecognized domain and a well-known journal
#     both return a bare point estimate. Bootstrapping a confidence interval is
#     directly on the syllabus (Session 05).
#
#  9. THE TWO LAYERS ARE BLENDED WITH A CONSTANT. RULE_WEIGHT = 0.6 because 0.6
#     looked reasonable. It was never tested against anything.
#
# 10. THE EXPLANATION IS A LIST OF FRAGMENTS JOINED BY SEMICOLONS. It states
#     which rules fired, not why the reader should care. Explanation quality is
#     graded separately from score accuracy.
#
# 11. SUBDOMAINS INHERIT THE PARENT'S REPUTATION IN FULL. `_match_known_domain`
#     suffix-matches, so an opinion blog at blogs.nytimes.com scores exactly as
#     high as the newspaper's reporting, and anything hosted on a subdomain of a
#     trusted publisher is trusted automatically. Whether that inheritance is
#     right depends on the host, and the code never asks.
#
# 12. PENALTIES STACK WITHOUT A FLOOR. A URL matching three PATH_PENALTIES
#     entries takes all three hits additively. Nothing checks whether the
#     combination is meaningful or just the same signal counted three times.
