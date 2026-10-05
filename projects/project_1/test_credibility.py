"""
test_credibility.py — contract tests for score_url().

    python test_credibility.py

These check the SHAPE of your output, not its quality. They must keep passing
however much you rewrite the internals — the app, the grader, and evaluate.py
all rely on this contract. Use `evaluate.py` to measure quality.

Deliverable 1 asks for "initial testing to validate input/output handling".
This file is that, and adding your own cases here is part of the deliverable.

No pytest required, deliberately — one less thing to install.
"""

from credibility import score_band, score_url

PASSED = 0
FAILED = 0


def check(condition: bool, description: str) -> None:
    """Assert-with-a-label so one failure doesn't stop the whole run."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {description}")
    else:
        FAILED += 1
        print(f"  FAIL  {description}")


print("\nContract: return shape")
result = score_url("https://www.nature.com/articles/example", use_llm=False)
check(isinstance(result, dict), "returns a dict")
check(set(result.keys()) == {"score", "explanation"}, "has exactly the keys 'score' and 'explanation'")
check(isinstance(result["score"], float), "score is a float")
check(isinstance(result["explanation"], str), "explanation is a str")
check(len(result["explanation"]) > 0, "explanation is not empty")

print("\nContract: score range")
for url in [
    "https://www.nature.com/x",
    "https://medium.com/@a/b",
    "http://unknown-site.xyz/page",
    "https://en.wikipedia.org/wiki/X",
]:
    score = score_url(url, use_llm=False)["score"]
    check(0.0 <= score <= 1.0, f"{url[:40]:<42} -> {score:.2f} is within [0, 1]")

print("\nContract: malformed input is handled, not raised")
for bad in ["", "   ", "not a url", "ftp://files.example.com/x", "javascript:alert(1)", "//example.com"]:
    try:
        bad_result = score_url(bad, use_llm=False)
        ok = isinstance(bad_result, dict) and 0.0 <= bad_result["score"] <= 1.0
        check(ok, f"{bad!r:<28} -> {bad_result['score']:.2f} (no exception)")
    except Exception as e:
        check(False, f"{bad!r:<28} raised {type(e).__name__}")

print("\nContract: determinism")
a = score_url("https://arxiv.org/abs/1706.03762", use_llm=False)
b = score_url("https://arxiv.org/abs/1706.03762", use_llm=False)
check(a == b, "same URL scored twice gives the same result")

print("\nSanity: ordering the baseline should already get right")
journal = score_url("https://www.nature.com/articles/x", use_llm=False)["score"]
blog = score_url("https://randomblog.blogspot.com/x", use_llm=False)["score"]
check(journal > blog, f"a journal ({journal:.2f}) outranks a personal blog ({blog:.2f})")

gov = score_url("https://www.census.gov/data", use_llm=False)["score"]
throwaway = score_url("http://whatever.xyz/page", use_llm=False)["score"]
check(gov > throwaway, f"a .gov source ({gov:.2f}) outranks a throwaway domain ({throwaway:.2f})")

print("\nContract: score_band")
check(score_band(0.9)[0] == "HIGH", "0.90 -> HIGH")
check(score_band(0.5)[0] == "MEDIUM", "0.50 -> MEDIUM")
check(score_band(0.1)[0] == "LOW", "0.10 -> LOW")

# =============================================================================
# OUR OWN TESTS — one group per fix in credibility.py. All of them run offline:
# use_llm=False everywhere, and the OpenAlex lookup is either switched off or fed
# a hand-made record through the module's cache, so no test touches the network.
# =============================================================================
import credibility as cred


def rules(url: str) -> float:
    """Rules-only, no network: the score the baseline logic alone would give."""
    return score_url(url, use_llm=False, use_metadata=False)["score"]


print("\nRobustness: non-string and odd inputs return a valid result instead of raising")
for odd in [None, 123, 4.5, ["https://a.com"], {"u": 1}, b"https://a.com", "http://[::1]/",
            "https://example.com/" + "x" * 100000, "https://example.com/\x00", "file:///etc/passwd"]:
    try:
        r = score_url(odd, use_llm=False, use_metadata=False)
        ok = isinstance(r, dict) and set(r) == {"score", "explanation"} and 0.0 <= r["score"] <= 1.0
        check(ok, f"{repr(odd)[:34]:<36} -> {r['score']:.2f} (valid shape, no exception)")
    except Exception as e:
        check(False, f"{repr(odd)[:34]:<36} raised {type(e).__name__}")
check(rules("https://nytimes.com.evil.com/") == rules("https://unknown-site.com/"),
      "a lookalike (nytimes.com.evil.com) is scored as an unknown site, not as the NYT")
check(rules("https://evilnytimes.com/") == rules("https://unknown-site.com/"),
      "a lookalike (evilnytimes.com) does not inherit the NYT score")

print("\nFix 3: preprints are not treated as peer reviewed")
check(rules("https://arxiv.org/abs/1706.03762") < rules("https://www.nature.com/articles/x"),
      "an arXiv paper scores below a Nature article")
check(rules("https://www.biorxiv.org/content/10.1101/2020.01.01.000001v1")
      == rules("https://www.biorxiv.org/content/abc"),
      "a DOI adds nothing on a preprint server (bioRxiv with and without DOI match)")
check(rules("https://www.medrxiv.org/content/10.1101/2021.01.01.21249000v1") < 0.60,
      "medRxiv does not fall back to the generic .org default")

print("\nFix 11: user-content subdomains do not inherit the parent's score")
check(rules("https://blogs.nytimes.com/2014/post/") < rules("https://www.nytimes.com/2014/story/"),
      "blogs.nytimes.com scores below the newsroom")
check(rules("https://blogs.nytimes.com/2014/post/") <= cred.SUBDOMAIN_CEILING,
      f"blogs.nytimes.com is at or under the {cred.SUBDOMAIN_CEILING:.2f} ceiling")
check(rules("https://forums.bbc.com/thread/1") <= cred.SUBDOMAIN_CEILING,
      "a forum subdomain of a trusted site is capped too")
check(rules("https://en.wikipedia.org/wiki/X") == rules("https://wikipedia.org/wiki/X"),
      "a language subdomain (en.wikipedia.org) is NOT penalised")
check(rules("https://blogs.example.org/post") < rules("https://www.example.org/post"),
      "an unknown domain's blog subdomain scores below its main site")

print("\nFix 5: personal pages on .edu are capped")
check(rules("https://web.mit.edu/~jdoe/notes.html") <= cred.PERSONAL_PAGE_CEILING,
      f"a /~user/ page on .edu is at or under {cred.PERSONAL_PAGE_CEILING:.2f}")
check(rules("https://news.mit.edu/2024/story") > rules("https://web.mit.edu/~jdoe/notes.html"),
      "a university news page outranks a student homepage")
check(rules("https://news.mit.edu/2024/story") >= 0.80,
      "an ordinary .edu page is not penalised by the new rule")

print("\nFix 12: promotional content is capped; stacked penalties are floored")
check(rules("https://www.nytimes.com/sponsored/brand-story/x") <= cred.PROMO_CEILING,
      "sponsored content on a trusted site is capped at the promo ceiling")
check(rules("https://example.com/press-release/x") <= cred.PROMO_CEILING,
      "a press release is capped too")
three = rules("https://example.com/blog/forum/comments/x")
none = rules("https://example.com/page")
check(abs((none - three) - 0.30) < 0.011,
      f"three stacked path penalties cost exactly the floor (0.30), not the 0.34 sum: lost {none - three:.2f}")
check(rules("https://www.nytimes.com/2024/story") > rules("https://www.nytimes.com/sponsored/x"),
      "real reporting outranks sponsored content on the same domain")

print("\nFix 4: OpenAlex metadata (fed from a fake record, no network)")
DOI_URL = "https://example.org/paper/10.9999/test.1"
cred._META_CACHE["10.9999/test.1"] = {"retracted": True, "cited_by": 3000, "venue_type": "journal", "venue": "Test Journal"}
retracted = score_url(DOI_URL, use_llm=False)
check(retracted["score"] <= cred.RETRACTED_CEILING, f"a retracted paper is capped at {cred.RETRACTED_CEILING:.2f}")
check("RETRACTED" in retracted["explanation"], "the explanation says the paper was retracted")
check(score_url(DOI_URL, use_llm=False, use_metadata=False)["score"] > retracted["score"],
      "the same URL with metadata switched off is not capped (the lookup is what lowers it)")

cred._META_CACHE["10.9999/test.2"] = {"retracted": False, "cited_by": 5000, "venue_type": "journal", "venue": "Test Journal"}
journal_url = "https://example.org/paper/10.9999/test.2"
check(score_url(journal_url, use_llm=False)["score"] > rules(journal_url),
      "an indexed, well-cited journal article scores above the rules-only value")

cred._META_CACHE["10.9999/test.3"] = {"retracted": False, "cited_by": 0, "venue_type": "repository", "venue": "Some Repository"}
repo_url = "https://example.org/paper/10.9999/test.3"
check(score_url(repo_url, use_llm=False)["score"] <= cred.REPOSITORY_CEILING,
      f"a repository-hosted work is capped at {cred.REPOSITORY_CEILING:.2f}")

cred._META_CACHE["10.9999/test.4"] = None   # what a 404 or an outage leaves behind
miss_url = "https://example.org/paper/10.9999/test.4"
check(score_url(miss_url, use_llm=False)["score"] == rules(miss_url),
      "an unknown DOI or failed lookup falls back to the rules without raising")

check(cred.extract_doi("https://doi.org/10.1038/nature14539") == "10.1038/nature14539", "extract_doi reads a DOI from a path")
check(cred.extract_doi("https://www.nejm.org/doi/full/10.1056/NEJMoa2034577") == "10.1056/NEJMoa2034577",
      "extract_doi finds a DOI after other path segments")
check(cred.extract_doi("https://www.nytimes.com/2024/story") is None, "extract_doi returns None when there is no DOI")
check(set(retracted.keys()) == {"score", "explanation"}, "the contract still holds when metadata is used")

print("\nFix 10: explanations")
for url in ["https://arxiv.org/abs/1706.03762", "http://unknown-site.xyz/x", "https://www.nature.com/articles/x"]:
    text = score_url(url, use_llm=False, use_metadata=False)["explanation"]
    check(text.split()[0] in ("High", "Medium", "Low") and "credibility (" in text,
          f"{url[:38]:<40} opens with a verdict")
check("peer review" in score_url("https://arxiv.org/abs/1706.03762", use_llm=False)["explanation"],
      "the arXiv explanation says WHY: no peer review")
check("rough estimate" in score_url("http://unknown-site.xyz/x", use_llm=False)["explanation"],
      "an unrecognised site is flagged as a rough estimate")
check("rough estimate" not in score_url("https://www.nature.com/articles/x", use_llm=False, use_metadata=False)["explanation"],
      "a recognised site is not flagged as uncertain")
for url in ["https://www.cdc.gov/flu/about/index.html", "https://web.mit.edu/~jdoe/notes.html"]:
    check("do not recognize" not in score_url(url, use_llm=False, use_metadata=False)["explanation"],
          f"{url[:38]:<40} is not described as an unrecognised site")
check("restricted to institutions" in score_url("https://www.cdc.gov/flu/about/index.html", use_llm=False)["explanation"],
      "an institutional ending is explained as such")
check(";" not in score_url("https://arxiv.org/abs/1706.03762", use_llm=False)["explanation"],
      "the explanation is sentences, not semicolon-joined fragments")

print("\nFix 9: blend weights, and ceilings that still bind after the blend (LLM faked, no network)")
check(abs(cred.RULE_WEIGHT + cred.LLM_WEIGHT - 1.0) < 1e-9, "the two layer weights sum to 1.0")
check(cred.RULE_WEIGHT == 0.4 and cred.LLM_WEIGHT == 0.6, "weights are the measured 0.40 rules / 0.60 LLM")

_real_llm = cred.llm_opinion


def fake_llm(score: float):
    """Make llm_opinion return a fixed score, so the blend can be tested exactly."""
    return lambda url: cred.Signal("llm", score, "faked for the test")


try:
    cred.llm_opinion = fake_llm(0.80)

    plain_url = "https://www.nytimes.com/2024/01/01/world/story.html"
    rule_only = score_url(plain_url, use_llm=False, use_metadata=False)["score"]
    blended = score_url(plain_url, use_llm=True, use_metadata=False)["score"]
    check(abs(blended - round(0.4 * rule_only + 0.6 * 0.80, 2)) < 0.011,
          f"an uncapped URL blends as 0.4*rules + 0.6*LLM ({rule_only:.2f}, 0.80 -> {blended:.2f})")

    blog = score_url("https://blogs.nytimes.com/2014/01/01/some-post/", use_llm=True, use_metadata=False)["score"]
    check(blog <= cred.SUBDOMAIN_CEILING,
          f"a generous model (0.80) cannot lift blogs.nytimes.com above its ceiling: {blog:.2f}")
    check(score_band(blog)[0] == "MEDIUM", "so the chip stays MEDIUM, not HIGH")

    promo = score_url("https://www.nytimes.com/sponsored/brand-story/x", use_llm=True, use_metadata=False)["score"]
    check(promo <= cred.PROMO_CEILING, f"nor can it lift sponsored content: {promo:.2f}")

    cred._META_CACHE["10.9999/blend.1"] = {"retracted": True, "cited_by": 10, "venue_type": "journal", "venue": "Test Journal"}
    check(score_url("https://example.org/p/10.9999/blend.1", use_llm=True)["score"] <= cred.RETRACTED_CEILING,
          "a retracted paper stays capped even when the model rates it highly")

    cred.llm_opinion = fake_llm(0.90)
    unknown_with_llm = score_url("https://some-unknown-nonprofit.org/page", use_llm=True, use_metadata=False)["explanation"]
    check("do not recognize the site" not in unknown_with_llm and "model's judgment carries" in unknown_with_llm,
          "with the model on, an unknown site is not called 'unrecognised' right after the model names it")

    cred.llm_opinion = fake_llm(0.10)
    low_blog = score_url("https://blogs.nytimes.com/2014/other-post/", use_llm=True, use_metadata=False)["score"]
    check(low_blog < cred.SUBDOMAIN_CEILING, f"a ceiling never RAISES a score: a low model opinion still wins ({low_blog:.2f})")
finally:
    cred.llm_opinion = _real_llm

print(f"\n{'=' * 60}")
print(f"  {PASSED} passed, {FAILED} failed")
print(f"{'=' * 60}\n")
raise SystemExit(1 if FAILED else 0)
