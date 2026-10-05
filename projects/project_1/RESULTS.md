# Project 1 results: credibility scorer, baseline vs improved

Everything below was measured with `evaluate.py` on **37 labelled URLs**: the 24 provided
plus 13 of our own. The LLM judge model was `claude-opus-5` (`JUDGE_MODEL` in
`credibility.py`). This folder holds the improved scorer; the original starter code is kept
unchanged in a separate folder, so the two can be compared.

## Headline

| Configuration | Original scorer | Improved scorer |
|---|---|---|
| Rules only, no network | MAE 0.184 / bands 54.1% / worst 0.67 | MAE 0.151 / 67.6% / 0.67 |
| Rules + OpenAlex metadata | n/a | MAE 0.126 / 73.0% / 0.44 |
| Rules + LLM | MAE 0.117 / 81.1% / 0.54 | MAE 0.072 / 89.2% / 0.42 |
| **Rules + LLM + OpenAlex** | n/a | **MAE 0.060 / 94.6% / 0.15** |

MAE is mean absolute error against our labels (lower is better). Bands is the share of URLs
whose HIGH/MEDIUM/LOW chip is right. Worst is the single largest error.

Full pipeline against the original with the LLM on: **MAE 0.117 to 0.060, band accuracy
81.1% to 94.6%, worst error 0.54 to 0.15.**

## What was changed (7 of the 12 listed weaknesses)

| # | Weakness | Change | Evidence |
|---|---|---|---|
| 3 | Preprints scored like peer-reviewed papers | Lower base scores for arXiv, bioRxiv, medRxiv, SSRN, ResearchSquare; no DOI bonus on preprint hosts | bioRxiv 0.82 to 0.57 (label 0.50); original-24 MAE 0.142 to 0.128 |
| 11 | Subdomains inherit the parent's score | `blog(s)`, `forum(s)`, `opinion`, `people`... on a known domain are capped at 0.55; small penalty on unknown domains | `blogs.nytimes.com` 0.82 to 0.55; `en.wikipedia.org` untouched |
| 5 | Any `.edu` scores high | A `/~name/` path is a personal page, capped at 0.50 | `web.mit.edu/~jdoe` 0.84 to 0.50 |
| 12 | Penalties stack; paid content scores middling | Sponsored, advertorial and press-release paths are capped at 0.25; stacked path penalties floored at -0.30 | `nytimes.com/sponsored/...` 0.62 to 0.25 |
| 4 | No notion of retraction (the novelty item) | OpenAlex lookup (free, no key) for URLs with a DOI: retracted paper capped at 0.10; journal bonus; repository cap; small citation bonus | Retracted Lancet paper: 0.72 to 0.10 rules-only; removes the worst LLM-on miss: the blend scored 0.53 at the original weights (label 0.05), because the rules gave 0.72 although the LLM alone gave 0.20 |
| 10 | Explanation is rule fragments joined by semicolons | Verdict first, plain-language reasons from a 28-domain notes table, retraction first, a "rough estimate" note for unfamiliar sites, institutional endings worded separately | Compare `credibility.py` against the original, or try URLs in the app sidebar |
| 9 | Blend weights never tested | Swept the rule weight 0 to 1; chose 0.40 rules / 0.60 LLM. Ceilings now also bind after the blend | MAE 0.077 to 0.062 from the weights; 0.062 to 0.058 more from post-blend ceilings (same collected scores) |

Not addressed: #1 (read the page), #2 (a sourced domain table), #6 (learn weights by
regression), #7 (calibration), #8 (uncertainty intervals).

## Measurement notes

- **Tests:** `python test_credibility.py` passes 77 checks (21 original, 56 ours). They run
  offline: the LLM is faked and OpenAlex records are injected. The file also passes with
  network connections blocked.
- **Robustness:** a 43-input fuzz (None, numbers, bytes, lists and dicts, IPv6 hosts, a
  100,000-character URL, null bytes, unicode, `file:` and `mailto:` links) found one defect
  in the original: lists and dicts raised `TypeError`, because the cache lookup ran before
  the input guard. The guard now runs first, and the other 41 inputs already returned a
  valid result. Lookalike domains (`nytimes.com.evil.com`, `evilnytimes.com`) are scored
  as unknown sites and do not inherit the real publisher's score. Tests cover all of this.
- **Blend sweep:** LLM scores were collected once per URL and blended offline at each weight.
  MAE is lowest near 0.20 rules weight and flat from 0.15 to 0.30. Leave-one-out
  cross-validation picked 0.20 on every fold, and tuning on the original 24 then testing on
  our 13 (and the reverse) both favoured 0.20. We use 0.40 because it has the best band
  accuracy, keeps the rules a real share, and is not a knife-edge optimum.
- **Run-to-run noise:** the original scorer's LLM-on MAE on the first 24 URLs was 0.086 (README),
  0.087 and 0.088 on repeat runs. Differences below about 0.005 are noise. The improved
  scorer's real run (0.060) and offline recomputation (0.058) differ by the same effect.

## Limits and honest caveats

1. **Labels are our own judgment**, in the same spirit as the provided block, and open to
   argument. Four of our 13 URLs were chosen to exercise specific fixes (retracted paper,
   `blogs.` subdomain, `~` page, medRxiv preprint) and one (`blogs.nytimes.com`) was added to
   test the post-blend ceiling. Improvement on those is partly built in. The fair held-out
   cases are the plain domains (CDC, Mayo Clinic, the Guardian, TechCrunch, a company blog,
   naturalnews).
2. **Weights, ceilings and several base scores are fitted or chosen after seeing our labels**
   (bioRxiv 0.55, SSRN 0.60, the 0.55 and 0.50 ceilings, the 0.40/0.60 blend). 37 URLs is a
   small sample, with one run each and no confidence intervals.
3. **The post-blend ceiling's benefit is mostly on a case we added to test it.** On the
   original labels it changes nothing.
4. **The provided PNAS URL carries a DOI that does not exist** (OpenAlex returns 404), so
   OpenAlex cannot help on it.
5. **OpenAlex only fires when the URL contains a DOI.** arXiv-style identifiers were not
   found (404 on the DOI form tried). Stacked bonuses on a known publisher can saturate a
   score at exactly 1.00 (NEJM with a lookup).
6. **Plain domains with no DOI are still the main source of error**, for example
   naturalnews.com, JAMA, WHO and the Guardian are off by 0.15 to 0.25. Page content (#1)
   would be the next step.
7. **A ceiling overrides the model by design.** A genuinely good personal page or blog post
   would be held to 0.50 or 0.55.
8. **The `/~` rule misses `people/` pages** on `.edu`, which cannot be told apart from
   departmental pages by URL alone.
9. The offline default of `evaluate.py` does not call OpenAlex, so plain runs are
   reproducible; use `--metadata` to include it.

## Reproduce

```bash
python test_credibility.py            # 77 offline checks
python evaluate.py                    # rules only, no network
python evaluate.py --metadata         # + OpenAlex (free, no key)
python evaluate.py --llm --metadata   # final configuration (needs ANTHROPIC_API_KEY, billed)
```
