# CS676: Algorithms for Data Science

Dennis Meytin · Pace University · Fall 2026

This repository holds my work for CS676. Each deliverable lives in its own folder; this page says what is where.

## What is where

| Folder | Contents |
|---|---|
| [`homework/`](homework/) | The homework exercises: [`01_lr.py`](homework/01_lr.py) (linear regression), [`02_logreg.py`](homework/02_logreg.py) (logistic regression); more added as assigned |
| [`projects/project_1/`](projects/project_1/) | **Project 1: Credibility Scoring** (see below) |
| `projects/project_2/` | Project 2 (not yet added) |
| `projects/project_3/` | Project 3 (not yet added) |

## Project 1: Credibility Scoring

A credibility scorer for web sources, plugged into a Streamlit research chatbot. Given a URL, `score_url()` returns a score between 0 and 1 and a plain-language explanation. I extended the provided baseline with preprint, subdomain, personal-page and sponsored-content rules, a real-metadata lookup through OpenAlex (retraction status, venue type, citations), better explanations, and a re-tuned blend between the rules and a Claude judgment.

**Headline result** (37 labelled URLs, language-model layer on): mean absolute error 0.117 to 0.060, and band accuracy (HIGH/MEDIUM/LOW) 81.1% to 94.6%. The full tables, how the blend weight was chosen, and the limitations are in [`projects/project_1/RESULTS.md`](projects/project_1/RESULTS.md). The technique report is [`projects/project_1/Project1_Report.pdf`](projects/project_1/Project1_Report.pdf).

| File | What it is |
|---|---|
| `credibility.py` | The scorer (the part I changed) |
| `evaluate.py` | Scores 37 labelled URLs and reports error |
| `test_credibility.py` | 77 offline checks |
| `main.py` | The Streamlit chat app (unchanged from the starter) |
| `RESULTS.md` | Results, what changed, and what still fails |
| `Project1_Report.pdf` | The technique report |

### Running it

This project uses [uv](https://docs.astral.sh/uv/). From `projects/project_1/`:

```bash
uv sync                                   # install the pinned dependencies
uv run python test_credibility.py         # 77 checks, no API key needed
uv run python evaluate.py                 # rules only, no key, no network
uv run python evaluate.py --metadata      # adds the OpenAlex lookup (free, needs network)
```

The chat app and the language-model layer need an Anthropic API key. Copy `.env.example` to `.env`, add your own key, then:

```bash
uv run python evaluate.py --llm --metadata   # final configuration (billed to your key)
uv run streamlit run main.py                 # the chat app
```

No real keys are stored in this repository; `.env` is git-ignored.
