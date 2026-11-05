# AI brand visibility tracker

[![ci](https://github.com/kuanyishen/ai-brand-visibility-tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/kuanyishen/ai-brand-visibility-tracker/actions/workflows/ci.yml)

**Does AI recommend a brand, how prominently, and would it say the same thing twice?**
This project asks AI assistants shopper-style questions, extracts every brand they mention, and measures visibility
with confidence intervals and statistical tests. First study: **Aritzia** in women's contemporary fashion
(independent analysis; not affiliated with Aritzia or any brand mentioned).

Dashboard: `docs/index.html` (publish with GitHub Pages: Settings, Pages, branch `main`, folder `/docs`).

## What it measures
- **Mention rate, share of voice, average rank** on prompts that never name a brand (the only fair visibility test).
- **Stability:** overlap of brand lists across repeated answers to the same prompt; how much variance is between
  prompts vs between repeats (ICC); how many prompts/repeats a reliable estimate needs.
- **Search and provider effects:** each model with and without web search, and Gemini vs Claude, compared prompt by prompt; cross-surface agreement; day-to-day drift.
- **Sentiment and recommendation** of the focus brand, **cited sources** in search-grounded answers.
- **Measurement audit:** two independent brand extractors (exact-name dictionary and an LLM reader) checked against each other.

## First results (440 answers, 40 prompts, 4 AI surfaces, 2026-11-05)
Gemini 2.5 Flash and Claude Sonnet 5.5, each with and without web search:
- Without naming a brand, **Aritzia is mentioned in 42-54% of answers on every surface** (3rd, 3rd, 5th and 4th of 93 brands). No surface differs significantly from another, and Aritzia is not statistically separable from its closest competitors (all paired p >= 0.22; 24 prompts can only detect gaps of about 22-32 points).
- **Web search makes answers more selective on both providers:** 3.0 fewer brands per answer for Gemini (p = 0.0003) and 3.7 for Claude (p < 0.001).
- **Different AIs name different brands.** For the same question, two different surfaces share 0.28-0.44 of their brands, less than one surface asked twice (0.46-0.65).
- **The two search tools cite very different pages.** Reddit is cited in 66 of 120 Gemini search answers and in none of Claude's 80; only 11% of cited domains overlap.
- **Single-shot measurements mislead:** 17-46% of prompts mention Aritzia in some repeats but not others; about 60-80 *different prompts* (not repeats) are needed to pin a mention rate to +/-5 points.
- Two days of Gemini data show no detectable drift.

## Method in one paragraph
Every prompt is asked several times per surface. Because answers to the same prompt are correlated, all confidence
intervals come from a **cluster bootstrap over prompts**; surface comparisons use paired per-prompt differences with a
sign-flip permutation test and Holm correction. Raw answers are kept verbatim in `data/raw/` with model version,
timestamp, token usage and cited sources, so every number can be regenerated.

## Layout
```
config/            prompts.yaml, brands.yaml (aliases), study.yaml (surfaces, repeats)
src/brandvis/      providers (Gemini, Claude), collect, extract, metrics, stats, check_keys
analysis/          analyze.py (all numbers + figures), build_site.py (dashboard)
data/raw|processed raw answers and extracted mentions (JSONL)
docs/              dashboard (index.html) and figures
tests/             29 tests: extraction rules, collector resume/quota, statistics, metrics
```

## Reproduce
```bash
pip install -e ".[dev]"
export GEMINI_API_KEY=... ANTHROPIC_API_KEY=...   # never commit keys
python -m brandvis.check_keys    # free: are the keys valid and the models available?
python -m brandvis.collect --batch 2026-11-05
python -m brandvis.extract --raw data/raw/2026-11-05.jsonl --out data/processed/2026-11-05.jsonl
python analysis/analyze.py && python analysis/build_site.py
pytest
```
The collector resumes where it stopped and halts cleanly when an API quota is exhausted. A full 4-surface run is about 2M tokens
(Claude's search surface dominates the cost because retrieved pages count as input). `.github/workflows/collect.yml` runs collection on demand from the Actions tab (inputs: surfaces, limit, extract; the defaults are a 6-call smoke test; a weekly schedule is included but commented out). Repository secrets needed: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`.

## Limitations
Two model families on one day; API is not the consumer product; the prompt set is my design (a Canada-heavy mix favours
Aritzia); only 24 unbranded prompts, so differences between brands are mostly not statistically separable; sentiment is
judged by another model; what an AI says is not what is true. Next: more AI surfaces (OpenAI, Perplexity), weekly drift tracking, a larger prompt set.
