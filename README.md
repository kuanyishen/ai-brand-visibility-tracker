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
- **Search effect:** the same model with and without Google Search grounding, compared prompt by prompt.
- **Sentiment and recommendation** of the focus brand, **cited sources** in search-grounded answers.
- **Measurement audit:** two independent brand extractors (exact-name dictionary and an LLM reader) checked against each other.

## First results (240 answers, 40 prompts, Gemini 2.5 Flash, 2026-11-03)
- Without naming a brand, **Aritzia is mentioned in 57% of answers (no search) and 46% (with search)**, 3rd and 1st among 93 tracked brands, but gaps to rivals (Everlane, Uniqlo, H&M) are within noise (Holm-adjusted p above 0.8).
- Visibility depends strongly on **topic**: near 100% for Canada, basics and workwear questions, 0% for activewear, denim and knitwear.
- Search makes answers more selective (3.7 fewer brands per answer, p = 0.0003); its effect on Aritzia's mention rate (-11 pp, 95% CI -25 to +3) is **inconclusive**, because 24 prompts can only detect differences of about 21 pp.
- Two answers to the same question share only ~60% (no search) / ~48% (search) of their brands. **More repeats cannot fix this; more prompts can** (about 70 prompts for a +/-5 pp standard error).
- Search-grounded answers lean on community and editorial sources: Reddit appears in 54% of answers, while the brand's own site is 2% of citations.

## Method in one paragraph
Every prompt is asked several times per surface. Because answers to the same prompt are correlated, all confidence
intervals come from a **cluster bootstrap over prompts**; surface comparisons use paired per-prompt differences with a
sign-flip permutation test and Holm correction. Raw answers are kept verbatim in `data/raw/` with model version,
timestamp, token usage and cited sources, so every number can be regenerated.

## Layout
```
config/            prompts.yaml, brands.yaml (aliases), study.yaml (surfaces, repeats)
src/brandvis/      providers (Gemini), collect, extract, metrics, stats
analysis/          analyze.py (all numbers + figures), build_site.py (dashboard)
data/raw|processed raw answers and extracted mentions (JSONL)
docs/              dashboard (index.html) and figures
tests/             28 tests: extraction rules, collector resume/quota, statistics, metrics
```

## Reproduce
```bash
pip install -e ".[dev]"
export GEMINI_API_KEY=...        # never commit keys
python -m brandvis.collect --batch 2026-11-03
python -m brandvis.extract --raw data/raw/2026-11-03.jsonl --out data/processed/2026-11-03.jsonl
python analysis/analyze.py && python analysis/build_site.py
pytest
```
The collector resumes where it stopped and halts cleanly when an API quota is exhausted. A full run is about 490k
tokens. `.github/workflows/collect.yml` repeats it on demand (weekly schedule is included but commented out).

## Limitations
One model family on one day; API is not the consumer product; the prompt set is my design (a Canada-heavy mix favours
Aritzia); only 24 unbranded prompts, so differences between brands are mostly not statistically separable; sentiment is
judged by another model; what an AI says is not what is true. Next: more AI surfaces, weekly drift tracking, a larger prompt set.
