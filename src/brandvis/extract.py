"""Turn raw answers into brand mentions.

Two independent extractors are run so each can audit the other:

* **dictionary** - exact alias matching against config/brands.yaml (high precision, fast, free). Gives the order of
  first mention, which defines a brand's *rank* inside an answer.
* **LLM** - a small model returns every brand/retailer mentioned plus how the answer portrays it (sentiment,
  recommended). It also finds brands that are not in the dictionary.

    python -m brandvis.extract --raw data/raw/2026-11-03.jsonl --out data/processed/2026-11-03.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .collect import RateLimiter
from .config import Brand, load_brands, load_study
from .providers import ProviderError, QuotaExhausted, make_provider

# ----------------------------------------------------------------------------- dictionary extractor


@dataclass(frozen=True)
class Mention:
    brand: str
    position: int   # character offset of the first mention
    count: int      # number of mentions of this brand in the answer
    rank: int       # 1 = first brand mentioned


def build_matcher(brands: list[Brand]) -> list[tuple[str, re.Pattern]]:
    """One compiled pattern per brand (aliases joined), longest alias first so 'Alo Yoga' beats 'Alo'."""
    out = []
    for b in brands:
        parts = []
        for alias, cs in [(a, False) for a in b.aliases] + [(a, True) for a in b.cs]:
            parts.append((alias, cs))
        for cs in (False, True):
            group = sorted((a for a, c in parts if c == cs), key=len, reverse=True)
            if group:
                body = "|".join(re.escape(a) for a in group)
                out.append((b.name, re.compile(rf"(?<![\w])(?:{body})(?![\w])", 0 if cs else re.IGNORECASE)))
    return out


def extract_dictionary(text: str, matcher: list[tuple[str, re.Pattern]]) -> list[Mention]:
    """Find brands in ``text``. Where aliases overlap, the longest match wins ("Nordstrom Rack" is not "Nordstrom")."""
    text = text.replace("\u2019", "'")                      # curly apostrophe -> straight; same length, offsets unchanged
    hits = [(m.start(), m.end(), brand) for brand, pat in matcher for m in pat.finditer(text)]
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    kept, last_end = [], -1
    for start, end, brand in hits:
        if start >= last_end:
            kept.append((start, end, brand))
            last_end = end
    found: dict[str, list[int]] = {}
    for start, _, brand in kept:
        found.setdefault(brand, []).append(start)
    ordered = sorted(found.items(), key=lambda kv: kv[1][0])
    return [Mention(b, pos[0], len(pos), i + 1) for i, (b, pos) in enumerate(ordered)]


# ----------------------------------------------------------------------------- LLM extractor

SCHEMA = {
    "type": "OBJECT",
    "properties": {"brands": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "name": {"type": "STRING"},
        "sentiment": {"type": "STRING", "enum": ["positive", "neutral", "negative", "mixed"]},
        "recommended": {"type": "BOOLEAN"}},
        "required": ["name", "sentiment", "recommended"]}}},
    "required": ["brands"],
}

INSTRUCTIONS = """You are annotating an AI assistant's answer to a shopping question.
List every clothing brand, clothing label or clothing retailer that the ANSWER mentions, in order of first appearance.
For each one give:
- name: a short canonical brand name. Aritzia's own labels (Wilfred, Babaton, TNA, Sunday Best, Super Puff) must be reported as "Aritzia".
- sentiment: how the ANSWER portrays the brand overall (positive, neutral, negative or mixed). Judge the answer, not your own opinion.
- recommended: true only if the answer recommends or endorses the brand for what the shopper asked.
Ignore fabric names, product types, websites that are not retailers, and generic words. Do not invent brands that are not in the answer.

QUESTION: {question}

ANSWER:
{answer}"""


def extract_llm(provider, question: str, answer: str, name_map: dict[str, str]) -> list[dict]:
    data = provider.generate_json(INSTRUCTIONS.format(question=question, answer=answer[:12000]), SCHEMA)
    out, seen = [], set()
    for i, b in enumerate(data.get("brands", [])):
        raw = str(b.get("name", "")).strip()
        canon = name_map.get(raw.lower(), raw)
        if not raw or canon in seen:
            continue
        seen.add(canon)
        out.append({"brand": canon, "known": raw.lower() in name_map, "order": len(out) + 1,
                    "sentiment": b.get("sentiment", "neutral"), "recommended": bool(b.get("recommended", False))})
    return out


def name_map_from(brands: list[Brand]) -> dict[str, str]:
    m = {}
    for b in brands:
        for a in (b.name, *b.aliases, *b.cs):
            m[a.lower()] = b.name
    return m


# ----------------------------------------------------------------------------- pipeline


def process(raw_path: Path, out_path: Path, *, use_llm: bool = True, workers: int = 4, rpm: float = 40,
            provider_factory: Callable = make_provider, log: Callable = print) -> dict:
    brands = load_brands()
    matcher, nmap = build_matcher(brands), name_map_from(brands)
    study = load_study()
    ex = study.extractor
    provider = provider_factory(ex["provider"], ex["model"], timeout=60, max_retries=2) if use_llm else None
    fallback = (provider_factory(ex["provider"], ex["fallback_model"], timeout=60, max_retries=2)
                if use_llm and ex.get("fallback_model") else None)
    done = set()
    if out_path.exists():
        old = [r for r in map(json.loads, out_path.read_text().splitlines()) if r]
        failed = {(r["batch"], r["prompt_id"], r["surface"], r["run_index"]) for r in old if use_llm and r.get("llm_error")}
        done = {(r["batch"], r["prompt_id"], r["surface"], r["run_index"]) for r in old} - failed
        if failed:                                  # drop failed rows so they are re-extracted rather than duplicated
            out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in old
                                        if (r["batch"], r["prompt_id"], r["surface"], r["run_index"]) not in failed))
    records = [r for r in map(json.loads, raw_path.read_text().splitlines()) if r]
    todo = [r for r in records if (r["batch"], r["prompt_id"], r["surface"], r["run_index"]) not in done]
    log(f"{len(records)} raw answers; {len(todo)} to extract")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    limiter, lock, stats = RateLimiter(rpm), threading.Lock(), {"ok": 0, "llm_failed": 0}

    def work(r):
        d = [m.__dict__ for m in extract_dictionary(r["text"], matcher)]
        llm, err, used = None, None, None
        if provider:
            limiter.wait()
            for model_name, prov in ((ex["model"], provider), (ex.get("fallback_model"), fallback)):
                if prov is None:
                    continue
                try:
                    llm, used, err = extract_llm(prov, r["prompt"], r["text"], nmap), model_name, None
                    break
                except QuotaExhausted as e:
                    err = str(e)[:200]
                    break
                except ProviderError as e:
                    err = str(e)[:200]
        return r, d, llm, err, used

    with out_path.open("a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(work, r) for r in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            r, d, llm, err, used = fut.result()
            keep = {k: r[k] for k in ("batch", "prompt_id", "prompt", "intent", "topic", "surface", "model", "grounded",
                                      "run_index", "collected_at", "sources", "search_queries")}
            rec = {**keep, "n_chars": len(r["text"]), "dict_mentions": d, "llm_mentions": llm, "llm_model": used, "llm_error": err}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            stats["ok"] += 1
            stats["llm_failed"] += err is not None
            if n % 20 == 0 or n == len(todo):
                log(f"  {n}/{len(todo)} extracted ({stats['llm_failed']} LLM failures)")
    return stats


def refresh(raw_path: Path, out_path: Path) -> int:
    """Recompute dictionary mentions from the raw text and re-normalise LLM brand names after config/brands.yaml
    changed, WITHOUT calling the LLM again. Returns the number of records rewritten."""
    brands = load_brands()
    matcher, nmap = build_matcher(brands), name_map_from(brands)
    raw = {(r["batch"], r["prompt_id"], r["surface"], r["run_index"]): r
           for r in map(json.loads, raw_path.read_text().splitlines()) if r}
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    for rec in rows:
        r = raw[(rec["batch"], rec["prompt_id"], rec["surface"], rec["run_index"])]
        rec["dict_mentions"] = [m.__dict__ for m in extract_dictionary(r["text"], matcher)]
        if rec["llm_mentions"] is not None:
            seen, merged = set(), []
            for m in rec["llm_mentions"]:
                canon = nmap.get(m["brand"].lower(), m["brand"])
                if canon in seen:
                    continue
                seen.add(canon)
                merged.append({**m, "brand": canon, "known": m["brand"].lower() in nmap, "order": len(merged) + 1})
            rec["llm_mentions"] = merged
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="re-run only the dictionary step on existing output (no API calls)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rpm", type=float, default=40)
    a = ap.parse_args()
    if a.refresh:
        print({"refreshed": refresh(a.raw, a.out)})
        return
    print(process(a.raw, a.out, use_llm=not a.no_llm, workers=a.workers, rpm=a.rpm))


if __name__ == "__main__":
    main()
