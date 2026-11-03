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
from .providers import ProviderError, make_provider

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
    found: dict[str, list[int]] = {}
    for brand, pat in matcher:
        for m in pat.finditer(text):
            found.setdefault(brand, []).append(m.start())
    ordered = sorted(found.items(), key=lambda kv: min(kv[1]))
    return [Mention(b, min(pos), len(pos), i + 1) for i, (b, pos) in enumerate(ordered)]


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
    provider = provider_factory(study.extractor["provider"], study.extractor["model"]) if use_llm else None
    done = set()
    if out_path.exists():
        done = {(r["batch"], r["prompt_id"], r["surface"], r["run_index"]) for r in map(json.loads, out_path.read_text().splitlines()) if r}
    records = [r for r in map(json.loads, raw_path.read_text().splitlines()) if r]
    todo = [r for r in records if (r["batch"], r["prompt_id"], r["surface"], r["run_index"]) not in done]
    log(f"{len(records)} raw answers; {len(todo)} to extract")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    limiter, lock, stats = RateLimiter(rpm), threading.Lock(), {"ok": 0, "llm_failed": 0}

    def work(r):
        d = [m.__dict__ for m in extract_dictionary(r["text"], matcher)]
        llm, err = None, None
        if provider:
            limiter.wait()
            try:
                llm = extract_llm(provider, r["prompt"], r["text"], nmap)
            except ProviderError as e:
                err = str(e)[:200]
        return r, d, llm, err

    with out_path.open("a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(work, r) for r in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            r, d, llm, err = fut.result()
            keep = {k: r[k] for k in ("batch", "prompt_id", "prompt", "intent", "topic", "surface", "model", "grounded",
                                      "run_index", "collected_at", "sources", "search_queries")}
            rec = {**keep, "n_chars": len(r["text"]), "dict_mentions": d, "llm_mentions": llm, "llm_error": err}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            stats["ok"] += 1
            stats["llm_failed"] += err is not None
            if n % 20 == 0 or n == len(todo):
                log(f"  {n}/{len(todo)} extracted ({stats['llm_failed']} LLM failures)")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rpm", type=float, default=40)
    a = ap.parse_args()
    print(process(a.raw, a.out, use_llm=not a.no_llm, workers=a.workers, rpm=a.rpm))


if __name__ == "__main__":
    main()
