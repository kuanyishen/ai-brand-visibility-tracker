"""Ask every prompt on every surface several times and append raw answers to a JSONL file.

Resumable: records already present for (batch, prompt, surface, run) are skipped, so a crashed or rate-limited run
can simply be started again.

    python -m brandvis.collect --out data/raw/2026-11-03.jsonl [--limit 12] [--workers 4] [--rpm 30]
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import Prompt, Study, load_prompts, load_study
from .providers import ProviderError, QuotaExhausted, make_provider


def job_key(batch: str, prompt_id: str, surface_id: str, run_index: int) -> tuple:
    return (batch, prompt_id, surface_id, run_index)


def existing_keys(path: Path) -> set[tuple]:
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add(job_key(r["batch"], r["prompt_id"], r["surface"], r["run_index"]))
    return keys


class RateLimiter:
    """At most ``rpm`` calls per minute across all threads."""

    def __init__(self, rpm: float | None):
        self.interval = 60.0 / rpm if rpm else 0.0
        self._lock, self._next = threading.Lock(), 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if delay:
            time.sleep(delay)


def run(study: Study, prompts: list[Prompt], out: Path, batch: str, *, workers: int = 4, rpm: float | None = 30,
        limit: int | None = None, provider_factory: Callable = make_provider, log: Callable = print) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(out)
    jobs = [(p, s, i) for p in prompts for s in study.surfaces for i in range(s.runs or study.runs_per_prompt)
            if job_key(batch, p.id, s.id, i) not in done]
    if limit:
        jobs = jobs[:limit]
    log(f"{len(done)} already collected; {len(jobs)} to do")
    providers = {s.id: provider_factory(s.provider, s.model) for s in study.surfaces}
    limiter, lock = RateLimiter(rpm), threading.Lock()
    stats = {"ok": 0, "failed": 0, "quota_exhausted": False}

    def work(job):
        p, s, i = job
        limiter.wait()
        r = providers[s.id].generate(p.text, temperature=study.temperature, grounded=s.grounded)
        return job, r

    with out.open("a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, j): j for j in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            p, s, i = futures[fut]
            try:
                _, r = fut.result()
            except QuotaExhausted as e:
                stats["quota_exhausted"] = True
                log(f"  STOPPING: {str(e)[:200]}")
                for other in futures:
                    other.cancel()
                break
            except ProviderError as e:
                stats["failed"] += 1
                log(f"  FAILED {p.id}/{s.id}/{i}: {str(e)[:160]}")
                continue
            rec = {"batch": batch, "prompt_id": p.id, "prompt": p.text, "intent": p.intent, "topic": p.topic,
                   "surface": s.id, "provider": s.provider, "model_requested": s.model, "model": r.model,
                   "grounded": r.grounded, "run_index": i, "temperature": study.temperature,
                   "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "text": r.text, "sources": r.sources, "search_queries": r.search_queries,
                   "usage": r.usage, "finish_reason": r.finish_reason, "latency_s": r.latency_s}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            stats["ok"] += 1
            if n % 10 == 0 or n == len(jobs):
                log(f"  {n}/{len(jobs)} done ({stats['failed']} failed)")
    return stats


def select_surfaces(study: Study, ids: str | None) -> Study:
    """Restrict the study to a comma-separated list of surface ids (None = all)."""
    if not ids:
        return study
    wanted = [i.strip() for i in ids.split(",") if i.strip()]
    known = {s.id for s in study.surfaces}
    missing = [w for w in wanted if w not in known]
    if missing:
        raise SystemExit(f"unknown surface id(s) {missing}; available: {sorted(known)}")
    return dataclasses.replace(study, surfaces=tuple(s for s in study.surfaces if s.id in wanted))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="JSONL file (default data/raw/<batch>.jsonl)")
    ap.add_argument("--batch", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rpm", type=float, default=30)
    ap.add_argument("--limit", type=int, default=None, help="only run the first N pending jobs (for testing)")
    ap.add_argument("--surfaces", default=None, help="comma-separated surface ids to run (default: all in config/study.yaml)")
    a = ap.parse_args()
    out = a.out or Path("data/raw") / f"{a.batch}.jsonl"
    stats = run(select_surfaces(load_study(), a.surfaces), load_prompts(), out, a.batch, workers=a.workers, rpm=a.rpm, limit=a.limit or None)
    print(stats)
    raise SystemExit(1 if stats["failed"] else 0)


if __name__ == "__main__":
    main()
