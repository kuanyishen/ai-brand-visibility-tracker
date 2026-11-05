import json

from brandvis.collect import existing_keys, run
from brandvis.config import Prompt, Study, Surface
from brandvis.providers import ProviderError, QuotaExhausted, Response

STUDY = Study("test", "Aritzia", 2, 1.0, (Surface("s1", "fake", "m", False), Surface("s2", "fake", "m", True)))
PROMPTS = [Prompt("p1", "q1", "discovery"), Prompt("p2", "q2", "versus")]


class Fake:
    def __init__(self, fail_on=None, quota_after=None):
        self.calls, self.fail_on, self.quota_after = 0, fail_on, quota_after

    def generate(self, prompt, *, temperature, grounded):
        self.calls += 1
        if self.quota_after is not None and self.calls > self.quota_after:
            raise QuotaExhausted("daily quota")
        if prompt == self.fail_on:
            raise ProviderError("boom")
        return Response(text=f"answer to {prompt}", model="m-1", grounded=grounded, sources=["a.com"] if grounded else [])


def test_collects_every_prompt_surface_run(tmp_path):
    out = tmp_path / "raw.jsonl"
    stats = run(STUDY, PROMPTS, out, "b1", workers=1, rpm=None, provider_factory=lambda p, m: Fake(), log=lambda *_: None)
    assert stats["ok"] == 8 and stats["failed"] == 0
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len({(r["prompt_id"], r["surface"], r["run_index"]) for r in recs}) == 8
    assert all(r["batch"] == "b1" and r["text"] for r in recs)


def test_resume_skips_finished_jobs(tmp_path):
    out = tmp_path / "raw.jsonl"
    run(STUDY, PROMPTS, out, "b1", workers=1, rpm=None, limit=3, provider_factory=lambda p, m: Fake(), log=lambda *_: None)
    assert len(existing_keys(out)) == 3
    fake = Fake()
    run(STUDY, PROMPTS, out, "b1", workers=1, rpm=None, provider_factory=lambda p, m: fake, log=lambda *_: None)
    assert fake.calls == 5 and len(existing_keys(out)) == 8


def test_failures_are_skipped_not_recorded(tmp_path):
    out = tmp_path / "raw.jsonl"
    stats = run(STUDY, PROMPTS, out, "b1", workers=1, rpm=None, provider_factory=lambda p, m: Fake(fail_on="q2"), log=lambda *_: None)
    assert stats["failed"] == 4 and stats["ok"] == 4
    assert all(json.loads(l)["prompt_id"] == "p1" for l in out.read_text().splitlines())


def test_daily_quota_stops_the_run(tmp_path):
    out = tmp_path / "raw.jsonl"
    shared = Fake(quota_after=2)
    stats = run(STUDY, PROMPTS, out, "b1", workers=1, rpm=None, provider_factory=lambda p, m: shared, log=lambda *_: None)
    assert stats["quota_exhausted"] is True
    assert stats["ok"] <= 2


def test_per_surface_repeat_counts(tmp_path):
    study = Study("test", "Aritzia", 3, 1.0, (Surface("cheap", "fake", "m", False), Surface("costly", "fake", "m", True, runs=1)))
    out = tmp_path / "raw.jsonl"
    run(study, PROMPTS, out, "b1", workers=1, rpm=None, provider_factory=lambda p, m: Fake(), log=lambda *_: None)
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    by = {}
    for r in recs:
        by.setdefault(r["surface"], []).append(r["run_index"])
    assert len(by["cheap"]) == 6 and len(by["costly"]) == 2          # 2 prompts x (3 | 1) repeats
