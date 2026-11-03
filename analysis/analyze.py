"""Compute every number and figure for the dashboard from the processed mentions.

    python analysis/analyze.py [--processed data/processed/2026-11-03.jsonl ...]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import plots
from brandvis.config import load_brands, load_prompts, load_study
from brandvis.metrics import (extractor_agreement, focus_by_group, focus_variance, load_processed, presence, rate_with_ci,
                              sentiment_table, source_table, stability, visibility_table)
from brandvis.stats import paired_diff_bootstrap, prompts_needed, runs_needed, se_of_mean, sign_flip_test

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
SURFACES = ["gemini-flash", "gemini-flash-search"]
SOCIAL = {"reddit.com", "youtube.com", "tiktok.com", "instagram.com", "pinterest.com", "lemon8-app.com", "facebook.com", "quora.com", "x.com"}
REVIEWS = {"trustpilot.com", "thingtesting.com", "sitejabber.com", "consumeraffairs.com", "bbb.org", "yelp.com"}


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adj, run = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def domain_category(domain: str, owned: set[str]) -> str:
    d = domain.lower().removeprefix("www.")
    base = re.sub(r"[^a-z0-9]", "", d.split(".")[0])
    if d in SOCIAL:
        return "social / community"
    if d in REVIEWS:
        return "reviews"
    if base in owned:
        return "retailer or brand site"
    return "editorial / blogs"


def prompt_rates(resp: pd.DataFrame, values: pd.Series, surface: str, intent: str = "discovery") -> pd.Series:
    sel = resp["surface"].eq(surface) & resp["intent"].eq(intent)
    return pd.Series(values[sel].astype(float).values, index=resp.loc[sel, "prompt_id"].values).groupby(level=0).mean()


def main(paths: list[Path]) -> dict:
    study, brands_cfg, prompts = load_study(), load_brands(), load_prompts()
    brands = [b.name for b in brands_cfg]
    focus = study.focus_brand
    resp, dic, llm = load_processed(*paths)
    pres = presence(resp, dic, brands)
    pres_llm = presence(resp, llm, brands)
    raw = [json.loads(l) for p in paths for l in (ROOT / "data" / "raw" / p.name).read_text().splitlines() if l.strip()]
    R: dict = {"meta": {
        "n_answers": len(resp), "n_prompts": resp.prompt_id.nunique(), "runs_per_prompt": study.runs_per_prompt,
        "surfaces": {s.id: {"model": s.model, "grounded": s.grounded} for s in study.surfaces},
        "intent_counts": {k: int(v) for k, v in resp.drop_duplicates("prompt_id").intent.value_counts().items()},
        "n_brands_tracked": len(brands), "collected_from": min(r["collected_at"] for r in raw)[:10], "collected_to": max(r["collected_at"] for r in raw)[:10],
        "total_tokens": int(sum((r.get("usage") or {}).get("totalTokenCount", 0) for r in raw)),
        "focus": focus, "category": study.category}}

    # ---- visibility (discovery prompts: the brand is never named) ------------------------------------------------
    R["visibility"], R["focus_rank"] = {}, {}
    for s in SURFACES:
        t = visibility_table(resp, pres, dic, brands, surface=s)
        R["visibility"][s] = t.head(30).round(5).to_dict("records")
        order = t.sort_values("mention_rate", ascending=False).brand.tolist()
        R["focus_rank"][s] = {"by_mention_rate": order.index(focus) + 1, "by_share_of_voice": t.brand.tolist().index(focus) + 1,
                              "n_brands_seen": int((t.mention_rate > 0).sum())}
    R["focus_by_intent"] = focus_by_group(resp, pres, focus, ["surface", "intent"]).round(5).to_dict("records")
    d = resp[resp.intent == "discovery"]
    ft = d.assign(m=pres.loc[d.index, focus].astype(float)).groupby(["topic", "surface"]).agg(rate=("m", "mean"), n_answers=("m", "size"), n_prompts=("prompt_id", "nunique")).reset_index()
    R["focus_by_topic"] = ft.round(4).to_dict("records")

    # ---- plain vs search: paired by prompt -------------------------------------------------------------------------
    cmp = {}
    for name, series in (("focus_mention", pres[focus]), ("brands_per_answer", pres.sum(axis=1))):
        a, b = prompt_rates(resp, series, "gemini-flash-search"), prompt_rates(resp, series, "gemini-flash")
        diff = (a - b.reindex(a.index)).values
        m, lo, hi = paired_diff_bootstrap(diff)
        sd = float(np.std(diff, ddof=1))
        cmp[name] = {"mean_diff_search_minus_plain": m, "lo": lo, "hi": hi, "p_value": sign_flip_test(diff), "n_prompts": len(diff),
                     "n_prompts_differing": int((diff != 0).sum()), "sd_of_diffs": sd, "mde_80pct": 2.8 * sd / np.sqrt(len(diff)),
                     "plain_mean": float(b.mean()), "search_mean": float(a.mean())}
    R["surface_comparison"] = cmp

    # ---- is the focus brand more/less visible than each competitor? (paired by prompt, Holm-adjusted) ----------------
    R["vs_competitors"] = {}
    for s in SURFACES:
        top = [r["brand"] for r in R["visibility"][s] if r["brand"] != focus][:10]
        f = prompt_rates(resp, pres[focus], s)
        rows = []
        for b in top:
            diff = (f - prompt_rates(resp, pres[b], s).reindex(f.index)).values
            m, lo, hi = paired_diff_bootstrap(diff)
            rows.append({"competitor": b, "diff": m, "lo": lo, "hi": hi, "p": sign_flip_test(diff)})
        for r, p in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = p
        R["vs_competitors"][s] = rows

    # ---- stability + variance decomposition ------------------------------------------------------------------------
    st = stability(resp, dic, pres, focus)
    R["stability"] = st.groupby(["surface", "intent"])[["jaccard_all", "jaccard_topk"]].mean().reset_index().round(4).to_dict("records")
    R["focus_consistency"] = {}
    for s in SURFACES:
        g = st[(st.surface == s) & (st.intent == "discovery")]
        R["focus_consistency"][s] = {"always": int((g.focus_rate == 1).sum()), "never": int((g.focus_rate == 0).sum()),
                                      "sometimes": int(((g.focus_rate > 0) & (g.focus_rate < 1)).sum()), "n_prompts": len(g)}
    R["variance"], R["design"] = {}, {}
    for s in SURFACES:
        v = focus_variance(resp, pres, focus, s)
        sb, sw, P, r = v["sigma_between"], v["sigma_within"], v["n_prompts"], v["runs"]
        R["variance"][s] = {**{k: float(x) for k, x in v.items()},
                            "prompts_for_se5_with_3_runs": prompts_needed(sb, sw, 3, 0.05), "prompts_for_se5_with_30_runs": prompts_needed(sb, sw, 30, 0.05),
                            "runs_for_se8": runs_needed(sb, sw, P, 0.08), "runs_for_se5": runs_needed(sb, sw, P, 0.05)}
        rg, pg = [1, 2, 3, 5, 10, 20, 50, 100], [12, 24, 36, 48, 72, 96, 144, 200]
        R["design"][s] = {"runs_grid": rg, "se_by_runs": [se_of_mean(sb, sw, P, x) for x in rg],
                          "prompts_grid": pg, "se_by_prompts": [se_of_mean(sb, sw, x, 3) for x in pg]}

    # ---- extractor audit and sensitivity ---------------------------------------------------------------------------
    ag = extractor_agreement(resp, pres, llm, brands)
    R["extractor"] = {"kappa": ag["kappa"], "agree": ag["agree"], "n_cells": ag["n_cells"], "per_brand": sorted(ag["per_brand"], key=lambda x: -(x["dict_only"] + x["llm_only"]))[:12],
                      "focus_rate_dictionary": {}, "focus_rate_llm": {}}
    for s in SURFACES:
        sel = resp.surface.eq(s) & resp.intent.eq("discovery")
        R["extractor"]["focus_rate_dictionary"][s] = rate_with_ci(resp[sel].reset_index(drop=True), pres.loc[sel, focus].values)
        R["extractor"]["focus_rate_llm"][s] = rate_with_ci(resp[sel].reset_index(drop=True), pres_llm.loc[sel, focus].values)
    unknown = llm[~llm["known"]]
    R["extractor"]["untracked_share_of_llm_mentions"] = float(len(unknown) / max(len(llm), 1))
    R["extractor"]["top_untracked"] = unknown.groupby("brand").rid.nunique().sort_values(ascending=False).head(15).to_dict()

    # ---- sentiment for the focus brand (LLM extractor) ---------------------------------------------------------------
    m = llm[llm.brand == focus].merge(resp, on="rid")
    sent = m.groupby(["surface", "intent", "sentiment"]).size().unstack(fill_value=0)
    rec = m.groupby(["surface", "intent"]).recommended.mean()
    R["sentiment"] = [{"surface": s, "intent": i, "n": int(sent.loc[(s, i)].sum()), **{k: int(sent.loc[(s, i)].get(k, 0)) for k in ("positive", "neutral", "mixed", "negative")},
                       "recommended": float(rec.loc[(s, i)])} for (s, i) in sent.index]

    # ---- sources --------------------------------------------------------------------------------------------------------
    owned = {re.sub(r"[^a-z0-9]", "", a.lower()) for b in brands_cfg for a in (b.name, *b.aliases, *b.cs)}
    top = source_table(resp, 15)
    top["category"] = [domain_category(x, owned) for x in top.domain]
    allsrc = source_table(resp, 10_000)
    allsrc["category"] = [domain_category(x, owned) for x in allsrc.domain]
    cat = allsrc.groupby("category").citations.sum()
    g = resp[resp.grounded]
    R["sources"] = {"top": top.round(5).to_dict("records"), "category_share": (cat / cat.sum()).round(4).to_dict(),
                    "n_grounded_answers": len(g), "mean_citations_per_answer": float(np.mean([len(x) for x in g.sources])),
                    "n_domains": int(len(allsrc)),
                    "answers_citing_reddit": int(sum("reddit.com" in {x.lower().removeprefix("www.") for x in lst} for lst in g.sources)),
                    "focus_brand_site_share": float(allsrc.loc[allsrc.domain.isin(["aritzia.com"]), "citations"].sum() / allsrc.citations.sum()),
                    "answers_citing_focus_site": int(sum("aritzia.com" in {x.lower().removeprefix("www.") for x in lst} for lst in g.sources))}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=float))
    plots.make_all(R, OUT)
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, nargs="+", default=[ROOT / "data" / "processed" / "2026-11-03.jsonl"])
    R = main(ap.parse_args().processed)
    print("wrote", OUT / "results.json")
