"""Compute every number and figure for the dashboard from the processed mentions.

    python analysis/analyze.py [--processed data/processed/<batch>.jsonl ...] [--batch 2026-11-05]

The main snapshot is one batch (all surfaces collected the same day). Other batches, if given, are used for the
day-to-day drift analysis of the surfaces they share.
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import plots
from brandvis.config import load_brands, load_study
from brandvis.metrics import (extractor_agreement, focus_by_group, focus_variance, load_processed, presence, rate_with_ci,
                              source_table, stability, visibility_table)
from brandvis.stats import jaccard, paired_diff_bootstrap, prompts_needed, runs_needed, se_of_mean, sign_flip_test

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
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


def paired(a: pd.Series, b: pd.Series) -> dict:
    """Paired comparison across prompts: mean of (a - b) with CI, sign-flip p and the smallest effect this design can detect."""
    idx = a.index.intersection(b.index)
    diff = (a.loc[idx] - b.loc[idx]).values
    m, lo, hi = paired_diff_bootstrap(diff)
    sd = float(np.std(diff, ddof=1))
    return {"mean_diff": m, "lo": lo, "hi": hi, "p_value": sign_flip_test(diff), "n_prompts": len(diff),
            "n_prompts_differing": int((diff != 0).sum()), "sd_of_diffs": sd, "mde_80pct": 2.8 * sd / np.sqrt(len(diff)),
            "a_mean": float(a.loc[idx].mean()), "b_mean": float(b.loc[idx].mean())}


def answer_sets(resp: pd.DataFrame, dic: pd.DataFrame) -> dict[str, set]:
    sets = dic.groupby("rid")["brand"].agg(set).to_dict()
    return {rid: sets.get(rid, set()) for rid in resp["rid"]}


def cross_jaccard(resp_a: pd.DataFrame, resp_b: pd.DataFrame, sets: dict, same: bool = False) -> float:
    """Mean over prompts of the mean Jaccard between answers of group A and group B (distinct answers only if same)."""
    per_prompt = []
    for pid, ga in resp_a.groupby("prompt_id"):
        gb = resp_b[resp_b.prompt_id == pid]
        vals = [jaccard(sets[x], sets[y]) for x, y in product(ga.rid, gb.rid) if not (same and x == y)]
        if vals:
            per_prompt.append(np.mean(vals))
    return float(np.mean(per_prompt))


def main(paths: list[Path], main_batch: str) -> dict:
    study, brands_cfg = load_study(), load_brands()
    brands = [b.name for b in brands_cfg]
    focus = study.focus_brand
    resp_all, dic_all, llm_all = load_processed(*paths)
    cfg = {s.id: s for s in study.surfaces}
    label = {s.id: (s.label or s.id) for s in study.surfaces}

    # main snapshot: one batch, every surface; only the repeats the study design calls for (drop extra smoke-test repeats)
    keep = resp_all.batch.eq(main_batch) & resp_all.apply(lambda r: r["surface"] in cfg and r["run_index"] < (cfg[r["surface"]].runs or study.runs_per_prompt), axis=1)
    resp = resp_all[keep].reset_index(drop=True)
    dic, llm = dic_all[dic_all.rid.isin(resp.rid)], llm_all[llm_all.rid.isin(resp.rid)]
    surfaces = [s.id for s in study.surfaces if s.id in set(resp.surface)]
    pres = presence(resp, dic, brands)
    pres_llm = presence(resp, llm, brands)
    raw = [json.loads(l) for p in paths for l in (ROOT / "data" / "raw" / p.name).read_text().splitlines() if l.strip()]
    raw = [r for r in raw if r["batch"] == main_batch and f'{r["batch"]}|{r["prompt_id"]}|{r["surface"]}|{r["run_index"]}' in set(resp.rid)]
    sets = answer_sets(resp, dic)

    R: dict = {"meta": {
        "batch": main_batch, "n_answers": len(resp), "n_prompts": resp.prompt_id.nunique(), "runs_per_prompt": study.runs_per_prompt,
        "surfaces": {s: {"label": label[s], "provider": cfg[s].provider, "model": cfg[s].model, "grounded": cfg[s].grounded,
                         "runs": cfg[s].runs or study.runs_per_prompt, "n_answers": int((resp.surface == s).sum())} for s in surfaces},
        "intent_counts": {k: int(v) for k, v in resp.drop_duplicates("prompt_id").intent.value_counts().items()},
        "n_brands_tracked": len(brands), "collected_on": min(r["collected_at"] for r in raw)[:10],
        "tokens": {p: int(sum((r.get("usage") or {}).get("totalTokenCount", 0) for r in raw if r["provider"] == p)) for p in sorted({r["provider"] for r in raw})},
        "focus": focus, "category": study.category}}
    R["surface_order"] = surfaces

    # ---- visibility (discovery prompts: the brand is never named) --------------------------------------------------------
    R["visibility"], R["focus_rank"] = {}, {}
    for s in surfaces:
        t = visibility_table(resp, pres, dic, brands, surface=s)
        R["visibility"][s] = t.head(30).round(5).to_dict("records")
        order = t.sort_values("mention_rate", ascending=False).brand.tolist()
        R["focus_rank"][s] = {"by_mention_rate": order.index(focus) + 1, "by_share_of_voice": t.brand.tolist().index(focus) + 1,
                              "n_brands_seen": int((t.mention_rate > 0).sum())}
    R["focus_by_intent"] = focus_by_group(resp, pres, focus, ["surface", "intent"]).round(5).to_dict("records")
    d = resp[resp.intent == "discovery"]
    ft = d.assign(m=pres.loc[d.index, focus].astype(float)).groupby(["topic", "surface"]).agg(rate=("m", "mean"), n_answers=("m", "size"), n_prompts=("prompt_id", "nunique")).reset_index()
    R["focus_by_topic"] = ft.round(4).to_dict("records")

    # ---- paired comparisons: search effect within each provider, provider effect within each mode ------------------------
    measures = {"focus_mention": pres[focus], "brands_per_answer": pres.sum(axis=1)}
    pairs = []
    for p in sorted({cfg[s].provider for s in surfaces}):
        plain = next((s for s in surfaces if cfg[s].provider == p and not cfg[s].grounded), None)
        search = next((s for s in surfaces if cfg[s].provider == p and cfg[s].grounded), None)
        if plain and search:
            pairs.append(("search_effect", search, plain, f"{p.title()}: with search minus without"))
    for grounded in (False, True):
        group = [s for s in surfaces if cfg[s].grounded == grounded]
        providers = {cfg[s].provider: s for s in group}
        if "anthropic" in providers and "gemini" in providers:
            pairs.append(("provider_effect", providers["anthropic"], providers["gemini"], f"Claude minus Gemini, {'with search' if grounded else 'no search'}"))
    R["pairs"] = []
    for kind, a, b, name in pairs:
        R["pairs"].append({"kind": kind, "a": a, "b": b, "name": name,
                           "metrics": {m: paired(prompt_rates(resp, ser, a), prompt_rates(resp, ser, b)) for m, ser in measures.items()}})

    # ---- is the focus brand more/less visible than each competitor? (paired by prompt, Holm-adjusted) -------------------
    R["vs_competitors"] = {}
    for s in surfaces:
        top = [r["brand"] for r in R["visibility"][s] if r["brand"] != focus][:10]
        f = prompt_rates(resp, pres[focus], s)
        rows = []
        for b in top:
            pr = paired(f, prompt_rates(resp, pres[b], s))
            rows.append({"competitor": b, "diff": pr["mean_diff"], "lo": pr["lo"], "hi": pr["hi"], "p": pr["p_value"]})
        for r, p in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = p
        R["vs_competitors"][s] = rows

    # ---- stability + variance decomposition + design ------------------------------------------------------------------------
    st = stability(resp, dic, pres, focus)
    R["stability"] = st.groupby(["surface", "intent"])[["jaccard_all", "jaccard_topk"]].mean().reset_index().round(4).to_dict("records")
    R["focus_consistency"], R["variance"], R["design"] = {}, {}, {}
    for s in surfaces:
        g = st[(st.surface == s) & (st.intent == "discovery")]
        R["focus_consistency"][s] = {"always": int((g.focus_rate == 1).sum()), "never": int((g.focus_rate == 0).sum()),
                                      "sometimes": int(((g.focus_rate > 0) & (g.focus_rate < 1)).sum()), "n_prompts": len(g)}
        v = focus_variance(resp, pres, focus, s)
        sb, sw, P, r = v["sigma_between"], v["sigma_within"], v["n_prompts"], v["runs"]
        R["variance"][s] = {**{k: float(x) for k, x in v.items()},
                            "prompts_for_se5_with_3_runs": prompts_needed(sb, sw, 3, 0.05), "prompts_for_se5_with_30_runs": prompts_needed(sb, sw, 30, 0.05),
                            "runs_for_se8": runs_needed(sb, sw, P, 0.08), "runs_for_se5": runs_needed(sb, sw, P, 0.05)}
        rg, pg = [1, 2, 3, 5, 10, 20, 50, 100], [12, 24, 36, 48, 72, 96, 144, 200]
        R["design"][s] = {"runs_grid": rg, "se_by_runs": [se_of_mean(sb, sw, P, x) for x in rg],
                          "prompts_grid": pg, "se_by_prompts": [se_of_mean(sb, sw, x, 3) for x in pg]}

    # ---- do the surfaces agree with each other? -------------------------------------------------------------------------------
    dd = resp[resp.intent == "discovery"]
    by_s = {s: dd[dd.surface == s] for s in surfaces}
    agree = {a: {b: cross_jaccard(by_s[a], by_s[b], sets, same=(a == b)) for b in surfaces} for a in surfaces}
    vis = {s: pd.Series({r["brand"]: r["mention_rate"] for r in visibility_table(resp, pres, dic, brands, surface=s).to_dict("records")}) for s in surfaces}
    rank_corr = {a: {b: float(spearmanr(vis[a], vis[b]).statistic) for b in surfaces} for a in surfaces}
    R["agreement"] = {"jaccard": agree, "spearman_mention_rates": rank_corr}

    # ---- day-to-day drift for surfaces present in more than one batch -------------------------------------------------------
    R["drift"] = []
    other = resp_all[~resp_all.batch.eq(main_batch) & resp_all.surface.isin(surfaces)]
    if len(other):
        pres_all = presence(resp_all, dic_all, brands)
        sets_all = answer_sets(resp_all, dic_all)
        for s in sorted(other.surface.unique()):
            for ob in sorted(other[other.surface == s].batch.unique()):
                new = resp_all[resp_all.batch.eq(main_batch) & resp_all.surface.eq(s) & resp_all.intent.eq("discovery")]
                old = resp_all[resp_all.batch.eq(ob) & resp_all.surface.eq(s) & resp_all.intent.eq("discovery")]
                if new.empty or old.empty:
                    continue
                a = pd.Series(pres_all.loc[new.index, focus].astype(float).values, index=new.prompt_id.values).groupby(level=0).mean()
                b = pd.Series(pres_all.loc[old.index, focus].astype(float).values, index=old.prompt_id.values).groupby(level=0).mean()
                pr = paired(a, b)
                R["drift"].append({"surface": s, "old_batch": ob, "new_batch": main_batch, **pr,
                                   "jaccard_across_days": cross_jaccard(old, new, sets_all), "jaccard_within_new_day": cross_jaccard(new, new, sets_all, same=True),
                                   "jaccard_within_old_day": cross_jaccard(old, old, sets_all, same=True)})

    # ---- extractor audit and sensitivity --------------------------------------------------------------------------------------
    ag = extractor_agreement(resp, pres, llm, brands)
    R["extractor"] = {"kappa": ag["kappa"], "agree": ag["agree"], "n_cells": ag["n_cells"], "per_brand": sorted(ag["per_brand"], key=lambda x: -(x["dict_only"] + x["llm_only"]))[:12],
                      "focus_rate_dictionary": {}, "focus_rate_llm": {}}
    for s in surfaces:
        sel = resp.surface.eq(s) & resp.intent.eq("discovery")
        R["extractor"]["focus_rate_dictionary"][s] = rate_with_ci(resp[sel].reset_index(drop=True), pres.loc[sel, focus].values)
        R["extractor"]["focus_rate_llm"][s] = rate_with_ci(resp[sel].reset_index(drop=True), pres_llm.loc[sel, focus].values)
    unknown = llm[~llm["known"]]
    R["extractor"]["untracked_share_of_llm_mentions"] = float(len(unknown) / max(len(llm), 1))
    R["extractor"]["top_untracked"] = unknown.groupby("brand").rid.nunique().sort_values(ascending=False).head(15).to_dict()

    # ---- sentiment for the focus brand (LLM extractor) ------------------------------------------------------------------------
    m = llm[llm.brand == focus].merge(resp, on="rid")
    sent = m.groupby(["surface", "intent", "sentiment"]).size().unstack(fill_value=0)
    rec = m.groupby(["surface", "intent"]).recommended.mean()
    R["sentiment"] = [{"surface": s, "intent": i, "n": int(sent.loc[(s, i)].sum()), **{k: int(sent.loc[(s, i)].get(k, 0)) for k in ("positive", "neutral", "mixed", "negative")},
                       "recommended": float(rec.loc[(s, i)])} for (s, i) in sent.index]

    # ---- sources, per grounded surface ----------------------------------------------------------------------------------------
    owned = {re.sub(r"[^a-z0-9]", "", a.lower()) for b in brands_cfg for a in (b.name, *b.aliases, *b.cs)}
    R["sources"], domains = {}, {}
    for s in [x for x in surfaces if cfg[x].grounded]:
        g = resp[resp.surface == s]
        allsrc = source_table(g, 10_000)
        allsrc["category"] = [domain_category(x, owned) for x in allsrc.domain]
        top = allsrc.head(15)
        cat = allsrc.groupby("category").citations.sum()
        domains[s] = set(allsrc.domain)
        R["sources"][s] = {"top": top.round(5).to_dict("records"), "category_share": (cat / cat.sum()).round(4).to_dict(),
                           "n_grounded_answers": len(g), "mean_citations_per_answer": float(np.mean([len(x) for x in g.sources])),
                           "n_domains": int(len(allsrc)),
                           "answers_citing_reddit": int(sum("reddit.com" in {x.lower().removeprefix("www.") for x in lst} for lst in g.sources)),
                           "focus_brand_site_share": float(allsrc.loc[allsrc.domain.isin([f"{focus.lower()}.com"]), "citations"].sum() / allsrc.citations.sum()),
                           "answers_citing_focus_site": int(sum(f"{focus.lower()}.com" in {x.lower().removeprefix("www.") for x in lst} for lst in g.sources))}
    ss = list(domains)
    if len(ss) == 2:
        both = domains[ss[0]] & domains[ss[1]]
        R["source_overlap"] = {"a": ss[0], "b": ss[1], "shared_domains": len(both), "jaccard": len(both) / len(domains[ss[0]] | domains[ss[1]]),
                               "n_a": len(domains[ss[0]]), "n_b": len(domains[ss[1]])}

    R["labels"] = label
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=float))
    plots.make_all(R, OUT)
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, nargs="+", default=[ROOT / "data" / "processed" / "2026-11-05.jsonl", ROOT / "data" / "processed" / "2026-11-03.jsonl"])
    ap.add_argument("--batch", default="2026-11-05", help="the main snapshot batch")
    a = ap.parse_args()
    main([p for p in a.processed if p.exists()], a.batch)
    print("wrote", OUT / "results.json")
