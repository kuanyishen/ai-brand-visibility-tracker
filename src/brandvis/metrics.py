"""Visibility metrics from extracted mentions.

All rates carry a cluster-bootstrap confidence interval that resamples PROMPTS (see stats.py).
"Visibility" metrics use the *discovery* prompts, where no brand is named, so a brand can only appear if the AI brings
it up on its own.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import cohens_kappa, icc1, jaccard, ratio_bootstrap, variance_components

SENTIMENT = {"positive": 1.0, "neutral": 0.0, "mixed": 0.0, "negative": -1.0}


def load_processed(*paths: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (responses, dictionary mentions, LLM mentions) as tidy tables keyed by ``rid``."""
    rows = [json.loads(l) for p in paths for l in Path(p).read_text().splitlines() if l.strip()]
    resp, dic, llm = [], [], []
    for r in rows:
        rid = f"{r['batch']}|{r['prompt_id']}|{r['surface']}|{r['run_index']}"
        resp.append({"rid": rid, "batch": r["batch"], "prompt_id": r["prompt_id"], "prompt": r["prompt"], "intent": r["intent"],
                     "topic": r["topic"], "surface": r["surface"], "run_index": r["run_index"], "grounded": r["grounded"],
                     "n_chars": r["n_chars"], "sources": r.get("sources") or [], "has_llm": r["llm_mentions"] is not None})
        dic += [{"rid": rid, **m} for m in r["dict_mentions"]]
        llm += [{"rid": rid, **m} for m in (r["llm_mentions"] or [])]
    return (pd.DataFrame(resp), pd.DataFrame(dic, columns=["rid", "brand", "position", "count", "rank"]),
            pd.DataFrame(llm, columns=["rid", "brand", "known", "order", "sentiment", "recommended"]))


def presence(resp: pd.DataFrame, mentions: pd.DataFrame, brands: list[str]) -> pd.DataFrame:
    """Boolean matrix: one row per answer (same order as ``resp``), one column per brand."""
    m = mentions.assign(v=True).pivot_table(index="rid", columns="brand", values="v", aggfunc="max", fill_value=False)
    out = m.reindex(index=resp["rid"], columns=brands, fill_value=False).astype(bool)
    out.index = resp.index          # align with ``resp`` so positional/boolean selection works on both
    return out


def _by_prompt(resp: pd.DataFrame, values: pd.Series | np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"prompt_id": resp["prompt_id"].values, "v": np.asarray(values, float)}).groupby("prompt_id")["v"].agg(["sum", "count"])


def rate_with_ci(resp: pd.DataFrame, indicator, seed: int = 0) -> dict:
    g = _by_prompt(resp, indicator)
    est, lo, hi = ratio_bootstrap(g["sum"].values, g["count"].values, seed=seed)
    return {"rate": est, "lo": lo, "hi": hi, "n_answers": int(g["count"].sum()), "n_prompts": len(g)}


def visibility_table(resp: pd.DataFrame, pres: pd.DataFrame, mentions: pd.DataFrame, brands: list[str], *, surface: str, intents=("discovery",)) -> pd.DataFrame:
    """Per brand: mention rate, share of voice, mean rank when mentioned, top-3 rate - all with prompt-level CIs."""
    sel = resp["surface"].eq(surface) & resp["intent"].isin(intents)
    r, P = resp[sel].reset_index(drop=True), pres[sel.values].reset_index(drop=True)
    total = P.sum(axis=1)                                          # tracked-brand mentions per answer
    ranks = mentions.merge(r[["rid"]], on="rid")
    out = []
    for b in brands:
        mr = rate_with_ci(r, P[b])
        g_num = _by_prompt(r, P[b])["sum"]
        g_den = _by_prompt(r, total)["sum"]
        sov, slo, shi = ratio_bootstrap(g_num.values, g_den.reindex(g_num.index).values)
        rb = ranks[ranks["brand"] == b]["rank"]
        out.append({"brand": b, "mention_rate": mr["rate"], "mr_lo": mr["lo"], "mr_hi": mr["hi"], "share_of_voice": sov,
                    "sov_lo": slo, "sov_hi": shi, "mean_rank": float(rb.mean()) if len(rb) else np.nan,
                    "top3_rate": float((ranks.assign(top=ranks["rank"] <= 3).query("brand == @b")["top"].sum()) / len(r)),
                    "n_answers": mr["n_answers"], "n_prompts": mr["n_prompts"]})
    return pd.DataFrame(out).sort_values("share_of_voice", ascending=False).reset_index(drop=True)


def focus_by_group(resp: pd.DataFrame, pres: pd.DataFrame, focus: str, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in resp.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        d = rate_with_ci(g.reset_index(drop=True), pres.loc[g.index, focus].values)
        rows.append({**dict(zip(by, key)), **d})
    return pd.DataFrame(rows)


def stability(resp: pd.DataFrame, mentions: pd.DataFrame, pres: pd.DataFrame, focus: str, top_k: int = 5) -> pd.DataFrame:
    """For every (prompt, surface): how similar are the brand lists across repeated answers?"""
    rank_by_rid = {rid: g.sort_values("rank")["brand"].tolist() for rid, g in mentions.groupby("rid")}
    rows = []
    for (pid, surf), g in resp.groupby(["prompt_id", "surface"]):
        if len(g) < 2:
            continue
        lists = [rank_by_rid.get(rid, []) for rid in g["rid"]]
        full = [jaccard(a, b) for a, b in combinations(lists, 2)]
        top = [jaccard(a[:top_k], b[:top_k]) for a, b in combinations(lists, 2)]
        rows.append({"prompt_id": pid, "surface": surf, "intent": g["intent"].iloc[0], "runs": len(g),
                     "jaccard_all": float(np.mean(full)), "jaccard_topk": float(np.mean(top)),
                     "focus_rate": float(pres.loc[g.index, focus].mean())})
    return pd.DataFrame(rows)


def focus_variance(resp: pd.DataFrame, pres: pd.DataFrame, focus: str, surface: str, intents=("discovery",)) -> dict:
    """How much of the variation in 'did the answer mention the focus brand?' is between prompts vs between repeats?"""
    sel = resp["surface"].eq(surface) & resp["intent"].isin(intents)
    groups = [pres.loc[g.index, focus].astype(float).values for _, g in resp[sel].groupby("prompt_id")]
    sb, sw = variance_components(groups)
    return {"icc": icc1(groups), "sigma_between": sb, "sigma_within": sw, "n_prompts": len(groups),
            "runs": int(np.median([len(g) for g in groups]))}


def extractor_agreement(resp: pd.DataFrame, dict_pres: pd.DataFrame, llm_mentions: pd.DataFrame, brands: list[str]) -> dict:
    llm = presence(resp, llm_mentions, brands)
    ok = resp["has_llm"].values
    a, b = dict_pres[ok].values.ravel(), llm[ok].values.ravel()
    per = []
    for br in brands:
        x, y = dict_pres[ok][br].values, llm[ok][br].values
        if x.any() or y.any():
            per.append({"brand": br, "dict_only": int((x & ~y).sum()), "llm_only": int((~x & y).sum()), "both": int((x & y).sum())})
    return {"kappa": cohens_kappa(a, b), "agree": float((a == b).mean()), "n_cells": int(a.size), "per_brand": per}


def sentiment_table(resp: pd.DataFrame, llm_mentions: pd.DataFrame, brand: str, by: list[str]) -> pd.DataFrame:
    m = llm_mentions[llm_mentions["brand"] == brand].merge(resp, on="rid")
    if m.empty:
        return pd.DataFrame()
    m["score"] = m["sentiment"].map(SENTIMENT)
    g = m.groupby(by)
    return g.agg(n=("rid", "size"), mean_sentiment=("score", "mean"), positive=("sentiment", lambda s: (s == "positive").mean()),
                 negative=("sentiment", lambda s: (s == "negative").mean()), recommended=("recommended", "mean")).reset_index()


def source_table(resp: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    g = resp[resp["grounded"]]
    s = pd.Series([d.lower().removeprefix("www.") for lst in g["sources"] for d in lst])
    if s.empty:
        return pd.DataFrame(columns=["domain", "citations", "share", "answers"])
    cited_in = pd.Series([d for lst in g["sources"] for d in {x.lower().removeprefix("www.") for x in lst}]).value_counts()
    vc = s.value_counts()
    return pd.DataFrame({"domain": vc.index, "citations": vc.values, "share": vc.values / vc.sum(),
                         "answers": cited_in.reindex(vc.index).values}).head(top_n)
