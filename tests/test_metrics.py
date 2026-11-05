import numpy as np
import pandas as pd
import pytest

from brandvis.metrics import extractor_agreement, focus_by_group, focus_variance, presence, sentiment_table, source_table, stability, visibility_table

BRANDS = ["Aritzia", "Zara", "Uniqlo"]


def build():
    """4 prompts x 2 runs on one surface. Aritzia: prompt1 always, prompt2 once, prompts 3-4 never."""
    layout = {  # (prompt, run) -> ordered brands
        ("p1", 0): ["Aritzia", "Zara"], ("p1", 1): ["Zara", "Aritzia"],
        ("p2", 0): ["Zara", "Uniqlo"], ("p2", 1): ["Aritzia", "Zara"],
        ("p3", 0): ["Uniqlo"], ("p3", 1): ["Uniqlo", "Zara"],
        ("p4", 0): [], ("p4", 1): ["Zara"],
    }
    resp, men = [], []
    for (p, r), bs in layout.items():
        rid = f"b|{p}|s|{r}"
        resp.append({"rid": rid, "prompt_id": p, "intent": "discovery", "surface": "s", "run_index": r, "grounded": False,
                     "sources": [], "has_llm": True, "topic": "t"})
        men += [{"rid": rid, "brand": b, "position": i * 10, "count": 1, "rank": i + 1} for i, b in enumerate(bs)]
    resp = pd.DataFrame(resp)
    men = pd.DataFrame(men)
    return resp, men, presence(resp, men, BRANDS)


def test_presence_matrix_shape_and_values():
    resp, men, pres = build()
    assert pres.shape == (8, 3)
    assert pres["Aritzia"].sum() == 3 and pres["Zara"].sum() == 6 and pres["Uniqlo"].sum() == 3


def test_mention_rate_share_of_voice_and_rank():
    resp, men, pres = build()
    t = visibility_table(resp, pres, men, BRANDS, surface="s").set_index("brand")
    assert t.loc["Aritzia", "mention_rate"] == pytest.approx(3 / 8)
    assert t.loc["Zara", "mention_rate"] == pytest.approx(6 / 8)
    assert t.loc["Zara", "share_of_voice"] == pytest.approx(6 / 12)       # 12 tracked mentions in total
    assert t["share_of_voice"].sum() == pytest.approx(1.0)
    assert t.loc["Aritzia", "mean_rank"] == pytest.approx((1 + 2 + 1) / 3)
    assert t.loc["Zara", "top3_rate"] == pytest.approx(6 / 8)
    assert (t["mr_lo"] <= t["mention_rate"]).all() and (t["mention_rate"] <= t["mr_hi"]).all()


def test_stability_jaccard():
    resp, men, pres = build()
    s = stability(resp, men, pres, "Aritzia").set_index("prompt_id")
    assert s.loc["p1", "jaccard_all"] == 1.0                 # same brand set both runs
    assert s.loc["p2", "jaccard_all"] == pytest.approx(1 / 3)  # {Zara, Uniqlo} vs {Aritzia, Zara}
    assert s.loc["p1", "focus_rate"] == 1.0 and s.loc["p3", "focus_rate"] == 0.0


def test_focus_variance_is_mostly_between_prompts():
    resp, men, pres = build()
    v = focus_variance(resp, pres, "Aritzia", "s")
    assert v["icc"] > 0.5 and v["n_prompts"] == 4


def test_focus_by_group_runs():
    resp, men, pres = build()
    g = focus_by_group(resp, pres, "Aritzia", ["surface"])
    assert g.loc[0, "rate"] == pytest.approx(3 / 8)


def test_sentiment_and_agreement_and_sources():
    resp, men, pres = build()
    llm = pd.DataFrame([{"rid": "b|p1|s|0", "brand": "Aritzia", "known": True, "order": 1, "sentiment": "positive", "recommended": True},
                        {"rid": "b|p2|s|1", "brand": "Aritzia", "known": True, "order": 1, "sentiment": "negative", "recommended": False}])
    st = sentiment_table(resp, llm, "Aritzia", ["surface"])
    assert st.loc[0, "mean_sentiment"] == 0.0 and st.loc[0, "recommended"] == 0.5
    ag = extractor_agreement(resp, pres, llm, BRANDS)
    assert 0 <= ag["agree"] <= 1 and ag["per_brand"][0]["brand"] == "Aritzia"
    gr = resp.assign(grounded=True, sources=[["forbes.com", "www.forbes.com"], ["tiktok.com"]] + [[]] * 6)
    src = source_table(gr)
    assert src.iloc[0]["domain"] == "forbes.com" and src.iloc[0]["citations"] == 2 and src.iloc[0]["answers"] == 1


def test_rank_correlation_must_match_brands_by_name():
    """Regression test: two surfaces with opposite brand orders must not look perfectly correlated."""
    import pandas as pd
    from scipy.stats import spearmanr

    a = pd.Series({"Zara": 0.9, "Uniqlo": 0.5, "Aritzia": 0.1}).rename("a")
    b = pd.Series({"Aritzia": 0.9, "Uniqlo": 0.5, "Zara": 0.1}).rename("b")   # same brands, reversed popularity, listed in b's own order
    positional = spearmanr(a.sort_values(ascending=False), b.sort_values(ascending=False)).statistic
    matched = spearmanr(*[pd.concat([a, b], axis=1).fillna(0)[c] for c in ("a", "b")]).statistic
    assert positional == 1.0 and matched == -1.0
