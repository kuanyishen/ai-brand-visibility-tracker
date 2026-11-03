"""Statistics for repeated-prompt designs.

The unit of analysis is the PROMPT, not the answer: repeated answers to the same prompt are far more alike than answers
to different prompts, so treating all answers as independent would give confidence intervals that are too narrow.
"""

from __future__ import annotations

import itertools

import numpy as np


def ratio_bootstrap(num, den, *, n_boot: int = 4000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Estimate sum(num)/sum(den) with a percentile CI from resampling whole prompts (clusters).

    ``num[i]`` and ``den[i]`` are the totals for prompt i (e.g. answers mentioning a brand / answers collected).
    """
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    est = num.sum() / den.sum() if den.sum() else float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = den[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(d > 0, num[idx].sum(axis=1) / d, np.nan)
    lo, hi = np.nanquantile(b, [alpha / 2, 1 - alpha / 2])
    return float(est), float(lo), float(hi)


def paired_diff_bootstrap(diffs, *, n_boot: int = 4000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of per-prompt differences with a percentile CI over prompts."""
    d = np.asarray(diffs, float)
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(b, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def sign_flip_test(diffs, *, n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided p-value for H0: the mean per-prompt difference is zero (differences symmetric around 0).

    Exact when at most 16 differences are non-zero, otherwise Monte Carlo.
    """
    d = np.asarray(diffs, float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    obs = abs(d.sum())
    if len(d) <= 16:
        signs = np.array(list(itertools.product((-1, 1), repeat=len(d))))
        sums = np.abs(signs @ d)
        return float(np.mean(sums >= obs - 1e-12))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, len(d)))
    return float((np.sum(np.abs(signs @ d) >= obs - 1e-12) + 1) / (n_perm + 1))


def variance_components(groups) -> tuple[float, float]:
    """One-way random-effects ANOVA estimates (between-group, within-group variance). Groups may differ in size."""
    groups = [np.asarray(g, float) for g in groups if len(g) > 0]
    k = len(groups)
    n = np.array([len(g) for g in groups])
    N = n.sum()
    grand = np.concatenate(groups).mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n**2).sum() / N) / (k - 1)
    return max((msb - msw) / n0, 0.0), float(msw)


def icc1(groups) -> float:
    """Share of total variance that lies BETWEEN prompts (0 = answers to one prompt are no more alike than random)."""
    sb, sw = variance_components(groups)
    return float(sb / (sb + sw)) if (sb + sw) > 0 else float("nan")


def se_of_mean(sigma_b2: float, sigma_w2: float, n_prompts: int, runs: int) -> float:
    """Standard error of the overall mean with ``n_prompts`` prompts and ``runs`` answers per prompt."""
    return float(np.sqrt(sigma_b2 / n_prompts + sigma_w2 / (n_prompts * runs)))


def runs_needed(sigma_b2: float, sigma_w2: float, n_prompts: int, target_se: float, max_runs: int = 1000) -> int | None:
    """Smallest runs-per-prompt that reaches ``target_se``; None if even unlimited runs cannot (prompt noise dominates)."""
    if np.sqrt(sigma_b2 / n_prompts) >= target_se:
        return None
    for r in range(1, max_runs + 1):
        if se_of_mean(sigma_b2, sigma_w2, n_prompts, r) <= target_se:
            return r
    return None


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def cohens_kappa(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float(np.mean(a == b))
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0
