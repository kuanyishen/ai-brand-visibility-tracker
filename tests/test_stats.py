import numpy as np
import pytest

from brandvis.stats import (cohens_kappa, icc1, jaccard, paired_diff_bootstrap, ratio_bootstrap, runs_needed,
                            se_of_mean, sign_flip_test, variance_components)


def test_ratio_bootstrap_point_estimate_and_coverage():
    rng = np.random.default_rng(0)
    covered = 0
    for s in range(200):
        p_prompt = rng.beta(2, 5, size=30)                  # prompt-level mention probabilities, mean 2/7
        den = np.full(30, 3)
        num = rng.binomial(den, p_prompt)
        est, lo, hi = ratio_bootstrap(num, den, n_boot=400, seed=s)
        covered += lo <= 2 / 7 <= hi
    assert covered / 200 > 0.88                             # nominal 95%; small-sample percentile CIs run a bit low


def test_clustering_widens_intervals_versus_naive():
    # all answers to a prompt identical: 3 runs add no information
    num, den = np.array([3, 3, 0, 0, 3, 0, 0, 0]), np.full(8, 3)
    _, lo, hi = ratio_bootstrap(num, den, seed=1)
    naive_half = 1.96 * np.sqrt(0.375 * 0.625 / 24)
    assert (hi - lo) / 2 > naive_half


def test_sign_flip_exact_and_symmetry():
    assert sign_flip_test([0.1] * 10) < 0.01                # consistent positive difference
    assert sign_flip_test([0.1, -0.1, 0.1, -0.1]) == pytest.approx(1.0)
    assert sign_flip_test([0, 0, 0]) == 1.0
    big = sign_flip_test(np.r_[np.full(20, 0.05), -0.05], seed=3)  # Monte Carlo branch
    assert big < 0.001


def test_paired_bootstrap_ci_contains_mean():
    d = np.random.default_rng(2).normal(0.1, 0.2, 40)
    m, lo, hi = paired_diff_bootstrap(d)
    assert lo < m < hi and m == pytest.approx(d.mean())


def test_icc_recovers_known_variance_components():
    rng = np.random.default_rng(0)
    groups = [rng.normal(mu, 1.0, size=8) for mu in rng.normal(0, 2.0, size=400)]   # true ICC = 4/(4+1) = 0.8
    assert icc1(groups) == pytest.approx(0.8, abs=0.03)
    sb, sw = variance_components(groups)
    assert sb == pytest.approx(4.0, rel=0.15) and sw == pytest.approx(1.0, rel=0.1)


def test_icc_is_zero_for_pure_noise():
    rng = np.random.default_rng(1)
    assert icc1([rng.normal(size=5) for _ in range(300)]) < 0.03


def test_runs_needed_diminishing_returns():
    # between-prompt variance sets a floor that more runs cannot remove
    assert se_of_mean(0.04, 0.2, 40, 1) > se_of_mean(0.04, 0.2, 40, 10) > se_of_mean(0.04, 0.2, 40, 1000)
    assert runs_needed(0.04, 0.2, 40, target_se=0.01) is None
    r = runs_needed(0.01, 0.2, 40, target_se=0.03)
    assert r is not None and se_of_mean(0.01, 0.2, 40, r) <= 0.03 < se_of_mean(0.01, 0.2, 40, r - 1)


def test_jaccard_and_kappa():
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard([], []) == 1.0
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0


def test_prompts_needed_inverts_se_formula():
    from brandvis.stats import prompts_needed

    p = prompts_needed(0.14, 0.11, runs=3, target_se=0.05)
    assert se_of_mean(0.14, 0.11, p, 3) <= 0.05 < se_of_mean(0.14, 0.11, p - 1, 3)
    # for a fixed target, more runs lower the prompt count only slightly: prompts, not runs, are the bottleneck
    assert prompts_needed(0.14, 0.11, 30, 0.05) > 0.8 * prompts_needed(0.14, 0.11, 3, 0.05) * 0.5
