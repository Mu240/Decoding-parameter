import numpy as np
import pandas as pd

from stats_utils import (cluster_mean_ci, cohen_kappa, collapse_point, gee_logit, holm, mcnemar_exact,
                         paired_cluster_bootstrap)


def test_holm_known_values():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_holm_keeps_nan():
    out = holm([0.01, np.nan])
    assert np.isnan(out[1]) and out[0] == 0.01


def test_cohen_kappa():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    assert abs(cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9  # chance-level agreement


def test_mcnemar():
    assert mcnemar_exact([1, 1, 0], [1, 1, 0]) == 1.0
    assert mcnemar_exact([1] * 10 + [0] * 2, [0] * 10 + [0] * 2) < 0.01


def test_cluster_ci_contains_mean():
    df = pd.DataFrame({"q": list(range(50)) * 2, "v": np.r_[np.ones(50), np.zeros(50)]})
    m, lo, hi, n = cluster_mean_ci(df, "v", "q")
    assert n == 50 and abs(m - 0.5) < 1e-9 and lo <= m <= hi


def test_paired_bootstrap_detects_shift():
    d = np.full(100, 0.2) + np.random.default_rng(0).normal(0, 0.05, 100)
    diff, lo, hi, p = paired_cluster_bootstrap(d)
    assert lo > 0 and p < 0.01
    _, _, _, p0 = paired_cluster_bootstrap(np.zeros(100))
    assert p0 >= 0.5


def test_collapse_point():
    assert collapse_point([0.0, 0.3, 0.7, 1.0], [0.6, 0.6, 0.58, 0.4], baseline_x=0.0, drop=0.05) == 1.0
    assert collapse_point([0.0, 0.3], [0.6, 0.6], baseline_x=0.0) is None


def test_gee_runs_and_recovers_direction():
    rng = np.random.default_rng(1)
    rows = []
    for q in range(80):
        for cfg, p in (("T0.0", 0.7), ("T1.6", 0.3)):
            rows.append({"q": f"q{q}", "config": cfg, "y": int(rng.random() < p)})
    out = gee_logit(pd.DataFrame(rows), "y", 'C(config, Treatment(reference="T0.0"))', "q")
    coef = out[out["term"].str.contains("T1.6")]["coef"].iloc[0]
    assert coef < 0
