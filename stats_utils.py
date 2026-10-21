"""Statistics: cluster bootstraps, paired contrasts, McNemar, Holm, GEE logistic regression, kappa."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_mean_ci(df: pd.DataFrame, metric: str, cluster: str, n_boot=1000, seed=0, alpha=0.05):
    """Mean of `metric` with a percentile bootstrap CI that resamples CLUSTERS (questions),
    because repeats/configs of one question are not independent observations."""
    per = df.groupby(cluster)[metric].mean().dropna().to_numpy(dtype=float)
    if len(per) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    boots = rng.choice(per, size=(n_boot, len(per)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(per.mean()), float(lo), float(hi), int(len(per))


def paired_cluster_bootstrap(d: np.ndarray, n_boot=2000, seed=0, alpha=0.05):
    """Paired difference (config A - config B) per cluster -> mean diff, CI, two-sided bootstrap p."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    p = max(min(p, 1.0), 1.0 / (n_boot + 1))
    return float(d.mean()), float(lo), float(hi), float(p)


def mcnemar_exact(a, b) -> float:
    """Exact McNemar p-value for paired binary outcomes a, b (arrays of 0/1)."""
    from scipy.stats import binomtest
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n = n10 + n01
    return 1.0 if n == 0 else float(binomtest(n10, n, 0.5).pvalue)


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaNs are left as NaN)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    order = idx[np.argsort(p[idx])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def cohen_kappa(y1, y2) -> float:
    y1, y2 = list(y1), list(y2)
    labels = sorted(set(y1) | set(y2))
    n = len(y1)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(y1, y2)) / n
    pe = sum((y1.count(l) / n) * (y2.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def gee_logit(df: pd.DataFrame, outcome: str, rhs: str, cluster: str) -> pd.DataFrame:
    """Logistic GEE with exchangeable correlation within question (a practical stand-in for a
    mixed-effects logit with a random question intercept). Returns coef, SE, p, odds ratio."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = df.dropna(subset=[outcome]).sort_values(cluster).copy()
    groups = pd.factorize(d[cluster])[0]
    res = smf.gee(f"{outcome} ~ {rhs}", groups=groups, data=d, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    out = pd.DataFrame({"coef": res.params, "se": res.bse, "p": res.pvalues})
    out["odds_ratio"] = np.exp(out["coef"])
    return out.reset_index().rename(columns={"index": "term"})


def collapse_point(xs, means, baseline_x=None, drop=0.05):
    """Lowest x whose mean is more than `drop` below the baseline mean (None if it never collapses)."""
    xs, means = list(xs), list(means)
    if not xs:
        return None
    base_i = xs.index(baseline_x) if baseline_x in xs else int(np.argmin(xs))
    base = means[base_i]
    for x, m in sorted(zip(xs, means)):
        if x > xs[base_i] and base - m > drop:
            return x
    return None
