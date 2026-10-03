"""Estimators for realised error on live traffic, from a sampled (shadowed) subset.

Setting: N requests were served by some policy. Each request i was shadowed (sent to the
reference model and judged) independently with known probability p_i, giving a label
e_i in {0, 1} (1 = served answer worse than the reference). Everything here is a pure
function of the sampled labels and their probabilities.

  * Horvitz–Thompson (IPW): unbiased for the population mean, sum(e_i / p_i) / N.
  * Hájek: ratio form sum(e_i / p_i) / sum(1 / p_i); slightly biased, usually lower variance,
    and needs no N.
  * Intervals: normal (fails for rare errors), and Clopper–Pearson on the Kish effective
    sample size (conservative, behaves at error rates near zero).
  * Exact binomial bounds (Clopper–Pearson, Wilson) for uniformly sampled or fully labelled sets.
  * required_n: labels needed before an upper bound can certify error <= alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Estimate:
    value: float
    lo: float
    hi: float
    n_labels: int
    n_eff: float
    method: str


def _arrays(e, p) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(e, dtype=float)
    p = np.asarray(p, dtype=float)
    if e.shape != p.shape:
        raise ValueError("e and p must have the same shape")
    if np.any((p <= 0) | (p > 1)):
        raise ValueError("sampling probabilities must be in (0, 1]")
    return e, p


def clopper_pearson(k: int, n: int, delta: float = 0.05) -> tuple[float, float]:
    """Two-sided exact interval with coverage >= 1 - delta."""
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else stats.beta.ppf(delta / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - delta / 2, k + 1, n - k)
    return float(lo), float(hi)


def cp_upper(k: float, n: float, delta: float = 0.05) -> float:
    """One-sided Clopper–Pearson upper bound: P(true rate <= bound) >= 1 - delta.

    Accepts non-integer k and n (used with effective sample sizes).
    """
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(stats.beta.ppf(1 - delta, k + 1, n - k))


def wilson(k: int, n: int, delta: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z = stats.norm.ppf(1 - delta / 2)
    ph = k / n
    den = 1 + z**2 / n
    mid = (ph + z**2 / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / den
    return float(max(0.0, mid - half)), float(min(1.0, mid + half))


def kish_n_eff(p) -> float:
    w = 1.0 / np.asarray(p, dtype=float)
    return float(w.sum() ** 2 / (w**2).sum()) if w.size else 0.0


def horvitz_thompson(e, p, n_population: int, delta: float = 0.05) -> Estimate:
    """Unbiased IPW mean with a normal interval (Poisson-sampling variance)."""
    e, p = _arrays(e, p)
    est = float((e / p).sum() / n_population)
    var = float(((1 - p) / p**2 * e**2).sum() / n_population**2)
    z = stats.norm.ppf(1 - delta / 2)
    half = z * np.sqrt(var)
    return Estimate(est, max(0.0, est - half), min(1.0, est + half), e.size, kish_n_eff(p), "ht")


def hajek(e, p, delta: float = 0.05) -> Estimate:
    """Ratio IPW mean with a linearised normal interval."""
    e, p = _arrays(e, p)
    if e.size == 0:
        return Estimate(float("nan"), 0.0, 1.0, 0, 0.0, "hajek")
    w = 1.0 / p
    est = float((w * e).sum() / w.sum())
    r = e - est
    var = float(((1 - p) * w**2 * r**2).sum() / w.sum() ** 2)
    z = stats.norm.ppf(1 - delta / 2)
    half = z * np.sqrt(var)
    return Estimate(est, max(0.0, est - half), min(1.0, est + half), e.size, kish_n_eff(p), "hajek")


def hajek_cp(e, p, delta: float = 0.05) -> Estimate:
    """Hájek point estimate with a Clopper–Pearson interval on the Kish effective sample size.

    Conservative and well behaved when errors are rare, where normal intervals collapse to
    zero width (no observed errors -> variance estimate 0).
    """
    e, p = _arrays(e, p)
    if e.size == 0:
        return Estimate(float("nan"), 0.0, 1.0, 0, 0.0, "hajek_cp")
    w = 1.0 / p
    est = float((w * e).sum() / w.sum())
    n_eff = kish_n_eff(p)
    k_eff = est * n_eff
    lo = 0.0 if k_eff <= 0 else float(stats.beta.ppf(delta / 2, k_eff, n_eff - k_eff + 1))
    hi = cp_upper(k_eff, n_eff, delta / 2)
    return Estimate(est, lo, hi, e.size, n_eff, "hajek_cp")


def bootstrap(e, p, delta: float = 0.05, reps: int = 2000, seed: int = 0) -> Estimate:
    """Percentile bootstrap of the Hájek estimate (resampling labelled requests)."""
    e, p = _arrays(e, p)
    if e.size == 0:
        return Estimate(float("nan"), 0.0, 1.0, 0, 0.0, "bootstrap")
    rng = np.random.default_rng(seed)
    w = 1.0 / p
    idx = rng.integers(0, e.size, size=(reps, e.size))
    ws, es = w[idx], e[idx]
    boots = (ws * es).sum(1) / ws.sum(1)
    est = float((w * e).sum() / w.sum())
    lo, hi = np.quantile(boots, [delta / 2, 1 - delta / 2])
    return Estimate(est, float(lo), float(hi), e.size, kish_n_eff(p), "bootstrap")


ESTIMATORS = {"hajek": hajek, "hajek_cp": hajek_cp, "bootstrap": bootstrap}


def required_n(alpha: float, true_err: float, delta: float = 0.05, n_max: int = 10_000_000) -> int:
    """Smallest n at which the one-sided CP upper bound on true_err * n errors is <= alpha.

    "How many uniformly sampled labels before you can certify error <= alpha, if the policy's
    real error is true_err?" Uses the expected error count (non-integer), which keeps the
    bound monotone in n; real counts fluctuate around it. Returns -1 when true_err >= alpha.
    """
    if true_err >= alpha:
        return -1

    def ok(n: int) -> bool:
        return cp_upper(true_err * n, n, delta) <= alpha

    hi = 1
    while not ok(hi):
        hi *= 2
        if hi > n_max:
            return -1
    lo = hi // 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def with_census(
    fn, e_census, e_sampled, p_sampled, n_sampled_pop: int, delta: float = 0.05
) -> Estimate:
    """Combine a fully observed stratum (p = 1, e.g. requests the reference model served itself)
    with an estimate over the sampled stratum.

    Folding p = 1 rows into a weighted estimator inflates the effective sample size with labels
    that carry no uncertainty, which makes intervals look far tighter than the evidence behind
    the sampled stratum supports. Here only the sampled stratum is estimated, and its interval is
    scaled by its share of traffic.
    """
    e_census = np.asarray(e_census, dtype=float)
    n_total = e_census.size + n_sampled_pop
    if n_sampled_pop == 0:
        v = float(e_census.mean()) if e_census.size else float("nan")
        return Estimate(v, v, v, 0, 0.0, f"{fn.__name__}+census")
    r = fn(e_sampled, p_sampled, delta)
    share = n_sampled_pop / n_total
    base = e_census.sum() / n_total
    return Estimate(
        float(base + share * r.value),
        float(base + share * r.lo),
        float(base + share * r.hi),
        r.n_labels,
        r.n_eff,
        f"{fn.__name__}+census",
    )
