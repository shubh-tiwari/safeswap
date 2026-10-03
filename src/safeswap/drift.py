"""Drift detectors over a stream of windows. Each returns one statistic per window; an alarm is
`stat > threshold`, with thresholds calibrated on no-drift (control) runs to a target
false-alarm rate (see `calibrate_threshold`).

Two families, because they see different things:
  * label-based (needs shadow labels): Bernoulli CUSUM, window upper-CI test. The only ones that
    can see a change in *quality* when inputs look the same (e.g. a provider swapping the model).
  * label-free (free, instant): action-mix shift, input-embedding mean shift. See changes in
    *traffic*, which may or may not change quality.
"""

from __future__ import annotations

import numpy as np

from . import estimators as est


def cusum_step(s: float, labels, p0: float, p1: float) -> tuple[float, float]:
    """Advance Page's Bernoulli CUSUM (rate p0 -> p1) over `labels`; return (state, peak)."""
    if not 0 < p0 < p1 < 1:
        raise ValueError("need 0 < p0 < p1 < 1")
    up = np.log(p1 / p0)
    down = np.log((1 - p1) / (1 - p0))
    peak = s
    for e in labels:
        s = max(0.0, s + (up if e else down))
        peak = max(peak, s)
    return s, peak


def bernoulli_cusum(labels_per_window: list[np.ndarray], p0: float, p1: float) -> np.ndarray:
    """Page's CUSUM for a Bernoulli rate rising from p0 to p1, fed labels in arrival order.

    Returns the max CUSUM value reached inside each window. Labels must come from uniform
    sampling, so each is an unbiased draw of the served-small error rate.
    """
    s, out = 0.0, []
    for lab in labels_per_window:
        s, peak = cusum_step(s, lab, p0, p1)
        out.append(peak)
    return np.array(out)


def window_ci_excess(
    labels_per_window: list[np.ndarray],
    p_per_window: list[np.ndarray],
    budget: float,
    delta: float = 0.05,
) -> np.ndarray:
    """Lower confidence bound of each window's error minus the budget (> 0: confidently over)."""
    out = []
    for e, p in zip(labels_per_window, p_per_window):
        r = est.hajek_cp(e, p, delta) if len(e) else None
        out.append(-1.0 if r is None else r.lo - budget)
    return np.array(out)


def action_mix_z(small_counts: np.ndarray, totals: np.ndarray, p_ref: float) -> np.ndarray:
    """|z| of each window's small-model share against the reference share."""
    phat = small_counts / np.maximum(totals, 1)
    se = np.sqrt(p_ref * (1 - p_ref) / np.maximum(totals, 1))
    return np.abs(phat - p_ref) / se


def embedding_mean_shift(windows: list[np.ndarray], ref_mean: np.ndarray) -> np.ndarray:
    """Distance of each window's mean (normalised) embedding from the reference mean."""
    return np.array([np.linalg.norm(w.mean(0) - ref_mean) if len(w) else 0.0 for w in windows])


def calibrate_threshold(control_runs: list[np.ndarray], false_alarm: float = 0.05) -> float:
    """Threshold such that only `false_alarm` of no-drift runs ever exceed it."""
    peaks = np.array([r.max() for r in control_runs])
    return float(np.quantile(peaks, 1 - false_alarm))


def first_alarm(stat: np.ndarray, threshold: float) -> int:
    """Index of the first window whose statistic exceeds the threshold, or -1."""
    hit = np.flatnonzero(stat > threshold)
    return int(hit[0]) if hit.size else -1
