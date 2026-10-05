"""Pick the cheapest policy setting whose end-to-end error is certifiably <= alpha.

Policies are indexed by a 1-D grid of settings lambda_0 < lambda_1 < ..., ordered from safest
(most traffic to the reference) to cheapest. `losses[i, j]` is 1 if request i would be served an
answer worse than the reference under setting j. A multi-threshold ladder (cache -> small -> large)
plugs in the same way once it is parameterised along a 1-D cost path.

Learn-then-Test (Angelopoulos et al. 2021): for each setting test H0_j: R(lambda_j) > alpha with an
exact binomial p-value on i.i.d. labelled calibration requests. Fixed-sequence testing walks the
grid from the safest setting and stops at the first non-rejection; it controls the family-wise
error at delta with no multiplicity penalty, because risk grows along the path. The chosen setting
satisfies P(R(lambda_hat) > alpha) <= delta.

Baselines for comparison:
  * naive: cheapest setting whose *empirical* risk is <= alpha (no guarantee).
  * bonferroni: LTT over the whole grid at delta / m (valid but conservative).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

NONE = -1  # no setting certified: fall back to always-reference


def binom_pvalues(losses: np.ndarray, alpha: float) -> np.ndarray:
    """p-value of H0: R > alpha for each setting, from k errors in n requests."""
    n = losses.shape[0]
    k = losses.sum(0)
    return stats.binom.cdf(k, n, alpha)


def fixed_sequence(losses: np.ndarray, alpha: float, delta: float) -> int:
    p = binom_pvalues(losses, alpha)
    j = NONE
    for i, pv in enumerate(p):
        if pv > delta:
            break
        j = i
    return j


def bonferroni(losses: np.ndarray, alpha: float, delta: float) -> int:
    p = binom_pvalues(losses, alpha)
    ok = np.flatnonzero(p <= delta / len(p))
    return int(ok.max()) if ok.size else NONE


def naive(losses: np.ndarray, alpha: float, delta: float | None = None) -> int:
    ok = np.flatnonzero(losses.mean(0) <= alpha)
    return int(ok.max()) if ok.size else NONE


METHODS = {"fixed_sequence": fixed_sequence, "bonferroni": bonferroni, "naive": naive}


def gate_losses(q: np.ndarray, worse: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Loss matrix for a threshold gate: request i is served small at setting j iff q_i <= grid_j,
    and is an error iff it is served small and the small answer is worse."""
    return ((q[:, None] <= grid[None, :]) & worse[:, None]).astype(np.int8)


def gate_costs(q: np.ndarray, c_small: np.ndarray, c_large: np.ndarray, grid: np.ndarray):
    small = q[:, None] <= grid[None, :]
    return np.where(small, c_small[:, None], c_large[:, None])
