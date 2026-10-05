"""Shadow sampling: decide which served requests also get a reference answer and a judge label.

Every request gets an inclusion probability p_i, recorded alongside it; estimators reweight by
1 / p_i. A uniform floor keeps every request reachable (p_i > 0), which is what makes the
IPW estimates unbiased. On top of the floor, "active" sampling spends more of the label budget
where the policy is uncertain.
"""

from __future__ import annotations

import numpy as np


def uniform(n: int, rate: float) -> np.ndarray:
    return np.full(n, float(rate))


def active(uncertainty, rate: float, floor: float, power: float = 0.5) -> np.ndarray:
    """Probabilities proportional to uncertainty**power, with mean ~= rate and every p_i >= floor.

    `rate` is the overall label budget as a fraction of traffic. `floor` is the guaranteed
    uniform share; the remaining (rate - floor) is spread in proportion to uncertainty**power.
    power = 0.5 is the variance-optimal (Neyman-style) allocation for an IPW mean of 0/1 labels
    when `uncertainty` is P(error): p_i ∝ sqrt(E[e_i^2]).
    """
    u = np.clip(np.asarray(uncertainty, dtype=float), 0.0, None) ** power
    if not 0 < floor <= rate <= 1:
        raise ValueError("need 0 < floor <= rate <= 1")
    extra = rate - floor
    if u.sum() == 0 or extra == 0:
        return uniform(u.size, rate)
    p = floor + extra * u * u.size / u.sum()
    for _ in range(20):  # clip at 1 and redistribute the excess
        over = p > 1
        if not over.any():
            break
        excess = (p[over] - 1).sum()
        p[over] = 1.0
        free = ~over
        if not free.any():
            break
        share = (
            u[free] / u[free].sum() if u[free].sum() > 0 else np.full(free.sum(), 1 / free.sum())
        )
        p[free] += excess * share
    return np.clip(p, floor, 1.0)


def draw(p, rng: np.random.Generator) -> np.ndarray:
    """Independent Bernoulli(p_i) inclusion (Poisson sampling)."""
    p = np.asarray(p, dtype=float)
    return rng.random(p.size) < p
