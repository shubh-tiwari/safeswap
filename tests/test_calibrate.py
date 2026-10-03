import numpy as np

from safeswap import calibrate


def _losses(n, risks, rng):
    """Nested losses with increasing risk along the grid (like a threshold path)."""
    u = rng.random(n)
    return (u[:, None] < np.asarray(risks)[None, :]).astype(np.int8)


RISKS = np.linspace(0, 0.1, 41)


def test_fixed_sequence_controls_violation_rate():
    rng = np.random.default_rng(0)
    alpha, delta = 0.05, 0.1
    viol = []
    for _ in range(500):
        j = calibrate.fixed_sequence(_losses(1000, RISKS, rng), alpha, delta)
        viol.append(j != calibrate.NONE and RISKS[j] > alpha)
    assert np.mean(viol) <= delta + 0.02


def test_naive_violates_often():
    rng = np.random.default_rng(1)
    viol = [RISKS[calibrate.naive(_losses(1000, RISKS, rng), 0.05)] > 0.05 for _ in range(300)]
    assert np.mean(viol) > 0.2


def test_none_when_nothing_certifiable():
    rng = np.random.default_rng(2)
    L = _losses(50, np.full(5, 0.2), rng)
    assert calibrate.fixed_sequence(L, 0.05, 0.05) == calibrate.NONE
    assert calibrate.bonferroni(L, 0.05, 0.05) == calibrate.NONE


def test_gate_losses_nested():
    q = np.array([0.05, 0.15, 0.3])
    worse = np.array([True, True, False])
    L = calibrate.gate_losses(q, worse, np.array([0.0, 0.1, 0.2, 0.5]))
    assert L.tolist() == [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]]
