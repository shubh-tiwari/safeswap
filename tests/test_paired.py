import numpy as np
import pytest
from scipy import stats

from safeswap.stats import paired


def _draw(rng, n, p_win, p_loss):
    u = rng.random(n)
    return np.where(u < p_win, "win", np.where(u < p_win + p_loss, "loss", "tie"))


@pytest.mark.parametrize(
    "n,p_win,p_loss",
    [(300, 0.10, 0.28), (100, 0.02, 0.05), (1000, 0.15, 0.15), (60, 0.0, 0.05)],
)
def test_tango_coverage(n, p_win, p_loss):
    rng = np.random.default_rng(0)
    true = p_win - p_loss
    hits = 0
    for _ in range(600):
        r = paired.compare(_draw(rng, n, p_win, p_loss))
        hits += r.delta_lo <= true <= r.delta_hi
    assert hits / 600 >= 0.93


def test_tango_edges():
    lo, hi = paired.tango_ci(0, 0, 50)
    assert lo < 0 < hi
    lo, hi = paired.tango_ci(0, 50, 50)
    assert lo == -1.0 and hi < -0.8
    lo, hi = paired.tango_ci(50, 0, 50)
    assert hi == 1.0 and lo > 0.8


def test_compare_counts_and_acceptability():
    r = paired.compare(["win"] * 10 + ["tie"] * 60 + ["loss"] * 30)
    assert (r.n, r.wins, r.ties, r.losses) == (100, 10, 60, 30)
    assert r.delta == pytest.approx(-0.2)
    assert r.acceptability == pytest.approx(0.7)
    assert r.accept_lo < 0.7 < r.accept_hi


def test_mcnemar_matches_scipy():
    assert paired.mcnemar_p(10, 30) == pytest.approx(stats.binomtest(10, 40, 0.5).pvalue)
    assert paired.mcnemar_p(0, 0) == 1.0


def test_compare_binary():
    base = [True, True, False, False]
    cand = [True, False, True, False]
    r = paired.compare_binary(base, cand)
    assert (r.wins, r.losses, r.ties) == (1, 1, 2)


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.6, 0.9])
    assert paired.benjamini_hochberg(p, 0.05).tolist() == [True, True, False, False, False, False]
    assert not paired.benjamini_hochberg(np.ones(5)).any()
