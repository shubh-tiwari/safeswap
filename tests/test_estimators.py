import numpy as np
import pytest
from scipy import stats

from safeswap import estimators as est
from safeswap import sampler


def population(n=20_000, base=0.03, seed=0):
    """Errors concentrated where 'uncertainty' is high, like a real router."""
    rng = np.random.default_rng(seed)
    u = rng.beta(1, 4, n)
    p_err = np.clip(base * u / u.mean(), 0, 1)
    e = (rng.random(n) < p_err).astype(float)
    return e, u


def test_clopper_pearson_matches_scipy():
    k, n = 7, 200
    lo, hi = est.clopper_pearson(k, n, 0.05)
    ref = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    assert lo == pytest.approx(ref.low, abs=1e-9)
    assert hi == pytest.approx(ref.high, abs=1e-9)


def test_cp_upper_edges():
    assert est.cp_upper(0, 0) == 1.0
    assert est.cp_upper(5, 5) == 1.0
    # rule of three: 0 errors in n -> upper ~ 3/n at 95%
    assert est.cp_upper(0, 1000, 0.05) == pytest.approx(3 / 1000, rel=0.01)


def test_wilson_contains_point():
    lo, hi = est.wilson(10, 100)
    assert lo < 0.1 < hi


def test_required_n():
    assert est.required_n(0.02, 0.02) == -1
    n = est.required_n(0.02, 0.01)
    assert est.cp_upper(0.01 * n, n) <= 0.02
    assert est.cp_upper(0.01 * (n - 1), n - 1) > 0.02
    # zero true error needs ~ 3 / alpha labels
    assert est.required_n(0.02, 0.0) == pytest.approx(150, abs=2)


def test_horvitz_thompson_unbiased():
    e, u = population()
    p = sampler.active(u, rate=0.05, floor=0.01)
    rng = np.random.default_rng(1)
    vals = []
    for _ in range(300):
        s = sampler.draw(p, rng)
        vals.append(est.horvitz_thompson(e[s], p[s], e.size).value)
    assert np.mean(vals) == pytest.approx(e.mean(), rel=0.03)


@pytest.mark.parametrize("fn", [est.hajek_cp, est.hajek])
@pytest.mark.parametrize("mode", ["uniform", "active"])
def test_interval_coverage(fn, mode):
    e, u = population()
    p = sampler.uniform(e.size, 0.05) if mode == "uniform" else sampler.active(u, 0.05, 0.01)
    rng = np.random.default_rng(2)
    hits = 0
    sims = 400
    for _ in range(sims):
        s = sampler.draw(p, rng)
        r = fn(e[s], p[s], 0.05)
        hits += r.lo <= e.mean() <= r.hi
    # nominal 95%; allow Monte-Carlo slack (sd ~ 1.1% at 400 sims)
    assert hits / sims >= 0.92


def test_active_sampling_budget_and_floor():
    _, u = population()
    p = sampler.active(u, rate=0.05, floor=0.01)
    assert p.mean() == pytest.approx(0.05, rel=0.02)
    assert p.min() >= 0.01
    assert p.max() <= 1.0


def test_active_beats_uniform_when_errors_follow_uncertainty():
    e, u = population()
    rng = np.random.default_rng(3)
    widths = {}
    for mode, p in {
        "uniform": sampler.uniform(e.size, 0.05),
        "active": sampler.active(u, 0.05, 0.01),
    }.items():
        w = []
        for _ in range(200):
            s = sampler.draw(p, rng)
            r = est.hajek(e[s], p[s])
            w.append(r.hi - r.lo)
        widths[mode] = np.mean(w)
    assert widths["active"] < widths["uniform"]


def test_census_stratum_does_not_shrink_interval():
    """90% of traffic served by the reference (known error 0), 10% sampled at 3%."""
    rng = np.random.default_rng(4)
    n_census, n_rest = 9000, 1000
    e_rest = (rng.random(n_rest) < 0.1).astype(float)
    true = e_rest.sum() / (n_census + n_rest)
    p = np.full(n_rest, 0.03)
    hits, widths_naive, widths_census = 0, [], []
    for _ in range(400):
        s = sampler.draw(p, rng)
        r = est.with_census(est.hajek_cp, np.zeros(n_census), e_rest[s], p[s], n_rest)
        hits += r.lo <= true <= r.hi
        widths_census.append(r.hi - r.lo)
        naive = est.hajek_cp(
            np.concatenate([np.zeros(n_census), e_rest[s]]),
            np.concatenate([np.ones(n_census), p[s]]),
        )
        widths_naive.append(naive.hi - naive.lo)
    assert hits / 400 >= 0.92
    assert np.mean(widths_census) > 2 * np.mean(widths_naive)  # naive is overconfident


def test_learned_gate_learns_a_text_signal():
    import pandas as pd

    from safeswap.policies import LearnedGate

    rng = np.random.default_rng(5)
    n = 2000
    hard = rng.random(n) < 0.3
    prompt = np.where(hard, "prove the theorem about primes", "what colour is the sky")
    prompt = [f"{p} {i}" for i, p in enumerate(prompt)]
    small_score = np.where(hard, rng.random(n) < 0.2, rng.random(n) < 0.95).astype(float)
    df = pd.DataFrame(
        {"prompt": prompt, "score:s": small_score, "score:l": np.ones(n), "segment": "x"}
    )
    gate = LearnedGate("s", "l", threshold=0.3).fit(df.iloc[:1000])
    test = df.iloc[1000:].reset_index(drop=True)
    q = gate.predict_worse(test)
    assert q[hard[1000:]].mean() > q[~hard[1000:]].mean() + 0.4
    d = gate.choose(test)
    assert (d.model[hard[1000:]] == "l").mean() > 0.9


def test_fully_checked_cheap_answers_still_get_an_interval():
    import pandas as pd

    from safeswap.report import _error_estimate

    ev = pd.DataFrame(
        {
            "shadowed": True,
            "shadow_prob": 1.0,
            "by_reference": False,
            "error": [1.0] * 10 + [0.0] * 40,
        }
    )
    r = _error_estimate(ev, est.hajek_cp, 0.05)
    assert r.value == pytest.approx(0.2)
    assert r.lo < 0.12 and r.hi > 0.3  # sample of a rate, not a census
