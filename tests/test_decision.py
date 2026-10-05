import numpy as np
import pandas as pd

from safeswap.replay.analysis import analyze
from safeswap.replay.decision import Policy, decide
from safeswap.stats import paired


def _res(w, t, l):
    return paired.compare(["win"] * w + ["tie"] * t + ["loss"] * l)


P = Policy(max_quality_drop_pp=1.0, min_samples=100, block_if_any_slice_drop_pp=5.0)


def test_hold_when_too_few_samples():
    assert decide(_res(0, 50, 0), [], None, P).verdict == "hold"


def test_ship_when_ci_lower_bound_above_allowed_drop():
    assert decide(_res(30, 2000, 30), [], None, P).verdict == "ship"


def test_rollback_when_clearly_worse():
    assert decide(_res(10, 200, 100), [], None, P).verdict == "rollback"


def test_hold_when_ci_too_wide():
    assert decide(_res(5, 180, 15), [], None, P).verdict == "hold"


def test_split_when_flagged_slice_and_rest_is_fine():
    v = decide(_res(30, 3500, 150), ["language: hi"], _res(20, 3000, 20), P)
    assert v.verdict == "split" and v.keep_on_baseline == ["language: hi"]


def _pairs(content, slice_vals):
    n = len(content)
    return pd.DataFrame(
        {
            "content": content,
            "format": "tie",
            "slice:language": slice_vals,
            "baseline_cost_usd": 0.004,
            "candidate_cost_usd": 0.0005,
            "judge_cost_usd": 0.001,
            "baseline_latency_ms": 1500.0,
            "candidate_latency_ms": 600.0,
            "identical": False,
            "judge_parsed": True,
            "judge_consistent": True,
        },
        index=range(n),
    )


def test_analyze_flags_bad_slice_and_splits():
    rng = np.random.default_rng(0)
    good = list(rng.choice(["win", "tie", "loss"], 2000, p=[0.02, 0.96, 0.02]))
    bad = list(rng.choice(["win", "tie", "loss"], 300, p=[0.0, 0.7, 0.3]))
    a = analyze(_pairs(good + bad, ["en"] * 2000 + ["hi"] * 300), P)
    flagged = [s.label for s in a.slices if s.flagged]
    assert flagged == ["hi"]
    assert a.verdict.verdict == "split"
    assert a.cost["change_pct"] < -80


def test_aa_false_alarm_rate():
    """Baseline vs itself (A/A): with symmetric noise, a non-ship verdict is a false alarm."""
    rng = np.random.default_rng(1)
    policy = Policy(max_quality_drop_pp=2.0, min_samples=500)
    alarms = 0
    for _ in range(200):
        content = rng.choice(["win", "tie", "loss"], 3000, p=[0.03, 0.94, 0.03])
        langs = rng.choice(["en", "hi", "ta", "es"], 3000)
        alarms += analyze(_pairs(list(content), list(langs)), policy).verdict.verdict != "ship"
    assert alarms / 200 <= 0.05
