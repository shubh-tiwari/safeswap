"""Phase 1 key experiment: do the intervals cover the true realised error at the nominal rate?

Policy: segment-prior or learned-gate router (Mixtral vs GPT-4) on RouterBench. For each shadow rate and sampling
mode, redraw the shadow sample many times and record coverage and mean width per estimator.

    uv run python experiments/02_coverage.py [--policy segment|gate] [--sims 300]
"""

import argparse

import numpy as np
import pandas as pd

from safeswap.data.routerbench import load_routerbench
from safeswap.monitor import runner as monitor
from safeswap.monitor.report import _error_estimate
from safeswap.routing import policies
from safeswap.stats import estimators as est

ap = argparse.ArgumentParser()
ap.add_argument("--sims", type=int, default=300)
ap.add_argument("--policy", choices=["segment", "gate"], default="segment")
ap.add_argument("--threshold", type=float, default=None)
a = ap.parse_args()

SMALL, LARGE = "mistralai/mixtral-8x7b-chat", "gpt-4-1106-preview"
df = load_routerbench()
calib = df.sample(frac=0.2, random_state=0)
df = df.drop(calib.index).reset_index(drop=True)
if a.policy == "segment":
    pol = policies.SegmentPrior(SMALL, LARGE, a.threshold or 0.8)
else:
    pol = policies.LearnedGate(SMALL, LARGE, a.threshold or 0.2)
dec = pol.fit(calib).choose(df)
print(f"{len(df):,} requests, small share {np.mean(dec.model == SMALL):.1%}")

rows = []
for rate in (0.01, 0.03, 0.05):
    for mode in ("uniform", "active"):
        cfg = {"rate": rate, "floor": rate / 3, "mode": mode}
        per = {k: {"hit": 0, "w": []} for k in est.ESTIMATORS}
        true = None
        for seed in range(a.sims):
            ev = monitor.replay_offline(df, dec, LARGE, cfg, seed)
            true = ev["true_error"].mean()
            for name, fn in est.ESTIMATORS.items():
                r = _error_estimate(ev, fn, 0.05)
                per[name]["hit"] += r.lo <= true <= r.hi
                per[name]["w"].append(r.hi - r.lo)
        labels = int((ev["shadowed"] & (ev["shadow_prob"] < 1)).sum())
        for name, v in per.items():
            rows.append(
                {
                    "rate": rate,
                    "mode": mode,
                    "labels~": labels,
                    "estimator": name,
                    "coverage": v["hit"] / a.sims,
                    "mean_width": np.mean(v["w"]),
                    "true_err": true,
                }
            )
t = pd.DataFrame(rows)
print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
