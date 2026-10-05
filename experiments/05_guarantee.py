"""Phase 3a: does the calibrated threshold keep its promise, and what does the promise cost?

Learned gate (Mixtral vs GPT-4). Held-out RouterBench split into a calibration pool and an
evaluation pool (the "future traffic"). Each repetition draws n labelled calibration requests,
chooses a threshold with each method, and checks the true end-to-end error on the evaluation pool.

    uv run python experiments/05_guarantee.py [--reps 500]
"""

import argparse

import numpy as np
import pandas as pd

from safeswap.data.routerbench import load_routerbench
from safeswap.routing import policies
from safeswap.stats import calibrate

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=500)
ap.add_argument("--delta", type=float, default=0.05)
a = ap.parse_args()

SMALL, LARGE = "mistralai/mixtral-8x7b-chat", "gpt-4-1106-preview"
df = load_routerbench()
train = df.sample(frac=0.2, random_state=0)
rest = df.drop(train.index).reset_index(drop=True)
gate = policies.LearnedGate(SMALL, LARGE).fit(train)
q = gate.predict_worse(rest)
worse = (rest[f"score:{SMALL}"] < rest[f"score:{LARGE}"]).to_numpy()
cs, cl = rest[f"cost:{SMALL}"].to_numpy(), rest[f"cost:{LARGE}"].to_numpy()

grid = np.linspace(0, 0.5, 101)
rng = np.random.default_rng(0)
in_cal = rng.random(len(rest)) < 0.5
L = calibrate.gate_losses(q, worse, grid)
C = calibrate.gate_costs(q, cs, cl, grid)
true_risk = L[~in_cal].mean(0)
savings = 1 - C[~in_cal].sum(0) / cl[~in_cal].sum()
L_cal = L[in_cal]
print(f"calibration pool {in_cal.sum():,}, evaluation pool {(~in_cal).sum():,}, delta={a.delta}\n")

rows = []
for alpha in (0.02, 0.05):
    ok = np.flatnonzero(true_risk <= alpha)
    best = ok.max()
    print(
        f"alpha={alpha}: best threshold in hindsight {grid[best]:.3f} -> savings "
        f"{savings[best]:.3f}, true error {true_risk[best]:.4f}"
    )
    for n in (300, 1000, 3000):
        for name, fn in calibrate.METHODS.items():
            viol, sav, none = [], [], 0
            for _ in range(a.reps):
                idx = rng.choice(L_cal.shape[0], n, replace=False)
                j = fn(L_cal[idx], alpha, a.delta)
                if j == calibrate.NONE:
                    none += 1
                    viol.append(False)
                    sav.append(0.0)
                    continue
                viol.append(true_risk[j] > alpha)
                sav.append(savings[j])
            rows.append(
                {
                    "alpha": alpha,
                    "n_labels": n,
                    "method": name,
                    "violation_rate": np.mean(viol),
                    "mean_savings": np.mean(sav),
                    "share_of_best": np.mean(sav) / savings[best],
                    "no_setting": none / a.reps,
                }
            )
print()
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
