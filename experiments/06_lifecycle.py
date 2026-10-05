"""Phase 3b: the whole loop under a silent model swap.

  calibrate (fixed-sequence LTT, alpha, delta) -> serve -> shadow 3% of small-served -> CUSUM
  -> on alarm: fall back to always-large and collect recalibration labels (run the small model
     on a share of fallback traffic and label it) -> recalibrate -> resume.

Compared against the same calibrated policy with no monitoring. Population risk of the setting in
effect is computed exactly on the evaluation pool for the current regime (pre/post swap), so the
timeline shows the real error, not a noisy window estimate.

    uv run python experiments/06_lifecycle.py [--runs 200]
"""

import argparse

import numpy as np
import pandas as pd

from safeswap.data.routerbench import load_routerbench
from safeswap.routing import policies
from safeswap.stats import calibrate, drift

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=200)
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--delta", type=float, default=0.05)
ap.add_argument("--n_cal", type=int, default=3000)
ap.add_argument("--n_recal", type=int, default=2000)
ap.add_argument("--recal_rate", type=float, default=0.5)
ap.add_argument("--shadow", type=float, default=0.03)
ap.add_argument("--cusum_h", type=float, default=None, help="default: calibrated on no-swap runs")
a = ap.parse_args()

SMALL, SWAP, LARGE = (
    "mistralai/mixtral-8x7b-chat",
    "mistralai/mistral-7b-chat",
    "gpt-4-1106-preview",
)
WINDOWS, SIZE, SHIFT = 60, 500, 20

df = load_routerbench()
train = df.sample(frac=0.2, random_state=0)
rest = df.drop(train.index).reset_index(drop=True)
q = policies.LearnedGate(SMALL, LARGE).fit(train).predict_worse(rest)
worse = {m: (rest[f"score:{m}"] < rest[f"score:{LARGE}"]).to_numpy() for m in (SMALL, SWAP)}
cl = rest[f"cost:{LARGE}"].to_numpy()
cost_small = {m: rest[f"cost:{m}"].to_numpy() for m in (SMALL, SWAP)}

grid = np.linspace(0, 0.5, 101)
rng0 = np.random.default_rng(0)
in_cal = rng0.random(len(rest)) < 0.5
cal_pool, ev = np.flatnonzero(in_cal), np.flatnonzero(~in_cal)
L = {m: calibrate.gate_losses(q, worse[m], grid) for m in worse}
risk = {m: L[m][ev].mean(0) for m in worse}  # exact population risk per setting and regime
sav = {
    m: 1 - calibrate.gate_costs(q[ev], cost_small[m][ev], cl[ev], grid).sum(0) / cl[ev].sum()
    for m in worse
}


def p0_of(j: int, labels: np.ndarray, served_small: np.ndarray) -> float:
    """Calibration-time error rate among small-served requests at setting j (CUSUM baseline)."""
    s = served_small[:, j]
    return float(np.clip(labels[s].mean() if s.any() else 0.05, 0.01, 0.45))


def run(seed: int, monitor: bool, swap: bool, h: float):
    rng = np.random.default_rng(seed)
    idx = rng.choice(cal_pool, a.n_cal, replace=False)
    j = calibrate.fixed_sequence(L[SMALL][idx], a.alpha, a.delta)
    p0 = p0_of(j, worse[SMALL][idx], q[idx][:, None] <= grid[None, :])
    state, recal, cusum_s, peak_trace = "serve", [], 0.0, []
    timeline = []
    for w in range(WINDOWS):
        regime = SWAP if (swap and w >= SHIFT) else SMALL
        reqs = rng.choice(ev, SIZE)
        if state == "fallback":
            timeline.append((w, regime, calibrate.NONE, state))
            take = reqs[rng.random(SIZE) < a.recal_rate]
            recal.extend(take.tolist())
            if len(recal) >= a.n_recal:
                r = np.array(recal[: a.n_recal])
                j = calibrate.fixed_sequence(L[regime][r], a.alpha, a.delta)
                p0 = p0_of(j, worse[regime][r], q[r][:, None] <= grid[None, :])
                state, recal, cusum_s = "serve", [], 0.0
            continue
        timeline.append((w, regime, j, state))
        if not monitor or j == calibrate.NONE:
            continue
        small = reqs[q[reqs] <= grid[j]]
        labels = worse[regime][small[rng.random(small.size) < a.shadow]]
        cusum_s, top = drift.cusum_step(cusum_s, labels, p0, min(2 * p0, 0.9))
        peak_trace.append(top)
        if top > h:
            state = "fallback"
    return timeline, peak_trace


def summarise(timelines):
    rows = []
    for tl in timelines:
        for w, regime, j, state in tl:
            r = 0.0 if j == calibrate.NONE else risk[regime][j]
            s = 0.0 if j == calibrate.NONE else sav[regime][j]
            rows.append({"window": w, "risk": r, "savings": s, "fallback": state == "fallback"})
    return pd.DataFrame(rows)


# CUSUM threshold: 5% of no-swap runs ever alarm over the 60 windows.
if a.cusum_h is None:
    peaks = [max(run(10_000 + s, True, False, np.inf)[1] or [0]) for s in range(a.runs)]
    a.cusum_h = float(np.quantile(peaks, 0.95))
print(
    f"alpha={a.alpha}, delta={a.delta}, n_cal={a.n_cal}, n_recal={a.n_recal}, "
    f"shadow={a.shadow:.0%}, CUSUM h={a.cusum_h:.2f}"
)
print(
    f"regime risk at best-in-hindsight setting: pre {risk[SMALL][risk[SMALL] <= a.alpha].max():.4f}"
)

res = {}
for name, monitor in (("monitored", True), ("unmonitored", False)):
    tls = [run(s, monitor, True, a.cusum_h)[0] for s in range(a.runs)]
    res[name] = summarise(tls)

ctrl = summarise([run(20_000 + s, True, False, a.cusum_h)[0] for s in range(a.runs)])
print(
    f"no-swap control: runs ever in fallback {ctrl.groupby(ctrl.index // WINDOWS)['fallback'].any().mean():.1%}"
)

phases = [
    ("before swap", 0, SHIFT),
    ("swap +0-4", SHIFT, SHIFT + 5),
    ("swap +5-14", SHIFT + 5, SHIFT + 15),
    ("swap +15-39", SHIFT + 15, WINDOWS),
]
rows = []
for name, d in res.items():
    for label, lo, hi in phases:
        g = d[(d.window >= lo) & (d.window < hi)]
        rows.append(
            {
                "policy": name,
                "period": label,
                "mean risk": g.risk.mean(),
                "P(risk > alpha)": (g.risk > a.alpha).mean(),
                "savings": g.savings.mean(),
                "in fallback": g.fallback.mean(),
            }
        )
print()
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
