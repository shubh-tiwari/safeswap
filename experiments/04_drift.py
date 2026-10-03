"""Phase 2: how fast do monitors notice drift, and which ones can see it at all?

Stream: 40 windows x 500 requests from held-out RouterBench, shift at window 20. Policy: learned
gate (Mixtral vs GPT-4, threshold 0.2), fixed throughout. Uniform shadowing of small-served
requests. Scenarios:
  control    no change (used to set every detector's threshold to a 5% false-alarm rate per run)
  mix        task-mix shift: segments where Mixtral is weak become 4x more frequent
  swap       the 'small' provider silently serves Mistral-7B instead of Mixtral; inputs and
             router are unchanged

    uv run python experiments/04_drift.py [--sims 200] [--rate 0.03]
"""

import argparse

import numpy as np
import pandas as pd

from safeswap import drift, policies
from safeswap.data import load_routerbench
from safeswap.features import Embed

ap = argparse.ArgumentParser()
ap.add_argument("--sims", type=int, default=200)
ap.add_argument("--rate", type=float, default=0.03)
ap.add_argument("--windows", type=int, default=40)
ap.add_argument("--size", type=int, default=500)
a = ap.parse_args()

SMALL, SWAP, LARGE, THRESH = (
    "mistralai/mixtral-8x7b-chat",
    "mistralai/mistral-7b-chat",
    "gpt-4-1106-preview",
    0.2,
)
SHIFT = a.windows // 2

df = load_routerbench()
calib = df.sample(frac=0.2, random_state=0)
test = df.drop(calib.index).reset_index(drop=True)
gate = policies.LearnedGate(SMALL, LARGE, THRESH).fit(calib)
q = gate.predict_worse(test)
small = q <= THRESH
worse = {m: (test[f"score:{m}"] < test[f"score:{LARGE}"]).to_numpy() for m in (SMALL, SWAP)}
emb = Embed().transform(test["prompt"])  # cached from experiment 03
ref_mean = Embed().transform(calib["prompt"]).mean(0)

# Reference split: what you'd know at deploy time from a one-off fully labelled evaluation.
rng0 = np.random.default_rng(123)
is_ref = rng0.random(len(test)) < 0.3
ref_idx, pool = np.flatnonzero(is_ref), np.flatnonzero(~is_ref)
p0 = worse[SMALL][ref_idx][small[ref_idx]].mean()
small_share_ref = small[ref_idx].mean()
budget = 1.5 * p0  # alarm-worthy: error among small-served 50% above baseline
seg_err = pd.Series(worse[SMALL][ref_idx]).groupby(test["segment"].to_numpy()[ref_idx]).mean()
weak = set(seg_err[seg_err > seg_err.median()].index)
w_mix = np.where(test["segment"].isin(weak).to_numpy()[pool], 4.0, 1.0)
w_mix /= w_mix.sum()
print(
    f"baseline: small share {small_share_ref:.3f}, error among small-served p0={p0:.3f}; "
    f"swap error among small-served {worse[SWAP][ref_idx][small[ref_idx]].mean():.3f}"
)


def stream(scenario: str, rng: np.random.Generator):
    n = a.windows * a.size
    pre = rng.choice(pool, SHIFT * a.size)
    if scenario == "mix":
        post = rng.choice(pool, n - pre.size, p=w_mix)
    else:
        post = rng.choice(pool, n - pre.size)
    idx = np.concatenate([pre, post])
    after = np.arange(n) >= pre.size
    err = np.where(after & (scenario == "swap"), worse[SWAP][idx], worse[SMALL][idx])
    sm = small[idx]
    shadow = sm & (rng.random(n) < a.rate)
    w = np.arange(n) // a.size
    labels = [err[(w == i) & shadow].astype(float) for i in range(a.windows)]
    probs = [np.full(len(lab), a.rate) for lab in labels]
    stats = {
        "label CUSUM": drift.bernoulli_cusum(labels, p0, 2 * p0),
        "window CI > budget": drift.window_ci_excess(labels, probs, budget),
        "action-mix |z|": drift.action_mix_z(
            np.bincount(w, sm, a.windows), np.bincount(w, minlength=a.windows), small_share_ref
        ),
        "input embedding shift": drift.embedding_mean_shift(
            [emb[idx[w == i]] for i in range(a.windows)], ref_mean
        ),
    }
    true_err = np.array([(err & sm)[w == i].mean() for i in range(a.windows)])
    return stats, true_err


runs = {
    s: [stream(s, np.random.default_rng(1000 * k + seed)) for seed in range(a.sims)]
    for k, s in enumerate(("control", "mix", "swap"))
}
detectors = list(runs["control"][0][0])
thr = {d: drift.calibrate_threshold([r[0][d] for r in runs["control"]], 0.05) for d in detectors}

for s in ("mix", "swap"):
    te = np.mean([r[1] for r in runs[s]], 0)
    print(
        f"\n{s}: true error (all traffic) before {te[:SHIFT].mean():.4f} -> after "
        f"{te[SHIFT:].mean():.4f}"
    )
rows = []
for d in detectors:
    row = {"detector": d}
    for s in ("control", "mix", "swap"):
        firsts = np.array([drift.first_alarm(r[0][d], thr[d]) for r in runs[s]])
        if s == "control":
            row["false_alarm/run"] = np.mean(firsts >= 0)
            continue
        early = firsts.__lt__(SHIFT) & (firsts >= 0)
        caught = firsts >= SHIFT
        row[f"{s}: detected"] = caught.mean()
        row[f"{s}: median delay (windows)"] = (
            np.median(firsts[caught] - SHIFT) if caught.any() else np.nan
        )
        row[f"{s}: false alarm before shift"] = early.mean()
    rows.append(row)
labels_per_window = a.size * small_share_ref * a.rate
print(
    f"\nshadow rate {a.rate:.0%} of small-served -> ~{labels_per_window:.1f} labels per window "
    f"of {a.size} requests\n"
)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
