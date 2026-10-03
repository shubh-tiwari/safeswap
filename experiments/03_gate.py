"""Phase 1b: a learned gate as router and uncertainty signal.

Trains P(small worse than reference | prompt text) on a 20% slice, then on the held-out 80%:
  * how well it ranks errors (AUROC, vs the segment prior that sees task names),
  * the savings / realised-error curve across thresholds, against the oracle.

    uv run python experiments/03_gate.py [--features tfidf|embed]
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from safeswap import policies
from safeswap.data import load_routerbench

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="tfidf")
ap.add_argument("--small", default="mistralai/mixtral-8x7b-chat")
a = ap.parse_args()

SMALL, LARGE = a.small, "gpt-4-1106-preview"
df = load_routerbench()
calib = df.sample(frac=0.2, random_state=0)
test = df.drop(calib.index).reset_index(drop=True)

y = (test[f"score:{SMALL}"] < test[f"score:{LARGE}"]).to_numpy()
gate = policies.LearnedGate(SMALL, LARGE, features=a.features).fit(calib)
q = gate.predict_worse(test)
seg = policies.SegmentPrior(SMALL, LARGE).fit(calib)
q_seg = 1 - test["segment"].map(seg.rate).fillna(seg.default).to_numpy()
print(f"held-out {len(test):,}, base error of always-small {y.mean():.3f}")
print(f"AUROC  learned gate ({a.features}): {roc_auc_score(y, q):.3f}")
print(f"AUROC  segment prior (sees task names): {roc_auc_score(y, q_seg):.3f}\n")

c_small, c_large = test[f"cost:{SMALL}"].to_numpy(), test[f"cost:{LARGE}"].to_numpy()
large_total = c_large.sum()
oracle_savings = 1 - np.where(y, c_large, c_small).sum() / large_total
rows = []
for t in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
    small = q <= t
    rows.append(
        {
            "threshold": t,
            "small_share": small.mean(),
            "savings": 1 - np.where(small, c_small, c_large).sum() / large_total,
            "realised_err": (small & y).mean(),
        }
    )
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\noracle savings at 0 error: {oracle_savings:.4f}")
