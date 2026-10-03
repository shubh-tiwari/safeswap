"""Phase 0: headroom. For each candidate small model vs the reference, on all of RouterBench:
always-small error and cost, and the oracle (cheapest model that is not worse) savings.

    uv run python experiments/01_oracle.py
"""

import pandas as pd

from safeswap.data import load_routerbench, routerbench_models

LARGE = "gpt-4-1106-preview"

df = load_routerbench()
large_cost = df[f"cost:{LARGE}"].sum()
rows = []
for m in routerbench_models(df):
    if m == LARGE:
        continue
    worse = df[f"score:{m}"] < df[f"score:{LARGE}"]
    oracle_cost = df[f"cost:{m}"].where(~worse, df[f"cost:{LARGE}"]).sum()
    rows.append(
        {
            "small": m,
            "always_small_err": worse.mean(),
            "always_small_savings": 1 - df[f"cost:{m}"].sum() / large_cost,
            "oracle_small_share": (~worse).mean(),
            "oracle_savings": 1 - oracle_cost / large_cost,
        }
    )
t = pd.DataFrame(rows).sort_values("oracle_savings", ascending=False)
print(f"{len(df):,} prompts, reference {LARGE}, always-large ${large_cost:.2f}\n")
print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
