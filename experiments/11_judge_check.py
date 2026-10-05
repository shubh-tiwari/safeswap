"""How far can the LLM judge be trusted? Paid (judge calls only; answers already exist).

Takes RouterBench pairs (Mixtral vs GPT-4 answers, both already in the dataset) where the answers
differ, balanced between "Mixtral scored lower" and "not lower", and asks the judge. Reports
agreement with the task-metric truth, order consistency and parse failures.

    uv run python experiments/11_judge_check.py --n 50 --cap 0.15
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from safeswap.data.routerbench import load_routerbench
from safeswap.llm.client import OpenRouter, SpendCapExceeded
from safeswap.scoring.judge import LLMJudge
from safeswap.stats.estimators import clopper_pearson

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--cap", type=float, default=0.15)
ap.add_argument("--judge", default="google/gemini-3.6-flash")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
load_dotenv(".env")

SMALL, LARGE = "mistralai/mixtral-8x7b-chat", "gpt-4-1106-preview"
df = load_routerbench()
diff = df[df[f"answer:{SMALL}"].str.strip() != df[f"answer:{LARGE}"].str.strip()]
truth = diff[f"score:{SMALL}"] < diff[f"score:{LARGE}"]
half = a.n // 2
pick = pd.concat(
    [
        diff[truth].sample(half, random_state=a.seed),
        diff[~truth].sample(a.n - half, random_state=a.seed),
    ]
).sample(frac=1, random_state=a.seed)

client = OpenRouter(spend_cap_usd=a.cap)
judge = LLMJudge(client, a.judge, max_tokens=1536)
rows = []
for i in range(len(pick)):
    row = pick.iloc[i]
    try:
        lab = judge.label(row["prompt"], row[f"answer:{SMALL}"], row[f"answer:{LARGE}"])
    except SpendCapExceeded as e:
        print(f"stopped: {e}")
        break
    rows.append(
        {
            "id": row["id"],
            "segment": row["segment"],
            "truth_worse": bool(row[f"score:{SMALL}"] < row[f"score:{LARGE}"]),
            "judge_worse": bool(lab.error),  # content verdict
            "cost_usd": lab.cost_usd,
            **lab.detail,
        }
    )
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(pick)} judged, spent ${client.spent_usd:.4f}")

res = pd.DataFrame(rows)
n = len(res)
agree = int((res.truth_worse == res.judge_worse).sum())
lo, hi = clopper_pearson(agree, n)
tp = int((res.truth_worse & res.judge_worse).sum())
fn = int((res.truth_worse & ~res.judge_worse).sum())
fp = int((~res.truth_worse & res.judge_worse).sum())
tn = int((~res.truth_worse & ~res.judge_worse).sum())
out_dir = Path("runs") / f"judge-check-{time.strftime('%Y%m%d-%H%M%S')}"
out_dir.mkdir(parents=True, exist_ok=True)
res.to_csv(out_dir / "labels.csv", index=False)
summary = {
    "judge": a.judge,
    "pairs": n,
    "agreement": agree / n,
    "agreement_ci": [lo, hi],
    "caught_when_worse": tp / max(tp + fn, 1),
    "false_alarm_when_not_worse": fp / max(fp + tn, 1),
    "order_consistent": float(res.consistent.mean()),
    "parse_ok": float(res.parsed.mean()),
    "format_worse": float(res.format_worse.mean()),
    "spent_usd": client.spent_usd,
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\njudge {a.judge} on {n} pairs (balanced: {res.truth_worse.sum()} truly worse)")
print(f"agreement with task truth: {agree}/{n} = {agree / n:.0%}  [95% CI {lo:.0%}-{hi:.0%}]")
print(f"  truly worse  -> judge said worse: {tp}/{tp + fn}")
print(f"  not worse    -> judge said worse: {fp}/{fp + tn}")
print(f"order-consistent verdicts: {res.consistent.mean():.0%}   parsed: {res.parsed.mean():.0%}")
print(f"format slips flagged (separately): {int(res.format_worse.sum())}/{n}")
print(f"spent ${client.spent_usd:.4f}\n-> {out_dir}")
