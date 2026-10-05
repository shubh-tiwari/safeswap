"""Charts for the README from the latest 300-prompt live run (reads runs/, no API calls).

uv run python experiments/13_live_charts.py
"""

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from safeswap.data.routerbench import load_routerbench
from safeswap.stats.estimators import clopper_pearson

OUT = Path("docs/img")
OUT.mkdir(parents=True, exist_ok=True)
run = max(glob.glob("runs/live-300-*"))
ev = pd.read_parquet(f"{run}/events.parquet")
cfg = yaml.safe_load(Path(f"{run}/config.yaml").read_text())
df = load_routerbench(seed=cfg["seed"], mix=cfg["dataset"]["mix"]).set_index("id")
ev["mc"] = ev["request_id"].map(
    df["prompt"].str.contains("Print only a single choice", regex=False)
)

NAMES = {
    "grade-school-math": "Math (GSM8K)",
    "mtbench": "MT-Bench",
    "mbpp": "Code (MBPP)",
    "bias_detection": "Bias detection",
    "consensus_summary": "Summaries",
    "abstract2title": "Paper titles*",
}
groups = [("All 300", ev), ("Multiple choice", ev[ev.mc]), ("Open-ended", ev[~ev.mc])]
groups += [(NAMES[s], ev[ev.segment == s]) for s in NAMES]
rows = []
for name, g in groups:
    k, n = int(g.error.sum()), len(g)
    lo, hi = clopper_pearson(k, n)
    rows.append((name, k / n, lo, hi, g.format_error.mean(), n))
t = pd.DataFrame(rows, columns=["name", "err", "lo", "hi", "fmt", "n"])[::-1]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREY, ORANGE = "#3b6ea8", "#9aa5b1", "#d98c3f"

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
y = range(len(t))
ax.barh(
    y,
    t.err * 100,
    color=[
        GREY if n.startswith("All") or n in ("Multiple choice", "Open-ended") else BLUE
        for n in t.name
    ],
    height=0.6,
    label="Worse in substance",
)
ax.errorbar(
    t.err * 100,
    y,
    xerr=[(t.err - t.lo) * 100, (t.hi - t.err) * 100],
    fmt="none",
    ecolor="#333",
    capsize=3,
    lw=1,
)
ax.scatter(t.fmt * 100, y, color=ORANGE, zorder=3, s=22, label="Format slips (separate)")
ax.set_yticks(list(y), [f"{n}  (n={c})" for n, c in zip(t.name, t.n)])
ax.set_xlabel("% of answers where Qwen3-30B was worse than Claude Sonnet 5.5  (95% range)")
ax.set_xlim(0, 100)
ax.axhline(len(NAMES) - 0.5, color="#ccc", lw=0.8)
ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, frameon=False, fontsize=8)
ax.set_title("Quality loss by task (judge: Gemini 3.6 Flash)", loc="left", fontsize=11, pad=18)
fig.text(
    0.01, 0.01, "* subjective task: judge partly prefers Sonnet's style", fontsize=8, color="#555"
)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "live_quality_by_task.png", facecolor="white")

per_k = 1000 / len(ev)
sonnet = ev["ref_cost_usd"].sum() * per_k
qwen = ev["cost_usd"].sum() * per_k
check_3pct = ev["measure_cost_usd"].sum() * per_k * 0.03
fig, ax = plt.subplots(figsize=(7.5, 2.3), dpi=150)
ax.barh([1], [sonnet], color=GREY, height=0.5)
ax.barh([0], [qwen], color=BLUE, height=0.5, label="Qwen answers")
ax.barh([0], [check_3pct], left=[qwen], color=ORANGE, height=0.5, label="Checking 3% of answers")
ax.text(sonnet, 1, f"  ${sonnet:.2f}", va="center")
ax.text(
    qwen + check_3pct,
    0,
    f"  ${qwen + check_3pct:.2f}  ({1 - (qwen + check_3pct) / sonnet:.0%} less)",
    va="center",
)
ax.set_yticks([0, 1], ["Qwen3-30B + checks", "Claude Sonnet 5.5"])
ax.set_xlabel("Cost per 1,000 requests (USD, from this run's actual API charges)")
ax.set_xlim(0, sonnet * 1.25)
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.set_title("Cost", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "live_cost.png", facecolor="white")
print(t[::-1].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"per 1k: sonnet ${sonnet:.2f}, qwen ${qwen:.3f}, checking@3% ${check_3pct:.3f}")
