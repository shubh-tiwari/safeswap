"""Turn an event log into the numbers that matter: savings, realised error ± CI, measurement cost.

Only shadowed requests carry labels and reference costs, so both the error rate and the
always-large baseline cost are IPW (Hájek) estimates. When the log also has offline ground truth
(`true_error`, `true_ref_cost_usd`), the true values are reported next to the estimates so you can
see whether the intervals actually cover them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from safeswap.stats import estimators as est


def _census(ev: pd.DataFrame) -> pd.Series:
    """Requests whose error is known exactly: served by the reference itself.

    Checked cheap-model answers are *not* census even at p = 1: they are a sample of a rate that
    still needs an interval. Older logs lack `by_reference`; there, reference-served rows are the
    ones whose reference cost is their own served cost.
    """
    if "by_reference" in ev:
        return ev["by_reference"].astype(bool)
    return (ev["shadow_prob"] >= 1) & (ev["cost_usd"] == ev["ref_cost_usd"])


def _error_estimate(ev: pd.DataFrame, fn, delta: float) -> est.Estimate:
    census = _census(ev)
    rest = ev[~census]
    s = rest[rest["shadowed"]]
    return est.with_census(
        fn,
        ev.loc[census, "error"].to_numpy(float),
        s["error"].to_numpy(float),
        s["shadow_prob"].to_numpy(float),
        len(rest),
        delta,
    )


def summarize(ev: pd.DataFrame, delta: float = 0.05) -> dict:
    s = ev[ev["shadowed"]]
    p = s["shadow_prob"].to_numpy(float)
    n = len(ev)
    out: dict = {"n_requests": n, "n_labels": int((ev["shadowed"] & ~_census(ev)).sum())}
    out["error"] = {
        name: {"value": r.value, "lo": r.lo, "hi": r.hi, "n_eff": r.n_eff}
        for name, fn in est.ESTIMATORS.items()
        for r in [_error_estimate(ev, fn, delta)]
    }

    served = float(ev["cost_usd"].sum())
    w = 1.0 / p
    large_est = float((w * s["ref_cost_usd"].to_numpy(float)).sum() / w.sum() * n)
    measure = float(ev["measure_cost_usd"].sum())
    out["cost"] = {
        "served_usd": served,
        "always_large_usd_est": large_est,
        "savings_pct_est": 100 * (1 - served / large_est) if large_est else float("nan"),
        "measurement_usd": measure,
        "net_savings_pct_est": 100 * (1 - (served + measure) / large_est)
        if large_est
        else float("nan"),
    }
    out["action_mix"] = ev["action"].value_counts(normalize=True).round(4).to_dict()
    if "format_error" in ev and ev["format_error"].notna().any():
        f = ev[ev["format_error"].notna() | _census(ev)].assign(error=lambda d: d["format_error"])
        r = _error_estimate(f, est.hajek_cp, delta)
        out["format_error"] = {"value": r.value, "lo": r.lo, "hi": r.hi}
    if "served_truncated" in ev:
        ref = ev["ref_truncated"].dropna()
        out["truncated"] = {
            "served": float(ev["served_truncated"].mean()),
            "reference": float(ref.astype(float).mean()) if len(ref) else 0.0,
        }

    if "true_error" in ev and ev["true_error"].notna().all():
        true_err = float(ev["true_error"].mean())
        out["truth"] = {
            "error": true_err,
            "covered": {k: v["lo"] <= true_err <= v["hi"] for k, v in out["error"].items()},
        }
        if "true_ref_cost_usd" in ev:
            large = float(ev["true_ref_cost_usd"].sum())
            out["truth"]["always_large_usd"] = large
            out["truth"]["savings_pct"] = 100 * (1 - served / large)
    return out


def segment_table(ev: pd.DataFrame, top: int = 10, delta: float = 0.05) -> pd.DataFrame:
    rows = []
    for seg, g in ev.groupby("segment"):
        s = g[g["shadowed"] & ~_census(g)]
        r = _error_estimate(g, est.hajek_cp, delta)
        rows.append(
            {
                "segment": seg,
                "n": len(g),
                "labels": len(s),
                "err": r.value,
                "hi": r.hi,
                "true": g["true_error"].mean() if "true_error" in g else np.nan,
            }
        )
    t = pd.DataFrame(rows).sort_values("n", ascending=False)
    return t.head(top)


def render(summary: dict) -> str:
    lines = [f"requests: {summary['n_requests']}   labels used: {summary['n_labels']}"]
    truth = summary.get("truth", {})
    lines.append("realised error (served worse than reference):")
    for k, v in summary["error"].items():
        cov = ""
        if "covered" in truth:
            cov = "  covers truth" if truth["covered"][k] else "  MISSES truth"
        lines.append(
            f"  {k:10s} {v['value']:.4f}  [{v['lo']:.4f}, {v['hi']:.4f}]  n_eff={v['n_eff']:.0f}{cov}"
        )
    if "error" in truth:
        lines.append(f"  {'TRUE':10s} {truth['error']:.4f}")
    c = summary["cost"]
    lines.append(
        f"cost: served ${c['served_usd']:.4f} | always-large est ${c['always_large_usd_est']:.4f}"
        + (f" (true ${truth['always_large_usd']:.4f})" if "always_large_usd" in truth else "")
    )
    lines.append(
        f"savings est {c['savings_pct_est']:.1f}%"
        + (f" (true {truth['savings_pct']:.1f}%)" if "savings_pct" in truth else "")
        + f" | measurement ${c['measurement_usd']:.4f} | net savings est {c['net_savings_pct_est']:.1f}%"
    )
    if "format_error" in summary:
        f = summary["format_error"]
        lines.append(
            f"format slips (counted separately): {f['value']:.4f}  [{f['lo']:.4f}, {f['hi']:.4f}]"
        )
    if "truncated" in summary:
        t = summary["truncated"]
        lines.append(f"hit max_tokens: served {t['served']:.1%}, reference {t['reference']:.1%}")
    lines.append(
        "action mix: " + ", ".join(f"{k} {v:.1%}" for k, v in summary["action_mix"].items())
    )
    return "\n".join(lines)


def write(ev: pd.DataFrame, run_dir: Path, delta: float = 0.05) -> str:
    summary = summarize(ev, delta)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    text = (
        render(summary)
        + "\n\nlargest segments:\n"
        + segment_table(ev).to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    (run_dir / "report.txt").write_text(text)
    return text
