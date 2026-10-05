"""Text and JSON swap reports (the HTML one lives in replay.html)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from safeswap.replay.analysis import Analysis


def _pp(x: float) -> str:
    return f"{x * 100:+.1f} pp"


def _ci(lo: float, hi: float) -> str:
    return f"95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}]"


def text(a: Analysis) -> str:
    m, o = a.meta, a.overall
    lines = [
        (
            f"Swap report: {m.get('baseline', 'baseline')} → {m.get('candidate', 'candidate')}"
            f"      {o.n:,} pairs, {m.get('mode', 'replay')}"
        ),
        "",
        (
            f"Quality (judge, vs. baseline)   {_pp(o.delta)}   {_ci(o.delta_lo, o.delta_hi)}"
            f"   win {o.wins} / tie {o.ties} / loss {o.losses}"
        ),
        (
            f"Format (judge, separate)        {_pp(a.format.delta)}   "
            f"{_ci(a.format.delta_lo, a.format.delta_hi)}"
        ),
    ]
    for name, r, base, cand in a.checks:
        lines.append(f"Check: {name:<34s} {base:.0%} → {cand:.0%}   ({_pp(r.delta)}, n={r.n})")
    c = a.cost
    if c.get("baseline_per_1k"):
        lines.append(
            f"Cost / 1K requests              ${c['baseline_per_1k']:.2f} → "
            f"${c['candidate_per_1k']:.2f}   ({c['change_pct']:+.0f}%)"
        )
    lb, lc = a.latency["baseline"], a.latency["candidate"]
    if lb["p95"] == lb["p95"]:  # not NaN
        lines.append(
            f"Latency p95                     {lb['p95'] / 1000:.1f} s → {lc['p95'] / 1000:.1f} s"
        )
    worse = [s for s in a.slices if s.tested and s.result.delta < 0]
    worse = sorted(worse, key=lambda s: s.result.delta)[:5]
    if worse:
        lines.append("")
        for i, s in enumerate(worse):
            flag = "  ← blocks ship" if s.flagged else ""
            head = "Worse on:  " if i == 0 else "           "
            lines.append(
                f"{head}{s.dimension} = {s.label:<24s} {_pp(s.result.delta)}  "
                f"(n={s.result.n}){flag}"
            )
    v = a.verdict
    lines += ["", f"Decision: {v.verdict.upper()} — {v.reason}"]
    if v.verdict == "split" and v.rest is not None:
        lines.append(
            f"          keep on baseline: {', '.join(v.keep_on_baseline)}; "
            f"ship the rest (est. {_pp(v.rest.delta)}, {_ci(v.rest.delta_lo, v.rest.delta_hi)})"
        )
    j = a.judge
    lines += [
        "",
        (
            f"Judge: {j['judged']} pairs judged ({j['identical']} identical answers skipped), "
            f"parsed {j['parsed']:.0%}, same verdict in both orders {j['order_consistent']:.0%}"
        ),
    ]
    return "\n".join(lines)


def to_json(a: Analysis) -> dict:
    return {
        "meta": a.meta,
        "verdict": {
            "verdict": a.verdict.verdict,
            "reason": a.verdict.reason,
            "keep_on_baseline": a.verdict.keep_on_baseline,
            "rest": a.verdict.rest.as_dict() if a.verdict.rest else None,
        },
        "quality": a.overall.as_dict(),
        "format": a.format.as_dict(),
        "checks": [
            {"name": n, "baseline_pass": b, "candidate_pass": c, **r.as_dict()}
            for n, r, b, c in a.checks
        ],
        "slices": [
            {
                "dimension": s.dimension,
                "label": s.label,
                "flagged": s.flagged,
                "tested": s.tested,
                **asdict(s.result),
            }
            for s in a.slices
        ],
        "cost": a.cost,
        "latency": a.latency,
        "judge": a.judge,
    }


def write(a: Analysis, run_dir: Path) -> str:
    out = text(a)
    (run_dir / "report.txt").write_text(out + "\n")
    (run_dir / "report.json").write_text(json.dumps(to_json(a), indent=2, default=float))
    return out
