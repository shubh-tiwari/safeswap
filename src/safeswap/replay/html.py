"""Self-contained HTML swap report: one file, inline CSS and SVG, no scripts, light and dark."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from safeswap.replay.analysis import Analysis

VERDICT_STYLE = {
    "ship": ("var(--good)", "Ship"),
    "split": ("var(--accent)", "Split"),
    "hold": ("var(--warn)", "Hold"),
    "rollback": ("var(--bad)", "Rollback"),
}

CSS = """
:root{--bg:#f7f8fa;--card:#fff;--fg:#1d2330;--muted:#667085;--line:#e4e7ec;--good:#1f8a5b;
--bad:#c4372d;--warn:#b7791f;--accent:#2f6fde;--win:#1f8a5b;--tie:#c9ced6;--loss:#d9534f}
@media (prefers-color-scheme:dark){:root{--bg:#0f1218;--card:#171b23;--fg:#e6e8ec;--muted:#98a2b3;
--line:#2a303b;--tie:#3a414d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 20px 48px}
h1{font-size:22px;margin:0}h2{font-size:15px;margin:0 0 12px}
.sub{color:var(--muted);margin-top:4px}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:18px;margin-top:16px}
.verdict{display:flex;gap:16px;align-items:center;border-left:6px solid var(--vc)}
.badge{background:var(--vc);color:#fff;font-weight:700;letter-spacing:.04em;padding:6px 12px;
border-radius:6px;text-transform:uppercase;font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.kpi .l{color:var(--muted);font-size:12px}.kpi .v{font-size:22px;font-weight:650;margin-top:2px}
.kpi .s{color:var(--muted);font-size:12px;margin-top:2px}
.neg{color:var(--bad)}.pos{color:var(--good)}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:7px 8px;
border-bottom:1px solid var(--line);vertical-align:middle}th{color:var(--muted);font-weight:500;
font-size:12px}td.num{text-align:right;font-variant-numeric:tabular-nums}
.dim td{background:transparent;color:var(--muted);font-size:12px;text-transform:uppercase;
letter-spacing:.05em;padding-top:14px}
.chip{font-size:11px;padding:2px 7px;border-radius:999px;background:var(--bad);color:#fff}
.chip.muted{background:var(--line);color:var(--muted)}
.bar{display:flex;height:12px;border-radius:6px;overflow:hidden;min-width:120px}
.legend span{margin-right:14px;color:var(--muted);font-size:12px}
.dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
details{border-top:1px solid var(--line);padding:8px 0}summary{cursor:pointer;color:var(--muted)}
pre{white-space:pre-wrap;background:var(--bg);border:1px solid var(--line);border-radius:6px;
padding:10px;max-height:240px;overflow:auto;font-size:12px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:760px){.kpis{grid-template-columns:repeat(2,1fr)}.cols{grid-template-columns:1fr}}
"""


def _pp(x: float) -> str:
    return f"{x * 100:+.1f} pp"


def _wtl_bar(w: int, t: int, l: int) -> str:
    n = max(w + t + l, 1)
    seg = [(w, "var(--win)"), (t, "var(--tie)"), (l, "var(--loss)")]
    inner = "".join(
        f'<div style="width:{100 * k / n:.2f}%;background:{c}" title="{k}"></div>'
        for k, c in seg
        if k
    )
    return f'<div class="bar">{inner}</div>'


def _forest(d: float, lo: float, hi: float, xmin: float, xmax: float, limit: float) -> str:
    w, h = 240, 18

    def x(v: float) -> float:
        return 6 + (w - 12) * (v - xmin) / (xmax - xmin)

    color = "var(--bad)" if hi < 0 else ("var(--good)" if lo > 0 else "var(--muted)")
    return (
        f'<svg width="{w}" height="{h}" role="img" aria-label="{_pp(d)}">'
        f'<line x1="{x(0):.1f}" y1="0" x2="{x(0):.1f}" y2="{h}" stroke="var(--line)" stroke-width="2"/>'
        f'<line x1="{x(-limit):.1f}" y1="0" x2="{x(-limit):.1f}" y2="{h}" stroke="var(--warn)" '
        f'stroke-dasharray="3 3"/>'
        f'<line x1="{x(lo):.1f}" y1="9" x2="{x(hi):.1f}" y2="9" stroke="{color}" stroke-width="2.5"/>'
        f'<circle cx="{x(d):.1f}" cy="9" r="4" fill="{color}"/></svg>'
    )


def render(a: Analysis, pairs: pd.DataFrame, max_drop_pp: float, examples: int = 8) -> str:
    m, o, v = a.meta, a.overall, a.verdict
    vc, vlabel = VERDICT_STYLE.get(v.verdict, ("var(--muted)", v.verdict))
    split = ""
    if v.verdict == "split" and v.rest is not None:
        split = (
            f'<div class="sub">Keep on baseline: <b>{escape(", ".join(v.keep_on_baseline))}</b>. '
            f"Ship the rest: {_pp(v.rest.delta)} [{_pp(v.rest.delta_lo)}, {_pp(v.rest.delta_hi)}]"
            "</div>"
        )
    c, lb, lc = a.cost, a.latency["baseline"], a.latency["candidate"]
    cost_v = f"{c['change_pct']:+.0f}%" if c.get("change_pct") is not None else "—"
    cost_s = (
        f"${c['baseline_per_1k']:.2f} → ${c['candidate_per_1k']:.2f} per 1K"
        if c.get("baseline_per_1k")
        else f"${c['candidate_per_1k']:.2f} per 1K"
    )
    lat_v = f"{lc['p95'] / 1000:.1f} s" if lc["p95"] == lc["p95"] else "—"
    lat_s = f"from {lb['p95'] / 1000:.1f} s (p95)" if lb["p95"] == lb["p95"] else "p95"
    qcls = "neg" if o.delta_hi < 0 else ("pos" if o.delta_lo > 0 else "")
    kpis = f"""
<div class="kpis">
 <div class="kpi"><div class="l">Quality vs. baseline</div><div class="v {qcls}">{_pp(o.delta)}</div>
  <div class="s">95% CI [{_pp(o.delta_lo)}, {_pp(o.delta_hi)}]</div></div>
 <div class="kpi"><div class="l">Not worse than baseline</div><div class="v">{o.acceptability:.0%}</div>
  <div class="s">95% CI [{o.accept_lo:.0%}, {o.accept_hi:.0%}]</div></div>
 <div class="kpi"><div class="l">Cost</div><div class="v pos">{cost_v}</div><div class="s">{cost_s}</div></div>
 <div class="kpi"><div class="l">Latency p95</div><div class="v">{lat_v}</div><div class="s">{lat_s}</div></div>
</div>"""

    big = [s for s in a.slices if s.tested]
    lows = [s.result.delta_lo for s in big] + [o.delta_lo, -max_drop_pp / 100]
    highs = [s.result.delta_hi for s in big] + [o.delta_hi, 0.0]
    xmin, xmax = max(-1.0, min(lows) - 0.02), min(1.0, max(highs) + 0.02)
    rows = []
    for dim in dict.fromkeys(s.dimension for s in a.slices):
        group = [s for s in a.slices if s.dimension == dim]
        rows.append(f'<tr class="dim"><td colspan="6">{escape(dim)}</td></tr>')
        for s in (g for g in group if g.tested):
            r = s.result
            chip = '<span class="chip">blocks ship</span>' if s.flagged else ""
            rows.append(
                f"<tr><td>{escape(s.label)}</td><td class='num'>{r.n}</td>"
                f"<td>{_wtl_bar(r.wins, r.ties, r.losses)}</td>"
                f"<td class='num'>{_pp(r.delta)}</td>"
                f"<td>{_forest(r.delta, r.delta_lo, r.delta_hi, xmin, xmax, max_drop_pp / 100)}"
                f"</td><td>{chip}</td></tr>"
            )
        small = [g for g in group if not g.tested]
        if small:
            items = ", ".join(
                f"{escape(g.label)} ({g.result.n}, {_pp(g.result.delta)})" for g in small
            )
            rows.append(
                f"<tr><td colspan='6'><details><summary>{len(small)} smaller "
                f"slice{'s' if len(small) > 1 else ''} "
                f"(&lt;{a.meta.get('policy', {}).get('min_slice_size', 30)} pairs, not tested)"
                f"</summary><div class='sub'>{items}</div></details></td></tr>"
            )
    checks = "".join(
        f"<tr><td>{escape(n)} <span class='sub'>n={r.n}</span></td>"
        f"<td class='num'>{b:.0%}</td><td class='num'>{cc:.0%}</td>"
        f"<td class='num {'neg' if r.delta < 0 else ''}'>{_pp(r.delta)}</td></tr>"
        for n, r, b, cc in a.checks
    )
    j = a.judge
    losses = pairs[pairs["content"] == "loss"].head(examples)
    ex = "".join(
        f"<details><summary>{escape(str(r['pair_id']))} · "
        f"{escape(str(r.get('slice:feature', r.get('slice:language', ''))))}</summary>"
        f"<pre>{escape(str(r['request'])[:1500])}</pre><div class='cols'>"
        f"<div><b>Baseline</b><pre>{escape(str(r['baseline_output'])[:2000])}</pre></div>"
        f"<div><b>Candidate</b><pre>{escape(str(r['candidate_output'])[:2000])}</pre></div></div>"
        "</details>"
        for _, r in losses.iterrows()
    )
    judge_cost_label = (
        "Judge cost (as originally charged)" if "cache" in m.get("mode", "") else "Judge spend"
    )
    title = f"{m.get('baseline', 'baseline')} → {m.get('candidate', 'candidate')}"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Swap report</title><style>{CSS}</style></head><body><main>
<h1>Swap report</h1>
<div class="sub">{escape(title)} · {o.n:,} pairs · {escape(m.get("mode", "replay"))}
 · judge {escape(m.get("judge", ""))} · {escape(m.get("created", ""))}</div>
<div class="card verdict" style="--vc:{vc}"><span class="badge">{vlabel}</span>
 <div><div><b>{escape(v.reason)}</b></div>{split}</div></div>
{kpis}
<div class="card"><h2>Outcomes</h2>{_wtl_bar(o.wins, o.ties, o.losses)}
 <div class="legend" style="margin-top:8px"><span><i class="dot" style="background:var(--win)"></i>
 candidate better {o.wins}</span><span><i class="dot" style="background:var(--tie)"></i>tie {o.ties}
 </span><span><i class="dot" style="background:var(--loss)"></i>candidate worse {o.losses}</span>
 <span>format: {_pp(a.format.delta)} (counted separately)</span></div></div>
<div class="card"><h2>Slices</h2><table><tr><th>Slice</th><th class="num">Pairs</th>
 <th>Better / tie / worse</th><th class="num">Δ quality</th>
 <th>95% CI (dashed: −{max_drop_pp:g} pp limit)</th><th></th></tr>{"".join(rows)}</table></div>
<div class="cols"><div class="card"><h2>Deterministic checks</h2><table><tr><th>Check</th>
 <th class="num">Baseline</th><th class="num">Candidate</th><th class="num">Δ</th></tr>{checks}
 </table></div>
<div class="card"><h2>Judge health</h2><table>
 <tr><td>Pairs judged</td><td class="num">{j["judged"]}</td></tr>
 <tr><td>Identical answers (no judge call)</td><td class="num">{j["identical"]}</td></tr>
 <tr><td>Replies parsed</td><td class="num">{j["parsed"]:.0%}</td></tr>
 <tr><td>Same verdict in both orders</td><td class="num">{j["order_consistent"]:.0%}</td></tr>
 <tr><td>{judge_cost_label}</td><td class="num">${c["judge_total"]:.2f}</td></tr></table></div></div>
<div class="card"><h2>Where the candidate lost</h2>{ex or '<div class="sub">No losses.</div>'}</div>
</main></body></html>"""


def write(a: Analysis, pairs: pd.DataFrame, run_dir: Path, max_drop_pp: float) -> Path:
    path = run_dir / "report.html"
    path.write_text(render(a, pairs, max_drop_pp), encoding="utf-8")
    return path
