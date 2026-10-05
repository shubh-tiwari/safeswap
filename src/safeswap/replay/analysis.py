"""Everything a swap report shows, computed from the pairs table.

Pairs table columns (written by replay.engine):
  pair_id, session_id, turn_index, content, format   outcomes: win | tie | loss (candidate side)
  slice:<dim>                                        slice label per dimension
  check:<name>:baseline, check:<name>:candidate      True / False / None
  baseline_cost_usd, candidate_cost_usd, judge_cost_usd
  baseline_latency_ms, candidate_latency_ms
  judge_parsed, judge_consistent, identical          judge health
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from safeswap.replay.decision import Policy, Verdict, decide
from safeswap.stats import paired


@dataclass
class SliceRow:
    dimension: str
    label: str
    result: paired.PairedResult
    flagged: bool = False
    tested: bool = False  # large enough to be tested


@dataclass
class Analysis:
    overall: paired.PairedResult
    format: paired.PairedResult
    slices: list[SliceRow]
    checks: list[tuple[str, paired.PairedResult, float, float]]  # name, result, base, cand pass
    cost: dict
    latency: dict
    judge: dict
    verdict: Verdict
    meta: dict = field(default_factory=dict)


def _slices(pairs: pd.DataFrame, policy: Policy) -> list[SliceRow]:
    rows = []
    for col in [c for c in pairs.columns if c.startswith("slice:")]:
        dim = col.split(":", 1)[1]
        if pairs[col].nunique(dropna=True) < 2:
            continue  # one value covers everything: not a slice
        for label, g in pairs.groupby(col, dropna=True):
            r = paired.compare(g["content"])
            rows.append(SliceRow(dim, str(label), r, tested=r.n >= policy.min_slice_size))
    tested = [s for s in rows if s.tested]
    if tested:
        discover = paired.benjamini_hochberg([s.result.p_value for s in tested], policy.fdr_q)
        block = policy.block_if_any_slice_drop_pp / 100
        for s, d in zip(tested, discover):
            s.flagged = bool(d) and s.result.delta <= -block
    return sorted(rows, key=lambda s: (s.dimension, s.result.delta))


def _checks(pairs: pd.DataFrame) -> list:
    out = []
    names = sorted({c.split(":")[1] for c in pairs.columns if c.startswith("check:")})
    for name in names:
        b, c = pairs[f"check:{name}:baseline"], pairs[f"check:{name}:candidate"]
        ok = b.notna() & c.notna()
        if not ok.any():
            continue
        bb, cc = b[ok].astype(bool), c[ok].astype(bool)
        out.append((name, paired.compare_binary(bb, cc), float(bb.mean()), float(cc.mean())))
    return out


def _p(x: pd.Series, q: float) -> float:
    x = x.dropna()
    return float(np.percentile(x, q)) if len(x) else float("nan")


def analyze(pairs: pd.DataFrame, policy: Policy, meta: dict | None = None) -> Analysis:
    overall = paired.compare(pairs["content"])
    slices = _slices(pairs, policy)
    flagged = [s for s in slices if s.flagged]
    rest = None
    if flagged:
        inside = np.zeros(len(pairs), dtype=bool)
        for s in flagged:
            inside |= (pairs[f"slice:{s.dimension}"].astype(str) == s.label).to_numpy()
        rest = paired.compare(pairs.loc[~inside, "content"])
    labels = [f"{s.dimension}: {s.label}" for s in flagged]
    verdict = decide(overall, labels, rest, policy)

    n = len(pairs)
    base_cost, cand_cost = pairs["baseline_cost_usd"], pairs["candidate_cost_usd"]
    cost = {
        "baseline_per_1k": float(base_cost.mean() * 1000) if base_cost.notna().any() else None,
        "candidate_per_1k": float(cand_cost.mean() * 1000),
        "judge_total": float(pairs["judge_cost_usd"].sum()),
    }
    if cost["baseline_per_1k"]:
        cost["change_pct"] = 100 * (cost["candidate_per_1k"] / cost["baseline_per_1k"] - 1)
    latency = {
        side: {
            "p50": _p(pairs[f"{side}_latency_ms"], 50),
            "p95": _p(pairs[f"{side}_latency_ms"], 95),
        }
        for side in ("baseline", "candidate")
    }
    judged = pairs[~pairs["identical"]]
    judge = {
        "pairs": n,
        "identical": int(pairs["identical"].sum()),
        "judged": len(judged),
        "parsed": float(judged["judge_parsed"].mean()) if len(judged) else 1.0,
        "order_consistent": float(judged["judge_consistent"].mean()) if len(judged) else 1.0,
    }
    return Analysis(
        overall=overall,
        format=paired.compare(pairs["format"]),
        slices=slices,
        checks=_checks(pairs),
        cost=cost,
        latency=latency,
        judge=judge,
        verdict=verdict,
        meta=meta or {},
    )
