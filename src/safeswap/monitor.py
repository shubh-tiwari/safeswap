"""Run a policy over traffic, shadow a sample, label it, and log one event per request.

The event log is the only thing reports read. Columns:

  request_id, segment, action        what was served
  by_reference                       served by the reference model itself (error known to be 0)
  cost_usd                           cost of the served call
  ref_cost_usd                       cost of the reference (large) answer, if known (NaN otherwise)
  shadow_prob, shadowed              sampling design; estimators reweight by 1 / shadow_prob
  error                              label for shadowed requests (1 = worse than reference), else NaN
  measure_cost_usd                   what shadowing this request cost (reference call + judge)
  true_error                         offline ground truth for every request (NaN on live traffic),
                                     used only to check whether the estimates are honest

Two modes:
  * replay_offline: RouterBench-style tables with per-model answers, costs and scores. Free.
  * run_live: serve and shadow through real API calls, labelled by an LLM judge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import sampler as smp
from .backends import OpenRouter
from .judge import LLMJudge
from .policies import Decision


def sampling_probs(decision: Decision, large: str, cfg: dict) -> np.ndarray:
    """Inclusion probabilities. Requests served by the reference model are never worse than
    themselves, so they need no label: p = 1 with a free, known label of 0."""
    rate, floor = cfg.get("rate", 0.03), cfg.get("floor", cfg.get("rate", 0.03))
    p = np.ones(len(decision.model))
    need = decision.model != large  # same label budget (rate x these) in every mode
    if cfg.get("mode", "uniform") == "active" and decision.uncertainty is not None:
        p[need] = smp.active(np.asarray(decision.uncertainty)[need], rate, floor)
    else:
        p[need] = rate
    return p


def replay_offline(
    df: pd.DataFrame, decision: Decision, large: str, shadow: dict, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    model = decision.model
    served_score = np.zeros(len(df))
    served_cost = np.zeros(len(df))
    for m in np.unique(model):
        sel = model == m
        served_score[sel] = df[f"score:{m}"].to_numpy(float)[sel]
        served_cost[sel] = df[f"cost:{m}"].to_numpy(float)[sel]
    ref_score = df[f"score:{large}"].to_numpy(float)
    ref_cost = df[f"cost:{large}"].to_numpy(float)
    true_error = (served_score < ref_score).astype(float)

    p = sampling_probs(decision, large, shadow)
    shadowed = smp.draw(p, rng)
    is_ref = model == large
    need_ref_call = shadowed & ~is_ref
    return pd.DataFrame(
        {
            "request_id": df["id"].to_numpy(),
            "segment": df["segment"].to_numpy(),
            "action": model,
            "by_reference": is_ref,
            "cost_usd": served_cost,
            "ref_cost_usd": np.where(shadowed | is_ref, ref_cost, np.nan),
            "shadow_prob": p,
            "shadowed": shadowed,
            "error": np.where(shadowed, true_error, np.nan),
            "measure_cost_usd": np.where(need_ref_call, ref_cost, 0.0),  # judge is free here
            "true_error": true_error,
            "true_ref_cost_usd": ref_cost,
        }
    )


def run_live(
    df: pd.DataFrame,
    decision: Decision,
    large: str,
    shadow: dict,
    client: OpenRouter,
    judge: LLMJudge,
    max_tokens: int = 1024,
    seed: int = 0,
    workers: int = 8,
    log=print,
) -> pd.DataFrame:
    """Serve and shadow through real API calls, `workers` requests at a time.

    Stops cleanly at the spend cap: requests not yet processed are dropped from the log.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .backends import SpendCapExceeded

    rng = np.random.default_rng(seed)
    p = sampling_probs(decision, large, shadow)
    shadowed = smp.draw(p, rng)

    def one(i: int) -> dict:
        prompt, model = df["prompt"].iat[i], decision.model[i]
        msgs = [{"role": "user", "content": prompt}]
        served = client.complete(model, msgs, max_tokens=max_tokens)
        row = {
            "request_id": df["id"].iat[i],
            "segment": df["segment"].iat[i],
            "action": model,
            "by_reference": model == large,
            "cost_usd": served.cost_usd,
            "ref_cost_usd": served.cost_usd if model == large else np.nan,
            "shadow_prob": p[i],
            "shadowed": bool(shadowed[i]),
            "error": np.nan,
            "format_error": np.nan,
            "served_truncated": served.truncated,
            "ref_truncated": np.nan,
            "measure_cost_usd": 0.0,
            "true_error": np.nan,
            "judge_detail": None,
        }
        if model == large:
            row.update(error=0.0, format_error=0.0)
        elif shadowed[i]:
            ref = client.complete(large, msgs, max_tokens=max_tokens)
            lab = judge.label(prompt, served.text, ref.text)
            row.update(
                ref_cost_usd=ref.cost_usd,
                ref_truncated=ref.truncated,
                error=float(lab.error),
                format_error=float(lab.detail.get("format_worse", False)),
                measure_cost_usd=ref.cost_usd + lab.cost_usd,
                judge_detail=lab.detail,
            )
        return row

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, i): i for i in range(len(df))}
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except SpendCapExceeded as e:
                log(f"  skipped request {futures[fut]}: {e}")
                continue
            done += 1
            if done % 25 == 0:
                log(f"  {done}/{len(df)} requests, spent ${client.spent_usd:.4f}")
    order = {rid: k for k, rid in enumerate(df["id"])}
    return pd.DataFrame(rows).sort_values("request_id", key=lambda c: c.map(order))


def save(events: pd.DataFrame, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out = events.copy()
    if "judge_detail" in out:
        out["judge_detail"] = out["judge_detail"].map(lambda d: None if d is None else str(d))
    path = run_dir / "events.parquet"
    out.to_parquet(path, index=False)
    return path
