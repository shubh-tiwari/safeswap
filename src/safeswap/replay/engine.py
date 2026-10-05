"""Replay logged turns through a candidate model and compare each answer with the logged baseline.

Turn-level replay: each turn keeps its real history; the candidate answers that turn only, so
every comparison is paired (same request, two answers). Deterministic checks run on both
answers first, then the pairwise judge. Calls run `workers` at a time and stop at the client's
spend cap. In cache-only mode a turn whose calls aren't cached is skipped and counted.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from safeswap.data.traces import ReplayUnit, length_bucket, request_text, turn_bucket
from safeswap.llm.client import CacheMiss, OpenRouter, SpendCapExceeded
from safeswap.scoring import checks
from safeswap.scoring.judge import LLMJudge

SLICE_KEYS = ("language", "feature", "format")


def _row(
    unit: ReplayUnit,
    client: OpenRouter,
    judge: LLMJudge,
    candidate: str,
    max_tokens: int,
    sample_p: float,
) -> dict:
    turn = unit.turn
    cand = client.complete(candidate, turn.messages, max_tokens=max_tokens)
    request = request_text(turn)
    base_trunc = turn.attributes.get("finish_reason") == "length"
    b_checks = checks.run(request, turn.output, base_trunc)
    c_checks = checks.run(request, cand.text, cand.truncated)
    o = judge.compare(request, cand.text, turn.output)
    row = {
        "pair_id": turn.turn_id,
        "session_id": unit.session_id,
        "turn_index": unit.turn_index,
        "baseline_model": turn.model,
        "candidate_model": candidate,
        "content": o.content,
        "format": o.format,
        "identical": bool(o.detail.get("identical", False)),
        "judge_parsed": bool(o.detail.get("parsed", True)),
        "judge_consistent": bool(o.detail.get("consistent", True)),
        "judge_cost_usd": o.cost_usd,
        "baseline_cost_usd": turn.attributes.get("cost_usd", np.nan),
        "candidate_cost_usd": cand.cost_usd,
        "baseline_latency_ms": turn.attributes.get("latency_ms", np.nan),
        "candidate_latency_ms": cand.latency_ms,
        "sample_prob": sample_p,
        "request": request,
        "baseline_output": turn.output,
        "candidate_output": cand.text,
        "slice:length": length_bucket(request),
        "slice:turn": turn_bucket(unit.turn_index),
    }
    for k in SLICE_KEYS:
        if k in unit.metadata:
            row[f"slice:{k}"] = str(unit.metadata[k])
    for name in b_checks:
        row[f"check:{name}:baseline"] = b_checks[name]
        row[f"check:{name}:candidate"] = c_checks[name]
    return row


def replay(
    units: list[ReplayUnit],
    candidate: str,
    client: OpenRouter,
    judge: LLMJudge,
    max_tokens: int = 1024,
    sample: float = 1.0,
    max_pairs: int | None = None,
    workers: int = 8,
    seed: int = 0,
    log=print,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    if sample < 1:
        units = [u for u in units if rng.random() < sample]
    if max_pairs is not None:
        units = units[:max_pairs]
    rows, stats = [], {"selected": len(units), "cache_miss": 0, "budget_stop": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_row, u, client, judge, candidate, max_tokens, sample) for u in units
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rows.append(fut.result())
            except CacheMiss:
                stats["cache_miss"] += 1
            except SpendCapExceeded:
                stats["budget_stop"] += 1
            if i % 50 == 0:
                log(f"  {i}/{len(units)} turns, spent ${client.spent_usd:.4f}")
    order = {u.turn.turn_id: k for k, u in enumerate(units)}
    pairs = pd.DataFrame(rows)
    if len(pairs):
        pairs = pairs.sort_values("pair_id", key=lambda c: c.map(order)).reset_index(drop=True)
    stats["pairs"] = len(pairs)
    stats["spent_usd"] = client.spent_usd
    return pairs, stats
