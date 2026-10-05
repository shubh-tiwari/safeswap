"""Replay end to end: read traces, replay, analyse, write pairs and reports to a run folder.

Run folder layout:
  pairs.parquet   one row per compared turn (see replay.analysis for columns)
  meta.json       models, policy, counts, spend
  report.txt / report.json / report.html
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from safeswap.data.traces import iter_turns, read_jsonl
from safeswap.llm.client import OpenRouter
from safeswap.replay import html, report
from safeswap.replay.analysis import analyze
from safeswap.replay.decision import Policy
from safeswap.replay.engine import replay
from safeswap.scoring.judge import LLMJudge


def render(pairs: pd.DataFrame, meta: dict, policy: Policy, run_dir: Path) -> str:
    a = analyze(pairs, policy, meta)
    out = report.write(a, run_dir)
    html.write(a, pairs, run_dir, policy.max_quality_drop_pp)
    return out


def run(
    logs: str | Path,
    candidate: str,
    judge_model: str,
    policy_path: str | Path | None = None,
    select: dict[str, str] | None = None,
    baseline_label: str | None = None,
    sample: float = 1.0,
    max_pairs: int | None = None,
    max_tokens: int = 1024,
    judge_max_tokens: int = 1536,
    judge_reasoning: str | None = "low",
    budget_usd: float = 1.0,
    cache_only: bool = False,
    workers: int = 8,
    seed: int = 0,
    out_root: str | Path = "runs",
    log=print,
) -> tuple[Path, str, dict]:
    policy = Policy.load(policy_path)
    units = list(iter_turns(read_jsonl(logs), select))
    if not units:
        raise SystemExit("no turns to replay (check --logs and --select)")
    client = OpenRouter(spend_cap_usd=budget_usd, cache_only=cache_only)
    judge = LLMJudge(
        client, judge_model, max_tokens=judge_max_tokens, reasoning_effort=judge_reasoning
    )
    pairs, stats = replay(
        units, candidate, client, judge, max_tokens, sample, max_pairs, workers, seed, log
    )
    if pairs.empty:
        raise SystemExit(f"no pairs produced: {stats}")
    baseline = baseline_label or str(pairs["baseline_model"].mode().iat[0])
    run_dir = Path(out_root) / f"replay-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "baseline": baseline,
        "candidate": candidate,
        "judge": judge_model,
        "mode": "replay (cache only)" if cache_only else "replay",
        "logs": str(logs),
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "policy": asdict(policy),
        "stats": stats,
        "network_calls": client._http is not None,
    }
    pairs.to_parquet(run_dir / "pairs.parquet", index=False)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir, render(pairs, meta, policy, run_dir), stats


def rerender(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    policy = Policy(**meta.get("policy", {}))
    return render(pd.read_parquet(run_dir / "pairs.parquet"), meta, policy, run_dir)
