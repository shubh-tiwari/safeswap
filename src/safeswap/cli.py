"""safeswap CLI.

safeswap run experiments/00_smoke.yaml     # replay or live run -> runs/<name>-<time>/
safeswap report runs/<dir>                 # re-render report from events.parquet
safeswap sample-size --alpha 0.02          # labels needed to certify error <= alpha
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml


def _run(cfg_path: str) -> None:
    from dotenv import load_dotenv

    from . import data, monitor, policies, report

    load_dotenv()
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    name = cfg.get("name", Path(cfg_path).stem)
    run_dir = Path("runs") / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    seed = cfg.get("seed", 0)
    small, large = cfg["models"]["small"], cfg["models"]["large"]

    ds = cfg["dataset"]
    df = data.load_routerbench(
        n=ds.get("n"),
        seed=seed,
        shots=ds.get("shots", 0),
        segments=ds.get("segments"),
        mix=ds.get("mix"),
    )
    calib = None
    if ds.get("calib_frac"):
        calib = df.sample(frac=ds["calib_frac"], random_state=seed)
        df = df.drop(calib.index).reset_index(drop=True)
    policy = policies.build(cfg["policy"], small, large, calib)
    decision = policy.choose(df)

    if cfg.get("mode", "offline") == "offline":
        ev = monitor.replay_offline(df, decision, large, cfg.get("shadow", {}), seed)
    else:
        from .backends import OpenRouter
        from .judge import LLMJudge

        client = OpenRouter(spend_cap_usd=cfg.get("spend_cap_usd", 1.0))
        judge = LLMJudge(
            client,
            cfg["models"]["judge"],
            cfg.get("judge_rule", "both"),
            cfg.get("judge_max_tokens", 1024),
            cfg.get("judge_reasoning", "low"),
        )
        ev = monitor.run_live(
            df,
            decision,
            large,
            cfg.get("shadow", {}),
            client,
            judge,
            cfg.get("max_tokens", 1024),
            seed,
            cfg.get("workers", 8),
        )
        print(f"live spend this run: ${client.spent_usd:.4f}")

    monitor.save(ev, run_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    print(report.write(ev, run_dir, cfg.get("delta", 0.05)))
    print(f"\n-> {run_dir}")


def _report(run_dir: str, delta: float) -> None:
    from . import report

    ev = pd.read_parquet(Path(run_dir) / "events.parquet")
    print(report.write(ev, Path(run_dir), delta))


def _sample_size(alpha: float, delta: float) -> None:
    from .estimators import required_n

    print(f"labels needed to certify error <= {alpha:.2%} at {1 - delta:.0%} confidence")
    for frac in (0.0, 0.25, 0.5, 0.75, 0.9):
        t = alpha * frac
        print(f"  true error {t:.4%}: {required_n(alpha, t, delta):>8,}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="safeswap")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("config")
    rep = sub.add_parser("report")
    rep.add_argument("run_dir")
    rep.add_argument("--delta", type=float, default=0.05)
    ss = sub.add_parser("sample-size")
    ss.add_argument("--alpha", type=float, default=0.02)
    ss.add_argument("--delta", type=float, default=0.05)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        _run(a.config)
    elif a.cmd == "report":
        _report(a.run_dir, a.delta)
    else:
        _sample_size(a.alpha, a.delta)


if __name__ == "__main__":
    main()
