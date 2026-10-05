"""safeswap CLI.

safeswap replay --logs traces.jsonl --candidate MODEL --judge MODEL   # swap report -> runs/replay-*/
safeswap fetch wildchat --n 2000 --out data/wildchat.jsonl            # real traces (ODC-BY)
safeswap export-run runs/live-300-* --out runs/demo/traces.jsonl      # saved live run -> traces
safeswap run experiments/00_smoke.yaml                                # monitor run -> runs/<name>-*/
safeswap report runs/<dir> [--open]                                   # re-render any run's report
safeswap sample-size --alpha 0.02                                     # labels to certify error <= alpha
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml


def _run(cfg_path: str) -> None:
    from dotenv import load_dotenv

    from safeswap.data import routerbench as data
    from safeswap.monitor import report
    from safeswap.monitor import runner as monitor
    from safeswap.routing import policies

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
        from safeswap.llm.client import OpenRouter
        from safeswap.scoring.judge import LLMJudge

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


def _report(run_dir: str, delta: float, open_html: bool) -> None:
    d = Path(run_dir)
    if (d / "pairs.parquet").exists():
        from safeswap.replay.pipeline import rerender

        print(rerender(d))
        if open_html:
            import webbrowser

            webbrowser.open((d / "report.html").resolve().as_uri())
        return
    from safeswap.monitor import report

    print(report.write(pd.read_parquet(d / "events.parquet"), d, delta))


def _select(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        k, sep, v = p.partition("=")
        if not sep:
            raise SystemExit(f"--select expects key=value, got {p!r}")
        out[k] = v
    return out


def _replay(a) -> None:
    from dotenv import load_dotenv

    from safeswap.replay.pipeline import run

    load_dotenv()
    run_dir, out, stats = run(
        a.logs,
        a.candidate,
        a.judge,
        a.policy,
        _select(a.select),
        a.baseline_label,
        a.sample,
        a.max_pairs,
        a.max_tokens,
        a.judge_max_tokens,
        a.judge_reasoning or None,
        a.budget,
        a.cache_only,
        a.workers,
        a.seed,
    )
    print(out)
    skipped = {k: v for k, v in stats.items() if k in ("cache_miss", "budget_stop") and v}
    if skipped:
        print(f"\nskipped turns: {skipped}")
    print(f"\nspent ${stats['spent_usd']:.4f}\n-> {run_dir}/report.html")


def _sample_size(alpha: float, delta: float) -> None:
    from safeswap.stats.estimators import required_n

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
    rep.add_argument("--open", action="store_true", help="open the HTML report (replay runs)")
    rp = sub.add_parser("replay", help="compare a candidate with logged baseline answers")
    rp.add_argument("--logs", required=True, help="trace JSONL (sessions or OpenAI-style calls)")
    rp.add_argument("--candidate", required=True, help="candidate model (OpenRouter slug)")
    rp.add_argument("--judge", required=True, help="judge model (OpenRouter slug)")
    rp.add_argument(
        "--baseline-label", help="display name for the baseline (default: logged model)"
    )
    rp.add_argument("--select", action="append", default=[], help="filter turns: key=value")
    rp.add_argument("--policy", help="decision policy YAML (see examples/policy.yaml)")
    rp.add_argument("--sample", type=float, default=1.0, help="share of turns to replay")
    rp.add_argument("--max-pairs", type=int)
    rp.add_argument("--max-tokens", type=int, default=1024)
    rp.add_argument("--judge-max-tokens", type=int, default=1536)
    rp.add_argument("--judge-reasoning", default="low", help="reasoning effort, '' to omit")
    rp.add_argument("--budget", type=float, default=1.0, help="hard spend cap in USD")
    rp.add_argument("--cache-only", action="store_true", help="no network calls; replay from cache")
    rp.add_argument("--workers", type=int, default=8)
    rp.add_argument("--seed", type=int, default=0)
    fe = sub.add_parser("fetch", help="download traces")
    fe.add_argument("source", choices=["wildchat"])
    fe.add_argument("--n", type=int, default=2000, help="conversations")
    fe.add_argument("--out", default="data/wildchat.jsonl")
    fe.add_argument("--language", action="append", help="keep only these languages")
    fe.add_argument("--seed", type=int, default=0)
    ex = sub.add_parser("export-run", help="turn a saved live run into a trace file")
    ex.add_argument("run_dir")
    ex.add_argument("--out", default="runs/demo/traces.jsonl")
    ss = sub.add_parser("sample-size")
    ss.add_argument("--alpha", type=float, default=0.02)
    ss.add_argument("--delta", type=float, default=0.05)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        _run(a.config)
    elif a.cmd == "report":
        _report(a.run_dir, a.delta, a.open)
    elif a.cmd == "replay":
        _replay(a)
    elif a.cmd == "fetch":
        from safeswap.data.wildchat import fetch

        n = fetch(a.n, a.out, a.seed, languages=a.language)
        print(f"wrote {n} sessions -> {a.out}")
    elif a.cmd == "export-run":
        from safeswap.data.export import export_run

        print(f"wrote {export_run(a.run_dir, a.out)} sessions -> {a.out}")
    else:
        _sample_size(a.alpha, a.delta)


if __name__ == "__main__":
    main()
