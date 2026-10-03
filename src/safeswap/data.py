"""Replay traffic.

RouterBench (withmartian/routerbench) ships, for ~36.5k prompts across ~60 task sets, the answer,
task score (0/1 for most sets) and cost of 11 models. That gives ground-truth labels and exact
always-X costs for free, which is what the statistics experiments need. The models are 2023-era;
relative behaviour (a cheaper model losing on some prompts) is what matters here, not the names.

Every loader returns a DataFrame with: id, prompt, segment, and per model `score:<m>`,
`answer:<m>`, `cost:<m>` columns.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(".cache/hf")


def _prompt_text(p) -> str:
    if isinstance(p, str) and p.startswith("["):
        try:
            p = ast.literal_eval(p)
        except (ValueError, SyntaxError):
            return p
    if isinstance(p, (list, tuple)):
        return "\n\n".join(str(x) for x in p)
    return str(p)


def routerbench_models(df: pd.DataFrame) -> list[str]:
    return sorted(c.split(":", 1)[1] for c in df.columns if c.startswith("score:"))


def load_routerbench(
    n: int | None = None,
    seed: int = 0,
    shots: int = 0,
    segments: list[str] | None = None,
    mix: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Load RouterBench. `mix` samples a fixed number of prompts per segment; the special key
    "_multiple_choice" draws from all multiple-choice prompts not otherwise listed."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "withmartian/routerbench",
        f"routerbench_{shots}shot.pkl",
        repo_type="dataset",
        cache_dir=str(CACHE_DIR),
        token=False,  # public dataset: never send a saved HF token
    )
    raw = pd.read_pickle(path)
    models = [c.split("|")[0] for c in raw.columns if c.endswith("|total_cost")]
    out = pd.DataFrame(
        {
            "id": raw["sample_id"].astype(str),
            "prompt": raw["prompt"].map(_prompt_text),
            "segment": raw["eval_name"].astype(str),
        }
    )
    for m in models:
        out[f"score:{m}"] = raw[m].astype(float)
        out[f"answer:{m}"] = raw[f"{m}|model_response"].astype(str)
        out[f"cost:{m}"] = raw[f"{m}|total_cost"].astype(float)
    if segments:
        out = out[out["segment"].isin(segments)]
    out = out.dropna(subset=[c for c in out.columns if c.startswith(("score:", "cost:"))])
    if mix:
        is_mc = out["prompt"].str.contains("Print only a single choice", regex=False)
        parts = []
        for seg, k in mix.items():
            pool = (
                out[is_mc & ~out["segment"].isin(mix)]
                if seg == "_multiple_choice"
                else (out[out["segment"] == seg])
            )
            parts.append(pool.sample(n=min(k, len(pool)), random_state=seed))
        return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    if n is not None and n < len(out):
        out = out.sample(n=n, random_state=seed)
    return out.reset_index(drop=True)
