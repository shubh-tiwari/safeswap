"""Cost changes under measurement. Each policy decides, per request, which model serves it.

`choose(df) -> Decision` works on a whole replay at once (offline). `uncertainty` is the policy's
own guess of how likely its choice is worse than the reference; active sampling spends labels
there. Policies that have no such signal return None and get uniform sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Decision:
    model: np.ndarray  # model id serving each request
    uncertainty: np.ndarray | None = None


class Always:
    def __init__(self, model: str):
        self.model = model

    def choose(self, df: pd.DataFrame) -> Decision:
        return Decision(np.full(len(df), self.model, dtype=object))


class RandomMix:
    """Send a fixed share of traffic to the small model at random (a naive 'router')."""

    def __init__(self, small: str, large: str, small_share: float, seed: int = 0):
        self.small, self.large, self.share, self.seed = small, large, small_share, seed

    def choose(self, df: pd.DataFrame) -> Decision:
        rng = np.random.default_rng(self.seed)
        pick = rng.random(len(df)) < self.share
        return Decision(np.where(pick, self.small, self.large).astype(object))


class Oracle:
    """Cheapest model that is not worse than the reference: the upper bound on savings."""

    def __init__(self, small: str, large: str):
        self.small, self.large = small, large

    def choose(self, df: pd.DataFrame) -> Decision:
        ok = df[f"score:{self.small}"].to_numpy() >= df[f"score:{self.large}"].to_numpy()
        return Decision(np.where(ok, self.small, self.large).astype(object))


class SegmentPrior:
    """Route whole task segments to the small model when its historical win rate is high enough.

    Fitted on a separate calibration slice; a realistic, cheap, *imperfect* router whose
    uncertainty (1 - per-segment non-loss rate) is informative for active sampling.
    """

    def __init__(self, small: str, large: str, threshold: float = 0.95):
        self.small, self.large, self.threshold = small, large, threshold
        self.rate: dict[str, float] = {}
        self.default = 0.0

    def fit(self, calib: pd.DataFrame) -> SegmentPrior:
        ok = calib[f"score:{self.small}"] >= calib[f"score:{self.large}"]
        self.rate = ok.groupby(calib["segment"]).mean().to_dict()
        self.default = float(ok.mean())
        return self

    def choose(self, df: pd.DataFrame) -> Decision:
        r = df["segment"].map(self.rate).fillna(self.default).to_numpy()
        model = np.where(r >= self.threshold, self.small, self.large).astype(object)
        unc = np.where(model == self.small, 1 - r, 0.0)
        return Decision(model, unc)


class LearnedGate:
    """Predict P(small answer worse than reference) from the request text; serve small when the
    prediction is <= threshold. The prediction doubles as the uncertainty for active sampling.

    Trained on a labelled slice (offline: RouterBench scores; live: shadow labels). Text features
    only, so it can't cheat with the task-set name.
    """

    def __init__(
        self,
        small: str,
        large: str,
        threshold: float = 0.2,
        features: str = "tfidf",
        C: float = 1.0,
    ):
        self.small, self.large, self.threshold = small, large, threshold
        self.features_kind, self.C = features, C

    def fit(self, calib: pd.DataFrame) -> LearnedGate:
        from sklearn.linear_model import LogisticRegression

        from safeswap.routing import features

        self.feat = features.make(self.features_kind)
        x = self.feat.fit_transform(calib["prompt"])
        y = (calib[f"score:{self.small}"] < calib[f"score:{self.large}"]).astype(int)
        self.clf = LogisticRegression(C=self.C, max_iter=2000).fit(x, y)
        return self

    def predict_worse(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self.feat.transform(df["prompt"]))[:, 1]

    def choose(self, df: pd.DataFrame, p_worse: np.ndarray | None = None) -> Decision:
        q = self.predict_worse(df) if p_worse is None else p_worse
        model = np.where(q <= self.threshold, self.small, self.large).astype(object)
        return Decision(model, np.where(model == self.small, q, 0.0))


def build(spec: dict, small: str, large: str, calib: pd.DataFrame | None = None):
    kind = spec["kind"]
    if kind == "always_small":
        return Always(small)
    if kind == "always_large":
        return Always(large)
    if kind == "random_mix":
        return RandomMix(small, large, spec.get("small_share", 0.5), spec.get("seed", 0))
    if kind == "oracle":
        return Oracle(small, large)
    if kind == "segment_prior":
        if calib is None:
            raise ValueError("segment_prior needs a calibration slice")
        return SegmentPrior(small, large, spec.get("threshold", 0.95)).fit(calib)
    if kind == "learned_gate":
        if calib is None:
            raise ValueError("learned_gate needs a calibration slice")
        return LearnedGate(
            small,
            large,
            spec.get("threshold", 0.2),
            spec.get("features", "tfidf"),
            spec.get("C", 1.0),
        ).fit(calib)
    raise ValueError(f"unknown policy kind: {kind}")
