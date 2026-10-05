"""Turn paired estimates into a verdict: ship, hold, rollback or split.

Policy file (YAML), all keys optional:

    decision:
      max_quality_drop_pp: 1.0        # ship only if the CI lower bound of the delta is above -1 pp
      min_samples: 1000               # fewer pairs -> hold
      block_if_any_slice_drop_pp: 5.0 # a slice this much worse (and significant) blocks a full ship
      min_slice_size: 30              # smaller slices are shown but never flagged
      fdr_q: 0.1                      # Benjamini-Hochberg level across slices

Verdicts:
  ship      candidate is no worse than the allowed drop, overall and in every large slice
  split     some slices are clearly worse; keep the baseline there, ship the candidate elsewhere
  rollback  candidate is clearly worse than the allowed drop (don't ship, or revert)
  hold      not enough evidence either way
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from safeswap.stats.paired import PairedResult


@dataclass
class Policy:
    max_quality_drop_pp: float = 1.0
    min_samples: int = 1000
    block_if_any_slice_drop_pp: float = 5.0
    min_slice_size: int = 30
    fdr_q: float = 0.1

    @classmethod
    def load(cls, path: str | Path | None) -> Policy:
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text()) or {}
        raw = raw.get("decision", raw)
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Verdict:
    verdict: str  # ship | split | rollback | hold
    reason: str
    keep_on_baseline: list[str] = field(default_factory=list)  # slice labels, for split
    rest: PairedResult | None = None  # estimate on traffic that would move, for split


def decide(
    overall: PairedResult,
    flagged: list[str],
    rest: PairedResult | None,
    policy: Policy,
) -> Verdict:
    """`flagged`: labels of slices that are significantly worse by at least the block threshold.
    `rest`: paired result on pairs outside all flagged slices (None if nothing flagged)."""
    max_drop = policy.max_quality_drop_pp / 100
    if overall.n < policy.min_samples:
        return Verdict("hold", f"only {overall.n} pairs; policy needs {policy.min_samples}")
    if flagged:
        if rest is not None and rest.n >= policy.min_slice_size and rest.delta_lo >= -max_drop:
            return Verdict(
                "split",
                f"{len(flagged)} slice(s) clearly worse; the rest is within "
                f"-{policy.max_quality_drop_pp:g} pp",
                flagged,
                rest,
            )
    elif overall.delta_lo >= -max_drop:
        return Verdict("ship", f"CI lower bound above -{policy.max_quality_drop_pp:g} pp")
    if overall.delta_hi < -max_drop:
        return Verdict(
            "rollback", f"whole CI below -{policy.max_quality_drop_pp:g} pp: candidate is worse"
        )
    why = "worse slices, and the rest isn't clearly fine" if flagged else "CI too wide to decide"
    return Verdict("hold", why, flagged, rest)
