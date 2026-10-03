"""Labels: is the served answer worse than the reference answer for this request?

Two sources:
  * TaskMetricJudge: ground truth from a task score (e.g. RouterBench's per-model correctness).
    Free and exact; used to validate everything else.
  * LLMJudge: pairwise comparison by a model from a different family than the ones being
    routed. Asked in both orders to cancel position bias; the verdicts per order are kept so
    position-bias and inconsistency rates can be reported.

The LLM judge gives two verdicts per comparison: CONTENT (substance, the error that counts) and
FORMAT (following output-format instructions), so format slips are reported separately instead of
inflating the error rate.

Error convention everywhere: 1 = served answer worse than reference, 0 = not worse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .backends import OpenRouter

PROMPT = """You are comparing two answers to the same request.

<request>
{request}
</request>

<answer_a>
{a}
</answer_a>

<answer_b>
{b}
</answer_b>

Give two separate verdicts.

CONTENT: which answer is better on substance: correctness first, then completeness and
helpfulness. Ignore formatting, extra explanation and whether output-format instructions were
followed. If both reach the same answer, say TIE.

FORMAT: which answer better follows the request's instructions about output format (e.g. "print
only the letter", "put the answer in brackets"). If the request has no such instructions or both
follow them equally, say TIE.

Reply with exactly two lines:
CONTENT: A, B or TIE
FORMAT: A, B or TIE"""

_CONTENT = re.compile(r"CONTENT:\s*\**\s*(A|B|TIE)\b", re.IGNORECASE)
_FORMAT = re.compile(r"FORMAT:\s*\**\s*(A|B|TIE)\b", re.IGNORECASE)


@dataclass
class Label:
    error: int  # 1 = served worse than reference
    cost_usd: float = 0.0
    detail: dict = field(default_factory=dict)


class TaskMetricJudge:
    """Served is worse iff its task score is lower than the reference's."""

    def label(self, served_score: float, reference_score: float) -> Label:
        return Label(int(served_score < reference_score))


class LLMJudge:
    def __init__(
        self,
        client: OpenRouter,
        model: str,
        rule: str = "both",
        max_tokens: int = 1024,
        reasoning_effort: str | None = "low",
    ):
        if rule not in ("both", "either"):
            raise ValueError("rule must be 'both' or 'either'")
        self.client = client
        self.model = model
        self.rule = rule  # 'both': worse only if reference preferred in both orders
        # Room for hidden reasoning: some judges (e.g. Gemini 3.6 Flash) can't turn it off and
        # return empty text if max_tokens is spent before the verdict.
        self.max_tokens = max_tokens
        self.extra = {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}

    def _ask(self, request: str, a: str, b: str) -> tuple[str, str, float]:
        msg = [{"role": "user", "content": PROMPT.format(request=request, a=a, b=b)}]
        c = self.client.complete(self.model, msg, max_tokens=self.max_tokens, **self.extra)
        mc, mf = _CONTENT.search(c.text), _FORMAT.search(c.text)
        content = mc.group(1).upper() if mc else "PARSE_FAIL"
        fmt = mf.group(1).upper() if mf else "PARSE_FAIL"
        return content, fmt, c.cost_usd  # deployment cost, even if replayed from cache

    def _worse(self, v1: str, v2: str) -> bool:
        ref_wins = [v1 == "B", v2 == "A"]  # served is A in order 1, B in order 2
        return all(ref_wins) if self.rule == "both" else any(ref_wins)

    def label(self, request: str, served: str, reference: str) -> Label:
        if served.strip() == reference.strip():
            return Label(0, 0.0, {"identical": True, "format_worse": False})
        v1, f1, c1 = self._ask(request, served, reference)
        v2, f2, c2 = self._ask(request, reference, served)
        return Label(
            int(self._worse(v1, v2)),
            c1 + c2,
            {
                "order1": v1,
                "order2": v2,
                "format1": f1,
                "format2": f2,
                "format_worse": self._worse(f1, f2),
                "consistent": (v1, v2) in {("A", "B"), ("B", "A"), ("TIE", "TIE")},
                "parsed": "PARSE_FAIL" not in (v1, v2, f1, f2),
            },
        )
