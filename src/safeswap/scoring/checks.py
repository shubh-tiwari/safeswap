"""Deterministic checks: cheap, exact, run on both answers before any judge call.

Each check returns True (pass), False (fail) or None (doesn't apply to this request).
"""

from __future__ import annotations

import json
import re

_REFUSAL = re.compile(
    r"\b(I can(?:'|no)t (?:help|assist|provide|comply)|I'm (?:sorry|unable)|I am (?:sorry|unable)"
    r"|as an AI(?: language model)?|I won't be able to)\b",
    re.IGNORECASE,
)
_SINGLE_CHOICE = re.compile(r"print only a single choice", re.IGNORECASE)
_LETTER = re.compile(r"^\W*\(?([A-D])\)?\W*$")
_ASKS_JSON = re.compile(r"\b(json|JSON)\b")


def _extract_json(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    return m.group(1) if m else text


def not_empty(request: str, answer: str, truncated: bool) -> bool | None:
    return bool(answer.strip())


def not_truncated(request: str, answer: str, truncated: bool) -> bool | None:
    return not truncated


def no_refusal(request: str, answer: str, truncated: bool) -> bool | None:
    return not _REFUSAL.search(answer[:400])


def valid_json(request: str, answer: str, truncated: bool) -> bool | None:
    if not _ASKS_JSON.search(request):
        return None
    try:
        json.loads(_extract_json(answer).strip())
        return True
    except (ValueError, TypeError):
        return False


def single_choice(request: str, answer: str, truncated: bool) -> bool | None:
    if not _SINGLE_CHOICE.search(request):
        return None
    return bool(_LETTER.match(answer.strip().strip("*")))


CHECKS = {
    "not empty": not_empty,
    "not truncated": not_truncated,
    "no refusal": no_refusal,
    "valid JSON (when asked)": valid_json,
    "single-letter answer (when asked)": single_choice,
}


def run(request: str, answer: str, truncated: bool = False) -> dict[str, bool | None]:
    return {name: fn(request, answer, truncated) for name, fn in CHECKS.items()}
