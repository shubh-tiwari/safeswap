"""Live-path tests with a fake client: no network, no keys."""

import threading

import numpy as np
import pandas as pd

from safeswap.backends import Completion, SpendCapExceeded
from safeswap.judge import LLMJudge
from safeswap.monitor import run_live
from safeswap.policies import Decision
from safeswap.report import summarize


class FakeClient:
    """Small model answers 'x)' (right letter, extra text) or a wrong letter; reference answers 'x'.
    The judge prefers the reference on content only when letters differ, and on format whenever
    the served answer has extra text."""

    def __init__(self, cap=10.0, cost=0.01):
        self.spent_usd, self.cap, self.cost = 0.0, cap, cost
        self._lock = threading.Lock()

    def complete(self, model, messages, max_tokens=1024, **_):
        with self._lock:
            if self.spent_usd >= self.cap:
                raise SpendCapExceeded("cap")
            self.spent_usd += self.cost
        text = messages[0]["content"]
        if model == "judge":
            a = text.split("<answer_a>")[1].split("</answer_a>")[0].strip()
            b = text.split("<answer_b>")[1].split("</answer_b>")[0].strip()
            content = "TIE" if a[0] == b[0] else ("A" if a[0] == "C" else "B")
            fmt = "TIE" if len(a) == len(b) else ("B" if len(a) > len(b) else "A")
            out = f"thinking...\nCONTENT: {content}\nFORMAT: {fmt}"
        elif model == "small":
            q = int(text.split()[-1])
            out = "D" if q % 4 == 0 else "C)"  # every 4th wrong; others right with format slip
        else:
            out = "C"
        return Completion(model, out, 10, 5, self.cost, 1.0, truncated=False)


def _df(n):
    return pd.DataFrame(
        {"id": [f"r{i}" for i in range(n)], "prompt": [f"q {i}" for i in range(n)], "segment": "s"}
    )


def test_judge_separates_format_from_content():
    j = LLMJudge(FakeClient(), "judge")
    slip = j.label("q", "C)", "C")
    assert slip.error == 0 and slip.detail["format_worse"] and slip.detail["parsed"]
    wrong = j.label("q", "D", "C")
    assert wrong.error == 1 and not wrong.detail["format_worse"]
    assert j.label("q", "C", "C").detail["identical"]


def test_run_live_parallel_logs_errors_and_format_separately():
    client = FakeClient()
    df = _df(40)
    dec = Decision(np.full(40, "small", dtype=object))
    ev = run_live(
        df,
        dec,
        "large",
        {"rate": 1.0},
        client,
        LLMJudge(client, "judge"),
        workers=4,
        log=lambda *_: None,
    )
    assert list(ev["request_id"]) == list(df["id"])  # original order kept
    assert ev["error"].mean() == 0.25  # every 4th answer wrong
    assert ev["format_error"].mean() == 0.75  # the rest are format slips only
    s = summarize(ev)
    assert s["error"]["hajek_cp"]["lo"] < 0.25 < s["error"]["hajek_cp"]["hi"]
    assert s["format_error"]["value"] == 0.75


def test_run_live_stops_at_spend_cap():
    client = FakeClient(cap=0.2, cost=0.01)  # room for ~20 calls; each request needs 4
    dec = Decision(np.full(40, "small", dtype=object))
    ev = run_live(
        _df(40),
        dec,
        "large",
        {"rate": 1.0},
        client,
        LLMJudge(client, "judge"),
        workers=4,
        log=lambda *_: None,
    )
    assert len(ev) < 40
    assert client.spent_usd <= 0.2 + 4 * 0.01  # overshoot bounded by calls in flight
