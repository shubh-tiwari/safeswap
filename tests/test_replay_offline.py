"""Replay end to end from a seeded call cache: real client in cache-only mode, no network."""

import pytest

from safeswap.data.traces import ReplayUnit, Turn
from safeswap.llm.cache import CallCache, key
from safeswap.llm.client import CacheMiss, OpenRouter
from safeswap.replay import html
from safeswap.replay.analysis import analyze
from safeswap.replay.decision import Policy
from safeswap.replay.engine import replay
from safeswap.scoring.judge import PROMPT, LLMJudge

CAND, JUDGE = "cand/model", "judge/model"
JUDGE_PARAMS = {"max_tokens": 1536, "temperature": 0.0, "reasoning": {"effort": "low"}}


def _put(cache, model, content, text, params):
    msgs = [{"role": "user", "content": content}]
    cache.put(
        key(model, msgs, params),
        {
            "model": model,
            "text": text,
            "input_tokens": 10,
            "output_tokens": 5,
            "cost_usd": 0.001,
            "latency_ms": 500.0,
            "truncated": False,
        },
    )


def _seed(cache, prompt, cand_text, base_text, verdict1, verdict2):
    _put(cache, CAND, prompt, cand_text, {"max_tokens": 1024, "temperature": 0.0})
    if cand_text != base_text:
        a = PROMPT.format(request=prompt, a=cand_text, b=base_text)
        b = PROMPT.format(request=prompt, a=base_text, b=cand_text)
        _put(cache, JUDGE, a, f"CONTENT: {verdict1}\nFORMAT: TIE", JUDGE_PARAMS)
        _put(cache, JUDGE, b, f"CONTENT: {verdict2}\nFORMAT: TIE", JUDGE_PARAMS)


def _unit(i, prompt, base_text, lang):
    t = Turn(
        f"t{i}",
        [{"role": "user", "content": prompt}],
        base_text,
        "base/model",
        {"cost_usd": 0.01, "latency_ms": 1500.0},
    )
    return ReplayUnit(f"s{i}", 1, t, {"language": lang})


def test_cache_only_replay_end_to_end(tmp_path):
    cache = CallCache(tmp_path / "calls.sqlite")
    units = []
    for i in range(40):
        prompt, base = f"question {i}", f"answer {i}"
        if i % 4 == 0:  # candidate worse in both orders -> loss
            _seed(cache, prompt, "bad", base, "B", "A")
        elif i % 4 == 1:  # orders disagree -> tie
            _seed(cache, prompt, "meh", base, "A", "A")
        else:  # identical answer -> tie without judge calls
            _seed(cache, prompt, base, base, None, None)
        units.append(_unit(i, prompt, base, "hi" if i < 20 else "en"))
    client = OpenRouter(api_key=None, cache=cache, cache_only=True)
    judge = LLMJudge(client, JUDGE, max_tokens=1536)
    pairs, stats = replay(units, CAND, client, judge, workers=4, log=lambda *_: None)
    assert stats["cache_miss"] == 0 and stats["pairs"] == 40
    assert client._http is None and client.spent_usd == 0.0  # nothing went to the network
    assert (pairs["content"] == "loss").sum() == 10
    assert pairs["identical"].sum() == 20
    a = analyze(pairs, Policy(min_samples=10), {"baseline": "base/model", "candidate": CAND})
    assert a.overall.losses == 10 and a.cost["change_pct"] < -80
    page = html.write(a, pairs, tmp_path, 1.0).read_text()
    assert "Swap report" in page and "language" in page


def test_cache_only_raises_without_network(tmp_path):
    client = OpenRouter(api_key=None, cache=CallCache(tmp_path / "c.sqlite"), cache_only=True)
    with pytest.raises(CacheMiss):
        client.complete(CAND, [{"role": "user", "content": "x"}])
    assert client._http is None
