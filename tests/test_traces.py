import json

import pandas as pd

from safeswap.data import wildchat
from safeswap.data.traces import iter_turns, read_jsonl, request_text, write_jsonl
from safeswap.scoring import checks


def test_read_both_shapes(tmp_path):
    p = tmp_path / "t.jsonl"
    session = {
        "session_id": "s1",
        "metadata": {"language": "English"},
        "turns": [
            {"messages": [{"role": "user", "content": "hi"}], "output": "hello", "model": "m"},
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "2+2?"},
                ],
                "output": "4",
                "model": "m",
            },
        ],
    }
    call = {
        "messages": [{"role": "user", "content": "q"}],
        "response": "a",
        "model": "m2",
        "cost_usd": 0.01,
        "metadata": {"feature": "faq"},
    }
    p.write_text(json.dumps(session) + "\n" + json.dumps(call) + "\n")
    sessions = read_jsonl(p)
    assert [len(s.turns) for s in sessions] == [2, 1]
    assert sessions[1].turns[0].attributes["cost_usd"] == 0.01
    units = list(iter_turns(sessions))
    assert [u.turn_index for u in units] == [1, 2, 1]
    assert request_text(units[0].turn) == "hi"  # single message passed verbatim
    assert "[assistant]" in request_text(units[1].turn)  # history rendered
    assert [u.session_id for u in iter_turns(sessions, {"feature": "faq"})] == ["2"]
    assert len(list(iter_turns(sessions, {"model": "m"}))) == 2


def test_write_read_roundtrip(tmp_path):
    p = tmp_path / "t.jsonl"
    call = {"messages": [{"role": "user", "content": "q"}], "response": "a"}
    p.write_text(json.dumps(call) + "\n")
    write_jsonl(read_jsonl(p), tmp_path / "o.jsonl")
    assert read_jsonl(tmp_path / "o.jsonl")[0].turns[0].output == "a"


def test_wildchat_conversion_drops_location_fields():
    df = pd.DataFrame(
        {
            "conversation_hash": ["h1"],
            "model": ["gpt-4-0314"],
            "language": ["English"],
            "turn": [2],
            "toxic": [False],
            "country": ["X"],
            "hashed_ip": ["abc"],
            "conversation": [
                [
                    {"role": "user", "content": "a", "country": "X", "hashed_ip": "abc"},
                    {"role": "assistant", "content": "b", "country": "X"},
                    {"role": "user", "content": "c"},
                    {"role": "assistant", "content": "d"},
                ]
            ],
        }
    )
    [s] = wildchat.to_sessions(df)
    assert [t.output for t in s.turns] == ["b", "d"]
    assert s.turns[1].messages[-1] == {"role": "user", "content": "c"}
    blob = json.dumps([t.__dict__ for t in s.turns]) + json.dumps(s.metadata)
    assert "hashed_ip" not in blob and "country" not in blob


def test_checks():
    mc = 'Pick one. Print only a single choice from "A" or "B" or "C" or "D" without explanation.'
    assert checks.run(mc, "B")["single-letter answer (when asked)"] is True
    assert checks.run(mc, "B) because")["single-letter answer (when asked)"] is False
    assert checks.run("hello", "hi")["single-letter answer (when asked)"] is None
    assert checks.run("Return JSON", '```json\n{"a": 1}\n```')["valid JSON (when asked)"] is True
    assert checks.run("Return JSON", "{a: 1}")["valid JSON (when asked)"] is False
    assert checks.run("q", "I'm sorry, I can't help with that.")["no refusal"] is False
    assert checks.run("q", "", truncated=True)["not empty"] is False
    assert checks.run("q", "x", truncated=True)["not truncated"] is False
