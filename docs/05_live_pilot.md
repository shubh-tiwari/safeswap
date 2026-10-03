# 05 — Live pilot (50 prompts, paid)

Date: 2026-10-03. Total spend **$0.21** of a $0.60 budget.

Models (OpenRouter): small `qwen/qwen3-30b-a3b-instruct-2507`, reference
`anthropic/claude-sonnet-5.5`, judge `google/gemini-3.6-flash` (reasoning effort low).

| Step | Cost |
|---|---|
| Probes (one call per model + judge options) | $0.001 |
| Judge check: 50 RouterBench pairs with known truth (`experiments/11_judge_check.py`) | $0.103 |
| Live pilot: 50 RouterBench prompts, all answered by Qwen, all checked (`experiments/10_live_smoke.yaml`) | $0.103 |

## Setup problems the probes caught

- **Gemini 3.6 Flash can't turn reasoning off.** With `max_tokens=16` it spent its budget
  thinking and returned empty text. Every verdict would have silently become "tie". The fix is
  reasoning effort "low" with `max_tokens=1024`: ~110 output tokens and ~$0.0005 per call. Unreadable
  replies are now recorded as `PARSE_FAIL`, never as a tie.
- `backends.py` now stops if a response has no reported cost, since the spend cap depends on it.
- Llama 4 Maverick ignored the verdict format, so it's not usable as a judge with this prompt.

## Judge check (50 balanced pairs, Mixtral vs GPT-4 answers from RouterBench)

- Parsed 100%, same verdict in both orders 96%.
- Truly worse (task score lower): judge said worse 24/25.
- Not worse: judge said worse 17/25. 15 of those are cases where **both models were right** but
  Mixtral wrote `A)` instead of `A` after being told to print only the letter.
- Agreement with the task score: 64% [49%, 77%]. Leaving out format-only differences, it agrees
  almost everywhere.

**The judge measures a stricter "worse" than the task score:** it counts instruction-following.
Neither is wrong; the error definition has to be chosen on purpose and stated with any number.

## Live pilot (Qwen vs Sonnet, 50 prompts)

- 27/50 answers identical (mostly multiple choice), so no judge call was needed.
- Judge said Qwen was worse on 10/50: **20%, 95% range [10%, 34%]**.
  - 5 format only: right letter, plus option text despite "print only a single choice".
  - 4 a different option chosen.
  - 1 unfinished: a Chinese riddle where Qwen hit the 512-token cap before giving its answer.
- Content-level loss is therefore ~4-5 of 50 (8-10%), and the rest is format.
- Cost: Qwen served all 50 for $0.0012, versus an estimated $0.053 for Sonnet (97.7% cheaper).
  Checking every request cost $0.10, so the pilot's net savings are negative by design. At a 1-3%
  check rate, measurement cost would be ~1-3% of that.

## Bug found and fixed

With every request checked (p = 1), the report treated all of them as exactly known and printed
a zero-width range [20%, 20%]. Only requests *served by the reference model* are known exactly;
checked cheap-model answers are a sample. Events now carry `by_reference`, and the census uses
it (with a fallback for older logs). Test: `test_fully_checked_cheap_answers_still_get_an_interval`.

## What it means for a bigger run

- Decide the error definition first. Either tell the judge to ignore formatting, or keep it and
  report format errors separately. Otherwise half the "errors" are formatting.
- RouterBench prompts are mostly multiple choice, where answers are often identical. Open-ended
  prompts would exercise the judge much more.
- Raise `max_tokens` for the small model, or count truncation as its own failure type.
