# Measuring what a cost change costs in quality

*safeswap: write-up of Phases 0-3. Shubham Tiwari, 2026-10-03. Everything here ran offline on
RouterBench with no model API calls. Details and raw numbers: docs 01-03 and `runs/`.*

## TL;DR

- Teams cut LLM cost with model swaps, routers and caches, and judge the quality impact by vibes
  and complaint tickets. The missing piece is not another router; it is **an honest estimate of
  realised quality loss on live traffic**, with an interval you can trust.
- A small uniform shadow sample (1-3% of the cheaper-served traffic) plus inverse-propensity
  weighting gives that estimate. Clopper-Pearson intervals on the effective sample size reach
  their nominal 95% coverage. Normal and bootstrap intervals run 1-4 points short.
- A quality budget is only meaningful with a bound. Picking a router threshold on observed error
  breaks a 2-5% error budget in **39-64%** of deployments. Learn-then-Test with fixed-sequence
  testing breaks it in **≤5%**, as promised, while keeping **80-92%** of the best achievable savings.
- A guarantee only certifies the traffic it was calibrated on. A silent provider-side model swap
  tripled the error. Input-drift monitors missed it **98-99%** of the time; a label CUSUM caught it
  **95-97%** of the time within ~1,500-3,000 requests. Monitoring plus recalibration restored the
  budget in every run.

## 1. Problem

Cost changes to an LLM system (cheaper model, router, semantic cache, prompt compression) trade
quality for money. In 2026 routing itself is commoditised: Bedrock and Azure ship routers,
NVIDIA's NeMo Switchyard and LiteLLM's Auto Router v2 are open source. None of them tell you how
much quality you actually lost on *your* traffic, and none keep telling you as traffic and
providers change.

Question this project answers: **"What did this cost change cost us in quality, how sure are we,
and when does that answer stop being true?"**

Definition used throughout: a request's *error* is 1 when the served answer is worse than what the
reference (large) model would have given. Realised error = the share of all traffic with error 1.

## 2. Approach

```
request ─▶ policy (router / cache / cheaper model) ─▶ served answer ─▶ user
                     │
                     └─ with prob p_i: also call reference + judge ─▶ label e_i, logged with p_i
                                                                          │
                     event log ─▶ IPW estimate ± CI ─▶ report, CUSUM alarm ─▶ recalibrate
```

Design decisions, and why:

| Decision | Why |
|---|---|
| Measure *realised* error relative to the reference, not absolute accuracy | It is what a cost change actually changes, and it needs no gold labels |
| Uniform shadow sampling with recorded p_i | Makes the estimate unbiased whatever the policy does. Active sampling turned out not to help (§4.2) |
| Requests served by the reference model treated as an exact census | Folding them into weights faked a 100x larger sample (§4.1) |
| Clopper-Pearson on Kish effective n as the default interval | Only interval that held nominal coverage at low error rates |
| Learn-then-Test, fixed-sequence along the cost path | Guarantee without Bonferroni's loss of savings, because risk is monotone along the path |
| Label-based CUSUM as the quality alarm; input drift only as a "traffic changed" hint | Input monitors are blind to provider-side changes and raise false quality alarms (§4.4) |
| Offline replay with ground truth before any paid run | Every statistical claim was checked against known truth for free |

## 3. Setup

- Data: RouterBench, 36,497 prompts across ~60 task sets, each with the answers, 0/1 scores and
  costs of 11 models. Scores give exact ground-truth labels, so every estimate can be checked.
- Reference: GPT-4 (1106). Small: Mixtral-8x7B (Mistral-7B as the "silent swap").
- Router: logistic regression on prompt text only (TF-IDF), predicting P(small worse). A 20%
  slice trains it; everything is evaluated on the held-out 80%.
- The models are 2023-era. The *relationships* (a cheap model losing on a hard-to-predict subset)
  are what the statistics depend on, not the model names.

## 4. Findings

### 4.1 Honest intervals need care: a precision bug worth remembering
Requests served by the reference itself have known error 0 (p = 1). Including them in a weighted
interval inflated the effective sample from 40 to 4,252, giving [0.77%, 1.41%] from 40 real
labels. Treating them as a census and estimating only the sampled stratum gives the honest
[0.39%, 2.52%]. Over 300 redraws per cell, Clopper-Pearson on effective n covered at 95-98%.
Normal and bootstrap intervals covered at 91-96%. *(doc 01 §3-4)*

### 4.2 Active sampling: a negative result with a reason
Sampling more where the router is uncertain did not narrow intervals. Theory says the best-case
gain from a risk signal m_i is (E√m)² / E[m]. The router has already chosen low-risk requests for
the cheap model, so their predicted risk sits in a narrow band, and the best possible gain is
**0.6%** narrower intervals. Even perfect knowledge would give only 26%. Uniform is the right
default. *(doc 01 §6)*

### 4.3 Routing is hard; the measurement layer is not optional
The text router ranks failures at AUROC 0.68 (TF-IDF and embeddings alike; a task-name prior
gets 0.66). At a ~2.5% error budget it saves 9% where an oracle saves 58%. Router savings claims
need independent measurement. *(doc 01 §5)*

### 4.4 Drift: input monitors watch the wrong thing
200 runs per scenario, detector thresholds set to a 5% false-alarm rate on no-drift runs.

| | silent model swap (error 4.6% → 16.1%) | harder task mix (error 4.6% → 3.5%) |
|---|---|---|
| label CUSUM | caught 97%, ~3 windows | quiet (correct) |
| input embedding / action mix | caught 1-2% | fired immediately (wrong alarm) |

At 1% shadowing (~1.4 labels per 500 requests) CUSUM still caught 95% of swaps; a per-window
CI test caught 57%. When labels are scarce, accumulate evidence across windows. *(doc 02)*

### 4.5 Guarantees: what they promise, what they cost, and when they expire
At δ = 5%, P(budget broken): naive 39-64%, fixed-sequence 1.4-4.8%, Bonferroni 0%.
Fixed-sequence kept 80% (α = 2%) and 92% (α = 5%) of hindsight-best savings with 3,000 labels;
Bonferroni kept 53% and certified nothing at 300 labels. Labels needed to certify α = 2% grow
sharply as true error nears the budget: 569 labels at 1%, 2,205 at 1.5%, 13,477 at 1.8%.

Full loop under a silent swap (α = 5%): unmonitored, the policy ran at 14.7% error in 100% of
windows while reporting 14.7% "savings". Monitored, CUSUM tripped, the policy fell back to the
reference, recalibrated on 2,000 fresh labels, and settled at 3.8% error, **within budget in
every run**, with 3.1% honest savings. *(doc 03)*

## 5. What I would do in production

1. Ship the monitor before the optimiser: uniform 1-3% shadowing of cheaper-served traffic,
   census for reference-served, Clopper-Pearson-on-n_eff intervals, net savings reported *after*
   measurement cost.
2. Calibrate any threshold with fixed-sequence LTT; refuse to deploy settings that can't be
   certified with the labels available, and say so.
3. Treat a provider model-version change as a drift event by default. Run label CUSUM
   continuously; use input drift only to trigger a re-check.
4. On alarm, fall back to the reference and recalibrate. Budget the recalibration labels up
   front (~2,000 at α = 5%).
5. Validate the judge before trusting any of it (next section). Every guarantee is about
   *judge-rated* error.

## 6. Limits

- Ground truth here is a task metric. With an LLM judge, judge bias moves the baseline and judge
  noise widens intervals; the guarantee becomes "≤ α as rated by this judge".
- Every guarantee assumes calibration traffic resembles production. Drift detection reduces
  the exposure but can't remove it (a few windows over budget before the alarm).
- Cache tiers weren't tested: RouterBench has no repeated prompts.
- 2023-era models. A live replication with current models is the remaining step.

## 7. Next: the paid phase (needs approval)

| Step | What | Est. API cost | Spend cap |
|---|---|---|---|
| P1 Live smoke | 20 prompts end to end with current models | ~$0.10 | $0.50 |
| P2 Judge validation | Judge 1,000 existing RouterBench Mixtral/GPT-4 answer pairs against task-metric truth (agreement, position bias); hand-label 200 open-ended pairs | ~$1.60 | $3 |
| P3 Live replication | 2,000 prompts fully labelled with current models; rerun coverage (§4.1) and calibration (§4.5) | ~$10.30 | $15 |
| P4 Live swap (optional) | Swap the small model for another on 1,000 prompts; rerun CUSUM detection | ~$1.70 | $3 |
| **Total** | | **~$14** | **$25** (headroom for longer answers) |

Assumptions: large = `anthropic/claude-sonnet-5.5` ($2/$10 per M tokens), small =
`qwen/qwen3-30b-a3b-instruct-2507` ($0.048/$0.193), judge = `google/gemini-3.6-flash`
($0.75/$3.75), all from OpenRouter's public price list on 2026-10-03. ~250 prompt tokens, ~300
answer tokens, ~1,000-token judge prompts, two judge calls per pair. About $0.005 per fully
labelled prompt, ~70% of it the large model's answer. If answers average 800 tokens, the total
roughly doubles (~$28). With Opus 5.5 as the reference, the large-model share roughly doubles.
The judge must run without reasoning tokens (or with a larger `max_tokens`), or its cost and
parse rate change.

### Time budget

| Item | Who | Effort | Wall clock |
|---|---|---|---|
| Add concurrent calls to `backends.py` (thread pool, rate-limit backoff) + judge-validation and live-replication scripts | Claude, in session | ~2-3 h | same day |
| P1 + P2 API runs | machine | — | ~15 min (P2 is ~2,000 short judge calls) |
| P3 API runs: 2,000 prompts x 4 calls = 8,000 calls | machine | — | ~6-7 h sequential at ~12 s/prompt; ~1 h with 8 workers |
| Hand-label 200 judge pairs (open-ended segments) | you | ~2 min/pair ≈ 7 h | split over 3-4 sittings |
| Read results, update docs 01-04 | you + Claude | ~3-4 h | — |
| **Total** | | **~12-15 h of your time**, mostly labelling | **~1 week part-time** |

For the learning goal, also budget ~2-3 weeks of part-time study to re-derive the pieces
yourself: IPW/Hájek variance, Clopper-Pearson, Kish n_eff, Neyman allocation, Page's CUSUM and
Learn-then-Test. That is where the staff-level depth comes from, more than the code.
