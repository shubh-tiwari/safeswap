# 03 — A calibrated guarantee, and what keeps it true (offline, no API calls)

Date: 2026-10-03. Scripts: `experiments/05_guarantee.py`, `experiments/06_lifecycle.py`.
Raw output: `runs/05_guarantee.txt`, `runs/06_lifecycle.txt`. Code: `src/safeswap/calibrate.py`.

## Method

Policy: the learned text gate (Mixtral vs GPT-4). A request goes to Mixtral iff its predicted
P(worse) <= λ, and λ is chosen from a grid of 101 values in [0, 0.5]. End-to-end risk R(λ) = share
of *all* traffic served a worse answer. It grows with λ.

Learn-then-Test, fixed-sequence: test H0: R(λ) > α at each λ with an exact binomial p-value,
walking from the safest λ and stopping at the first non-rejection. Guarantee:
P(R(λ̂) > α) <= δ over the draw of calibration labels, assuming future traffic resembles the
calibration sample. Compared with the naive rule (cheapest λ with empirical risk <= α) and with
Bonferroni over the grid.

## 3a. Does the promise hold, and what does it cost? (500 repetitions, δ = 0.05)

Calibration pool of 14,583 and evaluation pool of 14,615 held-out requests. Best λ in hindsight:
6.4% savings at α = 2%, 15.6% at α = 5%.

| α | labels | method | P(violation) | savings kept (vs hindsight) |
|---|---|---|---|---|
| 2% | 300 | fixed-sequence | 1.4% | 33% |
| 2% | 300 | bonferroni | 0% | 0% (certifies nothing) |
| 2% | 300 | naive | **58.6%** | 121% |
| 2% | 3000 | fixed-sequence | 1.6% | 80% |
| 2% | 3000 | bonferroni | 0% | 53% |
| 2% | 3000 | naive | **39.2%** | 104% |
| 5% | 1000 | fixed-sequence | 4.0% | 82% |
| 5% | 3000 | fixed-sequence | 3.8% | 92% |
| 5% | 3000 | naive | **64.2%** | 104% |

- The naive rule breaks the budget in 39-64% of deployments. Its "extra" savings are exactly the
  savings from breaking the budget.
- Fixed-sequence testing keeps P(violation) under δ everywhere and, with 3,000 labels, keeps
  80-92% of the best achievable savings.
- Bonferroni is valid but throws away 20-100% more savings. With 300 labels at α = 2% it can't
  certify anything. Ordering the hypotheses along the cost path is what makes the guarantee affordable.
- The price of certainty falls with labels: 33% → 62% → 80% of hindsight savings at 300 → 1,000 →
  3,000 labels (α = 2%). This matches the sample-size table in doc 01.

## 3b. The full loop under a silent model swap (200 runs, α = 5%)

60 windows x 500 requests. At window 20 the "small" endpoint silently serves Mistral-7B.
Monitored policy: 3% shadowing of small-served requests → Bernoulli CUSUM (threshold set so ~5%
of no-swap runs alarm; observed 7%) → on alarm, fall back to always-GPT-4 while collecting 2,000
recalibration labels (small model run on 50% of fallback traffic) → recalibrate → resume.

| policy | period | true risk | P(risk > α) | savings | in fallback |
|---|---|---|---|---|---|
| both | before swap | 4.3% | 4.5% | 14.2% | ~0 |
| unmonitored | after swap (all) | **14.7%** | **100%** | 14.7% | 0 |
| monitored | swap +0-4 windows | 11.5% | 79% | 11.5% | 21% |
| monitored | swap +5-14 | 1.9% | 7% | 1.7% | 71% |
| monitored | swap +15-39 | 3.8% | **0%** | 3.1% | 4% |

- A guarantee is a statement about the calibration distribution. A silent provider change voids
  it, and without monitoring the policy runs at 3x its budget indefinitely, reporting healthy
  savings the whole time.
- The monitored loop pays for the breach with a few windows over budget (detection delay), a
  fallback period at zero savings, and then a lower but honest steady state. 3.1% savings at
  3.8% risk is what the weaker model can actually deliver within α.
- Before the swap, the observed P(risk > α) = 4.5% matches δ = 5%. The calibration is doing exactly
  what it claims.

## What this says

1. **Without a bound, a quality budget means nothing.** Picking a threshold on observed error
   breaks the budget about half the time.
2. **Statistical structure buys savings.** Fixed-sequence testing along the cost path recovers
   most of what Bonferroni throws away, at the same guarantee.
3. **A guarantee is only as durable as its monitor.** Calibration certifies the past; label-based
   drift detection (doc 02) is what tells you when the certificate expired, and recalibration
   restores it.

## Limits

- A 1-D gate threshold only. A cache → small → large ladder fits the same machinery once
  parameterised along one cost path, but RouterBench has no repeated prompts to test a cache on.
- Labels are task-metric ground truth. With an LLM judge the guarantee is about *judge-rated*
  error, so judge validation (paid phase) decides what the guarantee actually means.
- The fallback costs money: 2,000 recalibration labels mean running the small model and a judge
  on ~4,000 extra requests.
