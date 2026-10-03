# 01 — Phase 0/1 findings (offline, RouterBench, no API calls)

Date: 2026-10-03. Data: RouterBench 0-shot (36,497 prompts, 11 models, task-score labels).
Reference ("large") model: gpt-4-1106-preview. Error = served answer scores lower than the reference.

## 1. Headroom (`experiments/01_oracle.py`)

| small model | always-small error | always-small savings | oracle savings |
|---|---|---|---|
| claude-instant-v1 | 26.0% | 92.9% | 69.1% |
| gpt-3.5-turbo-1106 | 25.1% | 92.6% | 67.3% |
| mixtral-8x7b-chat | 33.0% | 95.9% | 58.1% |
| mistral-7b-chat | 58.7% | 98.6% | 39.1% |

Plenty of headroom (oracle 40-70%), but always-small loses on a quarter to a half of requests, so
the router's quality loss is what has to be measured. The smoke router (segment prior, threshold
0.9) captured only 4.8% savings at 1.1% realised error: a long way from the oracle.

## 2. Labels needed to *certify* error <= alpha (`safeswap sample-size`)

At alpha = 2%, 95% confidence (one-sided Clopper-Pearson):

| true error | labels |
|---|---|
| 0% | 149 |
| 0.5% | 259 |
| 1.0% | 569 |
| 1.5% | 2,205 |
| 1.8% | 13,477 |

Certification gets expensive as the real error approaches the budget. A policy tuned to sit
just under its budget can't be certified on a small deployment's traffic.

## 3. A bug worth remembering: census rows fake precision

Requests served by the reference model have a known error of 0 (p = 1). Folding them into a
weighted Clopper-Pearson interval inflated the Kish effective sample size from 40 to 4,252,
giving [0.77%, 1.41%] from **40 real labels**. Treating them as an exact census and estimating
only the sampled stratum (`estimators.with_census`) gives the honest [0.39%, 2.52%].
Test: `test_census_stratum_does_not_shrink_interval`.

## 4. Do the intervals cover the truth? (`experiments/02_coverage.py`, 300 redraws each)

Segment-prior router (threshold 0.8, small share 28.7%), true error 4.11%. Full table: `runs/02_coverage.txt`.

| shadow rate (of small-served) | labels | hajek | hajek_cp | bootstrap |
|---|---|---|---|---|
| 1% uniform | 73 | 94.7% | 96.7% | 93.7% |
| 3% uniform | 235 | 94.7% | 97.0% | 95.7% |
| 5% uniform | 375 | 92.7% | 95.0% | 93.3% |
| 5% active | 382 | 91.3% | 97.0% | 93.3% |

- `hajek_cp` reaches the nominal 95% in every cell. Normal and bootstrap intervals slightly
  under-cover (91-95%), so use `hajek_cp` for anything you'd make a claim on.
- **Active sampling did not help** once the label budget was equal (an earlier run seemed to
  show it helping, because active sampling had silently been given ~3x the labels). The segment
  prior's uncertainty is too coarse to predict errors. Active sampling is only as good as the
  uncertainty signal behind it, so build a better one next (Phase 2 gate).

## 5. A learned gate (`experiments/03_gate.py`)

Logistic regression on prompt text only, predicting P(Mixtral worse than GPT-4). Trained on 20%,
evaluated on the held-out 80%.

| signal | AUROC |
|---|---|
| TF-IDF gate (word + char n-grams) | 0.676 |
| MiniLM embedding gate | 0.674 |
| segment prior (sees the task name) | 0.664 |

| gate threshold | small share | savings | realised error |
|---|---|---|---|
| 0.10 | 4.7% | 2.1% | 0.55% |
| 0.15 | 17.0% | 9.1% | 2.53% |
| 0.20 | 29.2% | 16.0% | 4.79% |
| 0.30 | 46.8% | 26.3% | 9.44% |

Oracle: 58% savings at 0% error. Text-only gates sit far below it, and richer features (embeddings)
don't move AUROC. Per-prompt failure is hard to predict from the prompt alone: a 0/1 task score is
noisy, and both models often fail or pass the same prompt by chance. At a 2% error budget a
realistic router saves single-digit percent on this data, so **savings claims from routers need
the measurement layer; they don't speak for themselves.**

## 6. Why active sampling can't help much here

Rerunning section 4 with the gate's prediction as the uncertainty signal: still no gain over
uniform (`runs/02_coverage_gate.txt`). Theory explains it. For an IPW mean of 0/1 labels the
variance-optimal design samples p_i ∝ sqrt(E[e_i^2]), and its best-case variance relative to
uniform is (E[sqrt m])^2 / E[m]:

| knowledge of each request's error | best-case variance vs uniform | CI width |
|---|---|---|
| gate's predicted probability | 0.989 | x0.994 |
| perfect (the true labels) | 0.549 | x0.741 |

The router already selected low-risk requests for the small model, so their predicted risk sits in
a narrow band (0.025-0.20, sd 0.037), leaving nothing for active sampling to exploit. **Uniform
shadowing is the right default; active sampling is worth it only with a much sharper signal.**
`sampler.active` now uses the sqrt allocation.

## Next (free, offline)

- ~~Phase 2: drift~~ done, see docs/02_drift.md.
- ~~Phase 3: calibrated guarantee~~ done, see docs/03_guarantee.md.

## Last (paid, needs approval)

- Live smoke run with an LLM judge (`experiments/10_live_smoke.yaml`) and judge validation
  against hand labels.
