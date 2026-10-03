# 02 — Drift: which monitors notice, and how fast (offline, no API calls)

Date: 2026-10-03. Script: `experiments/04_drift.py`. Raw output: `runs/04_drift_r*.txt`.

## Setup

- Stream of 40 windows x 500 requests from held-out RouterBench; shift at window 20.
- Policy: learned text gate (Mixtral vs GPT-4, threshold 0.2), never retrained. Baseline: 29% of
  traffic goes to Mixtral, with a 17.9% error rate among those requests (p0, from a one-off
  labelled reference split).
- Uniform shadowing of small-served requests.
- Scenarios:
  - **control**: no change.
  - **mix**: segments where Mixtral is weak become 4x more frequent.
  - **swap**: the "small" provider silently serves Mistral-7B. Inputs and router unchanged.
- Detectors. Each threshold is set on 200 control runs so that 5% of no-drift runs ever alarm:
  - label CUSUM: Bernoulli CUSUM on shadow labels, p0 → 2·p0.
  - window CI: a window's error lower bound is above the budget (1.5·p0).
  - action-mix |z|: the share of traffic going to the small model.
  - input embedding shift: distance of the window's mean MiniLM embedding from the reference.

## Results (200 runs per scenario, 3% shadow rate ≈ 4.3 labels per window)

What actually happened to quality (error over all traffic):
- mix: 4.6% → **3.5%**. Quality *improved*, because the router sends the harder prompts to GPT-4.
- swap: 4.6% → **16.1%**. Quality collapsed.

| detector | mix: detected | mix: delay | swap: detected | swap: delay |
|---|---|---|---|---|
| label CUSUM | 8% | 8 windows | **97%** | **3 windows** |
| window CI > budget | 5% | 6.5 | 95% | 3 |
| action-mix \|z\| | 98% | 0 | **1%** | — |
| input embedding shift | 98% | 0 | **2%** | — |

Shadow rate vs. how often the swap is caught, and the median delay:

| shadow rate | labels / window | label CUSUM | window CI |
|---|---|---|---|
| 1% | 1.4 | 95%, 6 windows | 57%, 7 windows |
| 3% | 4.3 | 97%, 3 | 95%, 3 |
| 10% | 14.5 | 98%, 1 | 97%, 0 |

## What this says

1. **Input-drift monitoring cannot see the failure that matters most.** A provider silently
   changing the model behind an endpoint tripled the error rate, and both label-free detectors
   missed it in 98-99% of runs. Only shadow labels saw it. That is the core argument for a
   measurement layer.
2. **Label-free detectors raise the wrong alarm.** They fired immediately on the task-mix shift,
   which actually *improved* quality. Useful as a cheap "traffic changed, re-check" trigger, but
   not as a quality alarm.
3. **Sequential statistics beat per-window tests when labels are scarce.** At 1% shadowing
   (~1.4 labels per window), CUSUM still caught 95% of swaps within ~3,000 requests. The window CI
   test caught 57%, because a single window never has enough labels to clear a confidence bound.
4. **Cost of vigilance.** 1% shadowing of small-served traffic is ~0.3% of all requests here, so
   detecting a silent model swap within a few thousand requests costs very little.

## Caveats

- p0 comes from a one-off fully labelled reference split. In practice it would come from an
  initial evaluation, and an error in p0 shifts CUSUM's false-alarm rate. Thresholds calibrated on
  simulated control runs assume the control stream looks like production.
- Labels here are task-metric ground truth. With an LLM judge, judge noise adds variance and any
  judge bias moves p0. Judge validation (paid phase) has to come before trusting live alarms.
