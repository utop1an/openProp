# Latent-mechanism shift and Cox benchmark

Date: 2026-08-26

## Research question

The duration-shift benchmark changes follow-up support while keeping the
transition mechanism fixed. This benchmark instead asks what a source-trained
typed persistence model preserves when the latent transition rate, hazard
shape, or typed factor effects change at test time.

The experiment is synthetic failure analysis, not evidence of real-world
robustness. Generator truth is used only by an evaluation oracle and is never
available to fitting, calibration, query parsing, or entity matching.

## Paired protocol

For each of seeds 31, 41, 53, 67, and 79, all five test conditions share the
same context rows, group identifiers, unit-exponential latent draws, censoring
draws, and candidate order. Source training and validation partitions are also
shared across conditions. Each context has 80 source training, validation, and
test samples.

| Test condition | Mechanism |
|---|---|
| In distribution | Source rate and exponential shape 1.0 |
| Global rate acceleration | Every source hazard multiplied by 2 |
| Decreasing hazard shape | Source rates with Weibull shape 0.6 |
| Increasing hazard shape | Source rates with Weibull shape 1.6 |
| Typed factor reversal | Context hazard transformed to `0.12^2 / source_hazard` |

The compared source-trained models are factorized exponential, Weibull,
piecewise exponential, and Cox proportional hazards with a Breslow cumulative
baseline. All scale calibration uses source validation only; shifted test data
never selects a parameter.

Run:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_latent_mechanism_shift.py
```

The command writes `artifacts/latent_mechanism_shift_results.json`.

## Metric legality and oracle regret

Parametric models report event/censoring NLL, Harrell's C-index, and integrated
Brier score (IBS). Cox reports C-index and IBS only. A Breslow baseline is a
stepwise cumulative hazard and does not define the continuous event density
needed for the same event-time NLL; the benchmark therefore leaves Cox NLL
absent instead of inventing one.

Raw NLL and IBS are not directly comparable across target mechanisms because
the event entropy and censoring distribution change. The primary robustness
quantity is therefore same-test oracle regret:

- NLL and IBS regret = fitted metric minus generator-oracle metric;
- C-index regret = oracle C-index minus fitted C-index.

Positive regret means worse than the oracle on exactly the same target rows.
Small differences can still reflect finite-sample variation.

## Verified five-seed results

Values below are means across five seeds. The raw oracle column is shown to
make the target difficulty explicit.

| Condition | Model | Raw NLL | NLL regret | Raw C-index | C-index regret | Raw IBS | IBS regret |
|---|---|---:|---:|---:|---:|---:|---:|
| In distribution | Exponential | 1.285 | 0.010 | 0.733 | 0.018 | 0.081 | 0.002 |
|  | Weibull | 1.284 | 0.010 | 0.733 | 0.018 | 0.081 | 0.002 |
|  | Piecewise | 1.287 | 0.013 | 0.751 | 0.000 | 0.081 | 0.002 |
|  | Cox | n/a | n/a | 0.751 | 0.000 | 0.080 | 0.001 |
| Global acceleration | Exponential | 1.212 | 0.162 | 0.740 | 0.016 | 0.082 | 0.021 |
|  | Weibull | 1.217 | 0.166 | 0.740 | 0.016 | 0.083 | 0.021 |
|  | Piecewise | 1.212 | 0.161 | 0.756 | 0.000 | 0.083 | 0.021 |
|  | Cox | n/a | n/a | 0.756 | 0.000 | 0.071 | 0.010 |
| Decreasing shape | Exponential | 1.489 | 0.151 | 0.682 | 0.012 | 0.141 | 0.009 |
|  | Weibull | 1.499 | 0.161 | 0.682 | 0.012 | 0.141 | 0.009 |
|  | Piecewise | 1.491 | 0.153 | 0.693 | 0.000 | 0.141 | 0.009 |
|  | Cox | n/a | n/a | 0.693 | 0.000 | 0.140 | 0.008 |
| Increasing shape | Exponential | 1.166 | 0.127 | 0.758 | 0.026 | 0.045 | 0.007 |
|  | Weibull | 1.160 | 0.121 | 0.758 | 0.026 | 0.045 | 0.007 |
|  | Piecewise | 1.171 | 0.132 | 0.784 | 0.000 | 0.045 | 0.007 |
|  | Cox | n/a | n/a | 0.784 | 0.000 | 0.045 | 0.007 |
| Typed factor reversal | Exponential | 3.646 | 2.712 | 0.182 | 0.637 | 0.638 | 0.533 |
|  | Weibull | 3.686 | 2.751 | 0.182 | 0.637 | 0.639 | 0.534 |
|  | Piecewise | 3.666 | 2.731 | 0.181 | 0.638 | 0.637 | 0.533 |
|  | Cox | n/a | n/a | 0.181 | 0.638 | 0.659 | 0.555 |

The in-distribution oracle means are NLL 1.274, C-index 0.751, and IBS 0.079.
Under typed factor reversal they are NLL 0.934, C-index 0.819, and IBS 0.104.

## Supported interpretation

Cox is a competitive strong baseline: in distribution it matches the oracle
risk ordering and has the lowest non-oracle IBS. Piecewise and Cox preserve
oracle ordering under global rate and shape changes because those mechanisms do
not change the context ordering. The source-calibrated survival curves still
incur nonzero IBS regret, especially under global acceleration.

All four models fail catastrophically when typed factor effects reverse. Their
C-index is approximately 0.18 against an oracle value of 0.82, and their IBS
regret exceeds 0.53. This is an explicit boundary: typed factorization supports
compositional reuse under a stable mechanism, but provides no robustness
guarantee to arbitrary changes in factor meaning.

## Limits and next experiment

This benchmark uses known synthetic contexts, irreversible single transitions,
non-informative censoring, and no target-domain adaptation. It does not validate
real transition rates, unseen property vocabularies, noisy language parsing,
informative observation policies, or real-world grounding. The oracle cannot
be deployed and is included only to make cross-mechanism evaluation valid.

The next evidence priority remains a leakage-safe semi-real longitudinal
benchmark. The parallel target-adaptation experiment is now complete: an
outcome-independent calibration split and source-preserving sign gate detect
the exact reversal and repair held-out performance without target-test access.
See the [target-adaptation benchmark](target-adaptation-benchmark.md) for the
ten-seed sample-efficiency result and its deliberately narrow claim boundary.
