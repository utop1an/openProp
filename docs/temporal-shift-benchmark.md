# Temporal duration-shift benchmark

Date: 2026-08-26

## Research question

The Weibull misspecification experiment keeps the follow-up distribution fixed
across train, validation and test. This benchmark asks whether typed persistence
models trained from shorter histories retain calibrated survival estimates on
the same latent dynamics over a longer test horizon.

This is duration-support shift, not latent mechanism shift. The transition-rate
factors and Weibull shape stay stationary, and censoring remains
non-informative.

## Paired protocol

For every true shape and seed, two training conditions share an identical test
partition:

| Condition | Train follow-up | Validation follow-up | Test follow-up |
|---|---:|---:|---:|
| In distribution | 24 h | 24 h | 24 h |
| Duration shift | 6 h | 12 h | 24 h |

The models are factorized exponential, factorized Weibull, and factorized
piecewise-exponential with boundaries at 2 h and 4 h. Each condition fits its
own model and uses only its validation split for one global scale calibration.
NLL scores every test record. Integrated Brier score uses horizons
1, 4, 8, 12 and 18 h, extending well beyond the shifted training support.

Run:

    python scripts/evaluate_temporal_shift.py

The command writes `artifacts/temporal_shift_results.json`. The default
protocol uses shapes 0.6 and 1.6, seeds 31, 41, 53, 67 and 79, and 80 samples per
context.

## Verified five-seed shifted results

| True shape | Model | Shifted NLL | Shifted IBS | Paired NLL penalty | Paired IBS penalty |
|---:|---|---:|---:|---:|---:|
| 0.6 | Factorized exponential | 1.621 +/- 0.089 | 0.117 +/- 0.013 | 0.002 +/- 0.014 | -0.003 +/- 0.002 |
| 0.6 | Factorized Weibull | 1.487 +/- 0.102 | 0.118 +/- 0.015 | 0.003 +/- 0.004 | 0.000 +/- 0.003 |
| 0.6 | Factorized piecewise | 1.568 +/- 0.092 | 0.119 +/- 0.017 | 0.022 +/- 0.029 | 0.001 +/- 0.005 |
| 1.6 | Factorized exponential | 1.235 +/- 0.048 | 0.034 +/- 0.004 | 0.014 +/- 0.009 | 0.002 +/- 0.001 |
| 1.6 | Factorized Weibull | 1.127 +/- 0.053 | 0.031 +/- 0.004 | 0.006 +/- 0.006 | 0.001 +/- 0.001 |
| 1.6 | Factorized piecewise | 1.179 +/- 0.044 | 0.030 +/- 0.004 | -0.005 +/- 0.011 | 0.000 +/- 0.001 |

A penalty is the metric under duration shift minus the same model's metric under
the paired in-distribution condition. Small negative values are sampling and
optimization variation, not evidence that less follow-up is beneficial.

## Supported interpretation

All three models remain stable under this moderate, non-informative
duration-support shift. Weibull has the best shifted NLL for both decreasing and
increasing hazards. Piecewise improves substantially over a fixed exponential
shape but does not surpass the correctly specified Weibull likelihood. At shape
1.6, piecewise gives the lowest shifted IBS by a small margin.

The narrow supported claim is that the typed survival boundary can extrapolate
from 6 h training follow-up to an 18 h evaluated horizon when latent dynamics
are stationary and validation extends to 12 h. The result does not establish
robustness to changing transition mechanisms, informative censoring, missed
detections, source conflict, or real observation policies. Those shifts remain
higher-priority external-validity tests. The follow-up
[latent-mechanism shift benchmark](latent-mechanism-shift-benchmark.md) now
tests global-rate, hazard-shape, and typed-factor changes with a Cox baseline
and same-test oracle regret; it exposes catastrophic failure under factor
reversal.

