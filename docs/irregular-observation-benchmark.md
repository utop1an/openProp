# Irregular observation-timing benchmark

## Question

The recurrent observation model originally required a common regular opportunity
grid. This benchmark asks whether replacing each actual elapsed interval with the
global mean creates a persistence error under bursty observation timing, and
whether a training-only continuous-time estimator removes that error.

## Method and information boundary

Each training episode stores a sequence of positive elapsed intervals and a
matching sequence of `missing`, `negative`, or `positive` outcomes. Timestamps
remain observation-history metadata; they are not properties. Latent binary state
paths are discarded by the generator and are unavailable to both estimators.

The exact-interval model computes a separate two-state CTMC transition matrix for
every elapsed interval. Its E-step uses scaled forward-backward inference. The
M-step updates inspection, sensitivity, and false-positive parameters from
posterior counts and maximizes the expected transition likelihood over forward
and return rates with deterministic bounded coordinate optimization. Every
accepted update preserves the observed-data likelihood. Four fixed
initializations are tried, and the lowest training observation NLL is retained.

The direct ablation collapses the same outcomes to one common mean interval and
runs the established regular-grid EM. It receives the true global mean interval,
so the intervention isolates loss of interval structure rather than an incorrect
time unit. The logged data-generating process is an oracle-style reference, not a
learned baseline.

## Frozen paired protocol

Run:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_irregular_observation.py
```

- paired seeds: 101, 211, 307, 401, 503;
- 600 training episodes per seed and condition;
- 16 observation opportunities and exactly 12 h total follow-up per episode;
- mean interval: 0.75 h;
- gap contrasts: 0.00, 0.50, 0.75, and 0.90;
- two long gaps and fourteen short gaps per episode, in a seed-paired random order;
- short/long gaps: 0.75/0.75, 0.375/3.375, 0.1875/4.6875, and 0.075/5.475 h;
- true forward/return rates: 0.30/0.45 per hour;
- state-specific inspection: 0.70/0.75, sensitivity 0.90, false-positive rate 0.04;
- 20,000 independent exact-state test rows per seed at 1, 2, 4, 8, or 12 h;
- schedule positions, training random streams, and exact-state test rows are
  paired across gap conditions;
- no validation or test tuning.
- deterministic forward/return rate search bounds: `1e-6` to `5.0` per hour.

The primary family is mean-grid minus exact-interval current-state test NLL at
the three nonzero contrasts. A paired max-standardized-deviation bootstrap uses
20,000 shared seed resamples to form simultaneous 95% intervals. Zero contrast
is a compatibility control excluded from the primary family.

## Verified result

| Gap contrast | Mean-grid rates (/h) | Exact-interval rates (/h) | Mean-grid NLL | Exact-interval NLL | NLL advantage, simultaneous 95% CI | W/T/L |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.300 / 0.455 | 0.299 / 0.455 | 0.62991 | 0.62990 | control only | - |
| 0.50 | 0.205 / 0.317 | 0.286 / 0.442 | 0.63444 | 0.63053 | 0.00391 [0.00151, 0.00630] | 5/0/0 |
| 0.75 | 0.148 / 0.222 | 0.291 / 0.448 | 0.64432 | 0.62986 | 0.01446 [0.01217, 0.01675] | 5/0/0 |
| 0.90 | 0.106 / 0.142 | 0.310 / 0.464 | 0.66114 | 0.62975 | 0.03140 [0.02907, 0.03373] | 5/0/0 |

The oracle exact-state NLL is 0.62961 in every condition. Exact-interval mean NLL
stays within 0.00093 of this reference, whereas the mean-grid error grows
monotonically with burstiness. On the actual irregular training schedules, the
mean-grid observation NLL is worse by 0.0305, 0.0815, and 0.0933 per episode at
the three nonzero contrasts.

## Supported claim and limitations

These results show that actual elapsed intervals matter for current-state
prediction under a matched recurrent binary CTMC: replacing a bursty schedule by
its mean interval creates systematic rate and NLL error, while training-only
exact-interval estimation removes most of it. This is synthetic mechanism
validation, not evidence of natural observation prevalence or real-world
grounding effectiveness.

The protocol fixes total follow-up, uses only two gap lengths per condition, a
known initial state, Markov binary dynamics, and homogeneous observation noise.
It does not cover informative opportunity timing, timestamp error, continuous or
multi-valued state, source-specific reliability, correlated/conflicting sources,
or semi-real histories. The largest condition is deliberately severe; the
positive simultaneous interval at contrast 0.50 is therefore important to the
bounded claim.

The frozen artifact is `artifacts/irregular_observation_results.json`. The
implementation is `src/openprop/irregular_recurrent_observation.py`; focused
contracts are in `tests/test_irregular_recurrent_observation.py`.
