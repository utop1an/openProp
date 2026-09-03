# False-positive observation-process stress test

Date: 2026-08-26

## Research question

The earlier hidden observation model allows missed detections but assumes every
positive result is specific. This is unsafe when a sensor, detector, or human
label can report a state change before the latent change occurred. This stress
test asks two separate questions:

1. how much bias is caused by retaining the perfect-specificity assumption; and
2. whether false-positive rate can be estimated from training observation
   sequences without transition truth, validation labels, or test outcomes.

## Model extension

The irreversible hidden state remains `unchanged -> changed`. At an observation
opportunity, the state-dependent inspection probabilities are `q0` and `q1`,
changed-state sensitivity is `s`, and unchanged-state false-positive rate is
`f`. The emission table becomes:

| Logged result | Unchanged state | Changed state |
|---|---:|---:|
| Missing | `1 - q0` | `1 - q1` |
| Negative | `q0 * (1 - f)` | `q1 * (1 - s)` |
| Positive | `q0 * f` | `q1 * s` |

The scaled forward-backward E-step assigns posterior positive counts to both
hidden states. The M-step estimates `f` from expected positives while unchanged
divided by expected inspections while unchanged. Estimation is opt-in: the
existing API fixes `f = 0` unless `estimate_false_positive_rate=True`, so the
extra nuisance parameter is never introduced silently.

## Frozen paired protocol

- seeds: 101, 211, 307, 401, 503;
- 1,200 training episodes and 1,000 independent exact-time test records per
  seed and condition;
- true hazard: 0.25/h;
- inspection probabilities: `q0 = 0.15`, `q1 = 0.75`;
- detection sensitivity: 0.65;
- false-positive rates: 0.00, 0.02, 0.05, and 0.10;
- identical latent training-transition draws and exact test rows across the
  four false-positive conditions within each seed;
- no latent transition time is stored in a training episode;
- no validation or test label enters estimation.

Four estimators are compared: interval-only fitting, EM with perfect
specificity fixed, EM that estimates false-positive rate, and a logged-parameter
observation-process upper bound. The primary family is fixed-specificity minus
estimated-specificity exact-test NLL at the three nonzero false-positive rates.
Its intervals use one shared paired-seed bootstrap and a family-wise
max-standardized-deviation critical value over 20,000 resamples.

Run:

    python scripts/evaluate_false_positive_observation.py

The command writes `artifacts/false_positive_observation_results.json` with all
20 runs, parameter estimates, exact-test metrics, and simultaneous intervals.

## Verified results

| False-positive rate | Interval-only hazard MAE | Fixed-specificity EM hazard MAE | Estimated-specificity EM hazard MAE | Estimated `f` | Fixed minus estimated NLL [simultaneous 95% CI] | Wins / ties / losses |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.0103 | 0.0059 | 0.0059 | 0.0000 | -- | -- |
| 0.02 | 0.0177 | **0.0046** | 0.0064 | 0.0211 | -0.00009 [-0.00056, 0.00038] | 3 / 0 / 2 |
| 0.05 | 0.0281 | 0.0110 | **0.0062** | 0.0467 | 0.00098 [-0.00014, 0.00210] | 4 / 0 / 1 |
| 0.10 | 0.0477 | 0.0284 | **0.0061** | 0.0947 | **0.00615 [0.00343, 0.00887]** | 5 / 0 / 0 |

At false-positive rate 0.10, fixing specificity at one overestimates the hazard
as 0.2784 on average, while estimating false positives gives 0.2451. The
estimated model reduces hazard MAE from 0.0284 to 0.0061 and improves exact-test
NLL in every seed; the family-wise simultaneous lower bound is positive.

The lower-noise rows are equally important. At rate 0.02, the fixed-specificity
model has slightly lower hazard MAE and the paired NLL mean favors it by less
than 0.0001. At rate 0.05, four of five seeds favor estimation but the
simultaneous interval crosses zero. Thus the experiment does not support a
blanket claim that estimating an additional nuisance parameter is always
better. It establishes a detectable high-false-positive regime and a finite-
sample power boundary.

## Supported interpretation and limits

Training-only EM can recover false-positive rate and remove substantial
specificity misspecification under this irreversible, regular-grid synthetic
generator. It reduces to the previous model when no false positives are
present, and its likelihood is non-increasing in every tested EM run.

This is not evidence for arbitrary sensor noise or real observation processes.
The model still assumes homogeneous source reliability, a common opportunity
grid, one irreversible transition, exponential dynamics, and a known initial
state. The five-seed experiment has insufficient evidence for a primary gain at
false-positive rates 0.02 or 0.05. Recurrent transitions, irregular coverage,
source-specific reliability, conflicting sources, and official longitudinal
validation remain open.
