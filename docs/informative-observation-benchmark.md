# Informative observation and missed-detection benchmark

Date: 2026-08-26

## Research question

Interval censoring is sufficient when inspection timing is independent of the
latent state and inspection outcomes are exact. Real logs can violate both
conditions: a changed state can make an inspection more likely, and an
inspection can miss the change. This benchmark asks whether persistence can be
recovered without treating either missing observations or negative detections
as hidden ground truth.

## Joint observation-state model

Each episode starts in the unchanged state and has a fixed observation-
opportunity grid with spacing 0.5 h. The latent state changes irreversibly with
exponential hazard `lambda`. At an opportunity, inspection is recorded with
probability `q0` before the transition and `q1` after it. A recorded changed
state is detected with sensitivity `s`.

For one grid step, the transition probability is
`1 - exp(-lambda * 0.5)`. The emission probabilities are:

| Logged result | Unchanged state | Changed state |
|---|---:|---:|
| Missing | `1 - q0` | `1 - q1` |
| Negative | `q0` | `q1 * (1 - s)` |
| Positive | `0` | `q1 * s` |

A two-state forward recursion marginalizes the hidden state sequence and fits
`lambda` by maximum likelihood. Training episodes contain only opportunity
spacing and the sequence of missing, negative, and positive results. Latent
transition times are discarded from training and generated independently only
for exact-time evaluation. The observation probabilities and sensitivity are
treated as logged policy and sensor-calibration inputs; they are never tuned on
test outcomes.

## Paired factorial protocol

The experiment crosses state-dependent inspection and missed detection:

| Condition | `q0` | `q1` | Sensitivity |
|---|---:|---:|---:|
| Non-informative, perfect | 0.35 | 0.35 | 1.00 |
| Informative, perfect | 0.15 | 0.75 | 1.00 |
| Non-informative, missed | 0.35 | 0.35 | 0.65 |
| Informative, missed | 0.15 | 0.75 | 0.65 |

All conditions use a true hazard of 0.25/h, 12 h follow-up, 1,200 training
episodes and the same 1,000 exact-time test episodes within each seed. Seeds are
101, 211, 307, 401, and 503. Three estimators are compared:

- `naive_detection`: first positive detection is treated as the exact event;
- `interval_only`: last negative to first positive is treated as an event
  interval, without modeling why observations are missing or false negative;
- `observation_aware`: the joint hidden-state observation likelihood above.

Run:

    python scripts/evaluate_informative_observation.py

The command writes `artifacts/informative_observation_results.json`, including
all 20 condition-seed runs and paired gains.

## Verified five-seed results

Values are population mean and standard deviation across seeds.

| Condition | Estimator | Hazard | Hazard MAE | Exact-test NLL | IBS |
|---|---|---:|---:|---:|---:|
| Non-informative, perfect | Naive | 0.191 +/- 0.003 | 0.059 +/- 0.003 | 2.311 +/- 0.017 | 0.155 +/- 0.001 |
|  | Interval only | 0.244 +/- 0.006 | 0.007 +/- 0.005 | 2.281 +/- 0.019 | 0.149 +/- 0.002 |
|  | Observation aware | 0.244 +/- 0.006 | 0.007 +/- 0.005 | 2.281 +/- 0.019 | 0.149 +/- 0.002 |
| Informative, perfect | Naive | 0.228 +/- 0.005 | 0.022 +/- 0.005 | 2.284 +/- 0.018 | 0.149 +/- 0.001 |
|  | Interval only | 0.318 +/- 0.010 | 0.068 +/- 0.010 | 2.312 +/- 0.020 | 0.153 +/- 0.003 |
|  | Observation aware | 0.244 +/- 0.005 | 0.006 +/- 0.005 | 2.281 +/- 0.019 | 0.149 +/- 0.002 |
| Non-informative, missed | Naive | 0.165 +/- 0.003 | 0.085 +/- 0.003 | 2.351 +/- 0.016 | 0.164 +/- 0.001 |
|  | Interval only | 0.197 +/- 0.003 | 0.053 +/- 0.003 | 2.305 +/- 0.017 | 0.153 +/- 0.001 |
|  | Observation aware | 0.245 +/- 0.005 | 0.007 +/- 0.004 | 2.281 +/- 0.019 | 0.149 +/- 0.002 |
| Informative, missed | Naive | 0.211 +/- 0.004 | 0.039 +/- 0.004 | 2.293 +/- 0.018 | 0.151 +/- 0.001 |
|  | Interval only | 0.260 +/- 0.007 | 0.010 +/- 0.007 | 2.282 +/- 0.019 | 0.149 +/- 0.002 |
|  | Observation aware | 0.245 +/- 0.006 | 0.006 +/- 0.005 | 2.281 +/- 0.019 | 0.149 +/- 0.002 |

Under the clean non-informative condition, interval-only and observation-aware
estimates are identical, which is an important reduction check. With only
state-dependent inspection, interval-only hazard MAE rises from 0.007 to 0.068;
the joint model gives 0.006. With only missed detection, interval-only MAE is
0.053 while the joint model gives 0.007. These joint-model improvements over
interval-only occur for every seed in both isolated violation conditions.

The combined condition is a deliberate warning against evaluating only one
mixture. State-dependent inspection pushes the interval estimate upward while
missed detection pushes it downward, so their errors partly cancel. The
interval-only MAE of 0.013 therefore looks deceptively strong even though the
factorial controls show that its observation assumptions are wrong.

## Supported interpretation and limits

This experiment supports a synthetic mechanism claim: explicitly modeling a
logged observation process can remove opposing biases from state-dependent
inspection and missed detection, while reducing exactly to interval censoring
when its assumptions hold.

It does not establish real-world effectiveness. The paired
[parameter-estimation benchmark](observation-parameter-estimation-benchmark.md)
shows that training-only EM closes the logged-parameter oracle gap under this
generator, but also becomes statistically unstable with only 50 episodes. The
[false-positive follow-up](false-positive-observation-benchmark.md) relaxes
perfect specificity for one homogeneous detector and retains its lower-noise
power boundary. The joint evidence still assumes a fixed opportunity grid, one
source, and exponential state dynamics. A separate
[recurrent-state follow-up](recurrent-observation-benchmark.md) relaxes the
single-transition assumption for a matched binary CTMC. Latent-dynamics shift,
irregular or multi-valued recurrence, conflicting sources, and semi-real longitudinal
validation remain necessary before making a broad robustness claim.
