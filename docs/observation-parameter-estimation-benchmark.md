# Training-only observation-parameter estimation

Date: 2026-08-26

## Research question

The informative-observation benchmark gives the joint likelihood logged
inspection propensities and detector sensitivity. That establishes a mechanism
upper bound but leaves an important deployment objection: these nuisance
parameters may not be known. This experiment asks whether the state-transition
hazard, pre- and post-transition inspection probabilities, and detection
sensitivity can be identified from training observation sequences alone.

## Forward-backward EM

The model retains the same two hidden states and three logged outcomes as the
factorial benchmark. A scaled forward-backward pass computes posterior state
occupancies and expected unchanged-to-changed transitions. The M-step updates:

- the transition probability from expected `unchanged -> changed` events over
  expected opportunities in the unchanged state;
- each inspection probability from expected non-missing observations over
  expected occupancy of its state;
- sensitivity from expected positives over expected inspected occupancy of the
  changed state.

The continuous-time hazard is recovered as `-log(1 - p) / delta`. Positive
observations anchor the changed state because specificity is fixed at one.
Training fails explicitly when no positive anchor exists. Every EM run stores
its observation negative-log-likelihood history, and tests require this loss to
be non-increasing and to match an independent forward-likelihood calculation.
No validation or test outcome enters parameter estimation.

## Five-seed factorial recovery

The protocol exactly reuses the four conditions, paired latent training
trajectories, paired exact test partitions, five seeds, 1,200 training episodes,
and 1,000 test episodes from the informative-observation benchmark.

Run:

    python scripts/evaluate_observation_parameter_estimation.py

The command writes
`artifacts/observation_parameter_estimation_results.json` with all 20 runs,
parameter errors, convergence diagnostics, exact-test metrics, and the gap to
the logged-parameter upper bound.

### Parameter recovery

Values are population mean and standard deviation across five seeds.

| Condition | Estimated hazard | Estimated `q0` | Estimated `q1` | Estimated sensitivity |
|---|---:|---:|---:|---:|
| Non-informative, perfect | 0.244 +/- 0.007 | 0.349 +/- 0.002 | 0.351 +/- 0.004 | 1.000 +/- 0.000 |
| Informative, perfect | 0.244 +/- 0.005 | 0.148 +/- 0.003 | 0.752 +/- 0.005 | 1.000 +/- 0.000 |
| Non-informative, missed | 0.244 +/- 0.005 | 0.347 +/- 0.002 | 0.351 +/- 0.003 | 0.650 +/- 0.009 |
| Informative, missed | 0.245 +/- 0.005 | 0.148 +/- 0.003 | 0.751 +/- 0.004 | 0.649 +/- 0.006 |

All 20 runs converge. Across conditions, mean absolute hazard error is
0.006-0.007, inspection-probability errors are at most 0.005, and sensitivity
error is below 0.009.

### Independent exact-test performance

| Condition | Estimator | Hazard MAE | Exact-test NLL | IBS |
|---|---|---:|---:|---:|
| Non-informative, perfect | Interval only | 0.007 +/- 0.005 | 2.281 +/- 0.019 | 0.149 +/- 0.002 |
|  | Logged parameters | 0.007 +/- 0.005 | 2.281 +/- 0.019 | 0.149 +/- 0.002 |
|  | Training-estimated | 0.007 +/- 0.005 | 2.281 +/- 0.019 | 0.149 +/- 0.002 |
| Informative, perfect | Interval only | 0.068 +/- 0.010 | 2.312 +/- 0.020 | 0.153 +/- 0.003 |
|  | Logged parameters | 0.006 +/- 0.005 | 2.281 +/- 0.019 | 0.149 +/- 0.002 |
|  | Training-estimated | 0.006 +/- 0.005 | 2.281 +/- 0.019 | 0.149 +/- 0.002 |
| Non-informative, missed | Interval only | 0.053 +/- 0.003 | 2.305 +/- 0.017 | 0.153 +/- 0.001 |
|  | Logged parameters | 0.007 +/- 0.004 | 2.281 +/- 0.019 | 0.149 +/- 0.002 |
|  | Training-estimated | 0.007 +/- 0.004 | 2.281 +/- 0.018 | 0.149 +/- 0.002 |
| Informative, missed | Interval only | 0.010 +/- 0.007 | 2.282 +/- 0.019 | 0.149 +/- 0.002 |
|  | Logged parameters | 0.006 +/- 0.005 | 2.281 +/- 0.019 | 0.149 +/- 0.002 |
|  | Training-estimated | 0.006 +/- 0.004 | 2.281 +/- 0.019 | 0.149 +/- 0.002 |

The estimated model's mean NLL gap to the logged-parameter upper bound is at
most 0.0002 in every factorial condition. This closes the oracle-parameter gap
for the current synthetic generator; it does not prove generic identifiability.

## Sample-identifiability stress test

A second paired experiment weakens post-transition inspection to 0.4, crosses
detector sensitivities 0.3 and 0.65, and uses nested training sizes 50, 100, 300,
and 1,200. Each seed retains the same 1,000 exact test episodes.

Run:

    python scripts/evaluate_observation_identifiability.py

The command writes `artifacts/observation_identifiability_results.json` with 40
runs. The table reports training-estimated hazard MAE and exact-test NLL minus
the logged-parameter upper bound.

| Sensitivity | Training episodes | Positive episodes | Hazard MAE | NLL oracle gap |
|---:|---:|---:|---:|---:|
| 0.30 | 50 | 39.6 | 0.071 +/- 0.070 | 0.044 +/- 0.080 |
| 0.30 | 100 | 78.6 | 0.022 +/- 0.011 | 0.000 +/- 0.007 |
| 0.30 | 300 | 238.8 | 0.013 +/- 0.008 | 0.000 +/- 0.002 |
| 0.30 | 1,200 | 964.8 | 0.013 +/- 0.006 | 0.000 +/- 0.001 |
| 0.65 | 50 | 47.6 | 0.042 +/- 0.041 | 0.003 +/- 0.010 |
| 0.65 | 100 | 93.4 | 0.015 +/- 0.013 | 0.000 +/- 0.001 |
| 0.65 | 300 | 276.8 | 0.012 +/- 0.006 | -0.000 +/- 0.000 |
| 0.65 | 1,200 | 1,100.4 | 0.007 +/- 0.004 | 0.000 +/- 0.000 |

Every run reaches the numerical convergence criterion, but 50-episode recovery
is statistically unreliable. At sensitivity 0.3, one seed has an NLL oracle
gap of 0.202 and hazard estimates range from 0.177 to 0.454. This separates
optimizer convergence from empirical identifiability and is a required failure
case, not a result to hide. At 100 or more episodes, mean exact-test NLL is near
the logged upper bound, although individual parameter errors remain larger than
in the 1,200-episode factorial experiment.

## Supported interpretation and limits

Under the original irreversible two-state generator, regular opportunity grid,
perfect-specificity setting, and sufficient positive anchors, forward-backward EM can
recover both persistence and observation parameters using training logs only.
It removes the previous need to supply generator-known nuisance parameters in
the main factorial setting.

The stress test also establishes a limit: numerical EM convergence is not an
identifiability guarantee at small sample sizes or weak detection. The follow-up
[false-positive benchmark](false-positive-observation-benchmark.md) extends the
same likelihood to homogeneous imperfect specificity and finds a clear benefit
only at the largest tested false-positive rate. The combined evidence still
includes a separate [recurrent binary-state benchmark](recurrent-observation-benchmark.md)
that estimates both transition directions from logged outcomes. It does not
cover unknown or irregular opportunity coverage, multi-valued or non-Markov
recurrence, source-specific or conflicting evidence, or
semi-real logs. These remain necessary before claiming general observation-
process robustness.
