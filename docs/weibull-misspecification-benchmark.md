# Weibull model-misspecification benchmark

Date: 2026-08-26

## Research question

The original compositional generator is exponential, so any exponential
persistence model is correctly specified by construction. This experiment asks
whether the evaluation detects time-varying transition risk rather than merely
rewarding typed context factorization.

The same 12/3/3 held-out-context split is generated with Weibull survival

    S(t | x) = exp(-(lambda(x) t)^k),

where lambda(x) retains the typed multiplicative context factors and the shared
shape k controls time dependence. Shape 0.6 gives decreasing risk, shape 1.0
recovers the exponential case, and shape 1.6 gives increasing risk.

## Models and protocol

- factorized exponential: the strong log-linear typed baseline with k fixed to 1;
- factorized Weibull: the same typed effects plus one learned shared shape;
- both models use exact/right/interval-censored likelihoods;
- one global rate multiplier is calibrated on validation only;
- complete validation and test context tuples remain held out;
- default evaluation uses seeds 31, 41, 53, 67 and 79 with 80 samples per
  context.

Run:

    python scripts/evaluate_weibull_misspecification.py

The command writes
`artifacts/weibull_misspecification_results.json`.

## Verified five-seed results

| True shape | Learned shape | Exponential NLL | Weibull NLL | Paired NLL improvement | Win rate |
|---:|---:|---:|---:|---:|---:|
| 0.6 | 0.610 +/- 0.021 | 1.403 +/- 0.067 | 1.280 +/- 0.076 | 0.123 +/- 0.018 | 5/5 |
| 1.0 | 1.016 +/- 0.042 | 1.239 +/- 0.069 | 1.240 +/- 0.071 | -0.001 +/- 0.002 | 3/5 |
| 1.6 | 1.631 +/- 0.066 | 1.144 +/- 0.050 | 1.045 +/- 0.044 | 0.099 +/- 0.012 | 5/5 |

Integrated Brier score also improves under misspecification: 0.135 to 0.134 for
shape 0.6 and 0.042 to 0.039 for shape 1.6. It remains effectively unchanged
when shape is 1.0. C-index is identical between the two models because a shared
Weibull shape does not change the ordering of context-specific rates.

## Supported interpretation

The benchmark supports a model-soundness claim: OpenProp's persistence boundary
can represent and evaluate non-exponential state dynamics, and a learned
Weibull shape recovers decreasing and increasing hazards without creating a
material advantage when the exponential model is correct.

This result does not establish that real object-state transitions are Weibull.
It is synthetic model-misspecification validation. Semi-real histories, temporal
latent-dynamics shift, informative observation policies, and Cox or
nonparametric baselines remain required.

