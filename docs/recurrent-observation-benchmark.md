# Recurrent binary-state observation benchmark

## Question

The original observation likelihood assumes one irreversible transition from
state 0 to state 1. This benchmark asks whether training-only logged observation
sequences can identify a reversible state process, and whether that extension
improves independent current-state prediction when returns really occur.

## Model and information boundary

The latent binary state follows a continuous-time Markov chain with forward rate
`lambda_01` and return rate `lambda_10`. For interval `t`, its exact transition
matrix is computed from both rates. Observation opportunities retain the typed
three-outcome interface: missing, negative, or positive. State-specific
inspection probabilities are `q0` and `q1`; sensitivity is `s`; and the
false-positive rate is `f`.

Training episodes contain only opportunity spacing and the three observation
outcomes. The generator discards latent state paths. A scaled forward-backward
EM fit estimates both transition rates and all four nuisance parameters from
training episodes only. Four deterministic initializations are tried and the
fit with lowest training observation NLL is retained. Test state is generated
independently and never enters estimation or model selection.

At matching time, `ReversibleBinaryPersistenceModel` is property-specific and
accepts only Boolean values for its trained property. Freshness is the
probability that the future state equals the last observed Boolean value; this
allows an unobserved change and return and therefore does not mean that the
state never changed. Unsupported types fail closed, and ordinary event-retention
still composes multiplicatively.

## Frozen protocol

Run:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_recurrent_observation.py
```

- paired seeds: 101, 211, 307, 401, 503;
- forward rate: 0.30/h;
- return rates: 0.00, 0.15, 0.30, and 0.45/h;
- 600 training episodes per seed and condition;
- 8 h follow-up on a regular 0.5 h opportunity grid;
- `q0=0.70`, `q1=0.75`, `s=0.90`, and `f=0.04`;
- 2,000 independent exact-state test rows at horizons 1, 2, 4, 8, or 12 h;
- training random streams and test horizon/outcome uniforms are paired across
  return-rate conditions within a seed;
- no validation or test tuning.

The learned reversible CTMC is compared with the existing irreversible EM fit.
The data-generating rates form an oracle-style reference, not a learned
comparator. Primary effects are irreversible minus reversible exact-test NLL at
the three nonzero return rates. A paired max-standardized-deviation bootstrap
uses 20,000 shared seed resamples to construct simultaneous 95% intervals. The
zero-return condition is a compatibility and added-complexity control and is
excluded from the primary family.

## Verified result

| Return rate (/h) | Irreversible NLL | Reversible NLL | Oracle NLL | NLL advantage, simultaneous 95% CI | W/T/L |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.46056 | 0.46061 | 0.46045 | control only | - |
| 0.15 | 0.69968 | 0.63930 | 0.63878 | 0.06038 [0.05016, 0.07060] | 5/0/0 |
| 0.30 | 0.73184 | 0.65085 | 0.64982 | 0.08099 [0.07314, 0.08884] | 5/0/0 |
| 0.45 | 0.70788 | 0.62526 | 0.62466 | 0.08263 [0.07560, 0.08965] | 5/0/0 |

Mean learned forward rates are 0.296, 0.296, 0.292, and 0.293/h across the four
conditions. Mean learned return rates are 0.0003, 0.1476, 0.3128, and 0.4871/h.
The near-zero control changes NLL by only 0.00005 against the reversible model,
while every nonzero condition favors it in all five paired seeds.

## Claim and limitations

This is evidence that a reversible binary CTMC can be identified from logged
observation outcomes and can repair irreversible-model current-state error under
the matched synthetic mechanism. It is not real-world grounding evidence and
does not establish robustness to irregular opportunities, non-Markov dynamics,
multi-valued states, unknown initial state, correlated or source-specific sensor
errors, conflicting sources, or model misspecification outside this family.

The auditable artifact is
`artifacts/recurrent_observation_results.json`. The core implementation is
`src/openprop/recurrent_observation.py`; focused contracts are in
`tests/test_recurrent_observation.py`.
