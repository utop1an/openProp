# Source-specific reliability under conflicting observations

## Question

OpenProp stores source identity as observation provenance rather than as a typed
property value. This benchmark asks whether tying all sources to one observation
model corrupts current-state grounding when sources differ in availability,
sensitivity, and false-positive rate.

## Model and information boundary

One recurrent binary CTMC supplies the shared latent state and forward/return
rates. Conditional on that state, each source emits `missing`, `negative`, or
`positive` with its own state-specific inspection probabilities, sensitivity,
and false-positive rate. Scaled forward-backward EM estimates the shared dynamics
and source-specific emissions from training histories only. Four deterministic
initializations are tried; likelihood-decreasing or emission-order-violating
fits fail closed.

The direct ablation sees the same source-labelled opportunities but ties all
source emission parameters. Thus it discards source identity only in the
statistical parameterization; it does not receive less data. Evaluation rows
store `current_truth` beside an observation history, and the filter receives
only the history. Training episodes contain no latent path or current truth.

## Frozen paired protocol

Run:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_source_reliability.py
```

- paired seeds: 101, 211, 307, 401, 503;
- 400 training episodes per seed and condition;
- 24 half-hour opportunities and 12 h follow-up per episode;
- true forward/return rates: 0.30/0.45 per hour;
- two sources and 20,000 independent filtered-current-state test rows per seed;
- conflict severities: 0.00, 0.33, 0.67, and 1.00;
- source-average inspection probabilities remain 0.65/0.65, sensitivity 0.80,
  and false-positive rate 0.10 at every severity;
- severity moves the two sources symmetrically in opposite directions, reaching
  inspection 0.90/0.40 versus 0.40/0.90, sensitivity 0.95 versus 0.65, and
  false-positive rate 0.02 versus 0.18 at severity 1.00;
- latent state random streams and evaluation state paths are paired within seed;
- no validation or test tuning.

The primary family is pooled minus source-aware filtered-current-state NLL at
the three nonzero severities. A paired max-standardized-deviation bootstrap uses
20,000 shared seed resamples for simultaneous 95% intervals. Zero severity is a
nested compatibility control excluded from the family.

## Verified result

| Severity | Pooled NLL | Source-aware NLL | Oracle NLL | NLL advantage, simultaneous 95% CI | W/T/L |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.29591 | 0.29595 | 0.29507 | control only | - |
| 0.33 | 0.30078 | 0.27906 | 0.27818 | 0.02173 [0.02059, 0.02287] | 5/0/0 |
| 0.67 | 0.29621 | 0.21386 | 0.21295 | 0.08235 [0.07978, 0.08492] | 5/0/0 |
| 1.00 | 0.29745 | 0.12738 | 0.12651 | 0.17006 [0.16565, 0.17447] | 5/0/0 |

The corresponding source-aware Brier scores are 0.09012, 0.08460, 0.06346,
and 0.03636, compared with pooled scores near 0.090 in every condition. At the
largest severity, source-aware mean rates are 0.300/0.462 per hour, whereas the
pooled model shifts to 0.262/0.427. Every fit converged.

## Supported claim and limitations

Under this matched synthetic recurrent mechanism, preserving source identity in
the observation likelihood prevents conflicting reliability profiles from
flattening current-state evidence. The zero-severity near-tie and fixed
source-average parameters isolate heterogeneity rather than observation budget.
This is synthetic mechanism validation, not evidence about natural source
reliability or real-world grounding effectiveness.

The model assumes two conditionally independent sources, stable source identity,
stationary per-source parameters, a known initial state, regular opportunities,
and a binary Markov state. It does not cover correlated failures, adversarial
sources, source appearance/disappearance, timestamp error, learned source
embeddings, continuous or multi-valued state, or semi-real histories.

The frozen artifact is `artifacts/source_reliability_results.json`. The model is
in `src/openprop/source_reliability_observation.py`; evaluation-only truth is in
`src/openprop/source_reliability_evaluation.py`; focused contracts are in
`tests/test_source_reliability_observation.py` and
`tests/test_source_reliability_evaluation.py`.
