# Leakage-safe target adaptation under mechanism shift

Date: 2026-08-26

## Research question

The latent-mechanism shift benchmark showed a catastrophic and interpretable
failure: when typed context effects reverse, every source-trained survival
model reverses the correct risk ordering. This experiment asks whether a small,
labeled target calibration set can detect and repair that failure without using
target-test labels or generator truth.

This is synthetic mechanism validation. It does not establish that six labeled
real trajectories will be sufficient in a new environment.

## Frozen protocol

The experiment uses ten seeds: 31, 41, 53, 67, 79, 97, 109, 127, 149, and 173.
For each seed and target context, 80 paired target records are generated for all
five mechanism conditions. A SHA-256 rank of only `group_id` and the split seed
selects a maximum 32-record calibration pool per context. Event times,
censoring, and condition labels cannot affect membership.

Calibration sizes of 2, 4, 8, 16, and 32 records per context are nested subsets
of that frozen pool. The remaining 48 records from each of three contexts form
one fixed 144-record test set for every calibration size. Calibration and test
group IDs are disjoint. All conditions retain paired group IDs, contexts,
latent draws, and censoring draws.

The source factorized exponential model is fitted on source training data and
scaled on source validation data. Four target strategies are evaluated:

- `scale_only`: fit one target log-hazard offset while preserving source rank;
- `affine_log_risk`: fit an offset and slope in target log-risk space;
- `sign_gated`: use the affine model only when its calibration-only slope is
  negative; otherwise return the source model unchanged;
- `target_per_context`: a strong transductive target-only MLE that observes the
  same context identities in calibration and test.

The sign threshold, optimizer, regularization, sample sizes, and evaluation
horizons are fixed across conditions. No method or hyperparameter is selected
on target test. Generator hazards remain evaluation-only.

Run:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_target_adaptation.py
```

The command writes `artifacts/target_adaptation_results.json`.

## Reversal detection

The affine slope was negative in all 50 typed-factor-reversal decisions: ten
seeds at each of five calibration sizes. It was nonnegative in all 200
seed-condition-size decisions across in-distribution, global acceleration,
decreasing-shape, and increasing-shape conditions. Consequently, the
sign-gated method exactly equals the source model on every non-reversal test
and activates the affine repair on every reversal test.

These 250 decisions are paired and the calibration subsets are nested; they
must not be described as 250 independent trials.

## Typed-factor-reversal results

Values are means across ten seeds on the same 144 held-out test records per
seed. `k` is calibration records per context; the total target-label count is
three times `k`.

| k | Labels | Mean slope | Source NLL | Gated NLL | Target-only NLL | Source C-index | Gated C-index | Target-only C-index | Source IBS | Gated IBS | Target-only IBS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 6 | -1.278 | 3.506 | 1.191 | 1.220 | 0.183 | 0.817 | 0.802 | 0.627 | 0.129 | 0.154 |
| 4 | 12 | -1.177 | 3.506 | 1.072 | 1.113 | 0.183 | 0.817 | 0.787 | 0.627 | 0.122 | 0.133 |
| 8 | 24 | -1.136 | 3.506 | 0.973 | 0.988 | 0.183 | 0.817 | 0.800 | 0.627 | 0.114 | 0.119 |
| 16 | 48 | -1.051 | 3.506 | 0.938 | 0.959 | 0.183 | 0.817 | 0.806 | 0.627 | 0.109 | 0.115 |
| 32 | 96 | -1.031 | 3.506 | 0.937 | 0.938 | 0.183 | 0.817 | 0.814 | 0.627 | 0.108 | 0.109 |

At `k=2`, sign gating improves NLL by 2.314, C-index by 0.634, and IBS by
0.498 on average relative to the paired source model. The two-parameter adapter
also exceeds the target-only per-context baseline at every reported sample size
on mean NLL, C-index, and IBS. This is evidence that reusing source risk can be
sample-efficient than estimating each target context independently when the
target transformation is globally affine in log-risk.

At `k=2`, paired improvements have seed-cluster bootstrap 95% confidence
intervals of [2.157, 2.464] for NLL reduction, [0.613, 0.655] for C-index
increase, and [0.471, 0.522] for IBS reduction. Each metric improves in all ten
seeds; the two-sided exact sign-test p-value is 0.001953. Bootstrap resampling
uses seeds as the independent units, 20,000 deterministic resamples, and never
resamples nested calibration sizes as if they were independent.


## What the result does and does not establish

Supported under this generator:

- calibration-only slope sign can distinguish the exact typed-factor reversal
  from four order-preserving shifts;
- a source-preserving gate prevents adaptation from changing non-reversal
  predictions;
- when the target log-hazard is an affine transform of source log-risk, six
  labels can outperform a target-only per-context estimator;
- all comparisons use fixed held-out tests and standard survival metrics.

Not supported:

- arbitrary nonlinear, local, partial, or noisy factor changes;
- unseen target feature values or context identities;
- informative censoring, recurrent transitions, or observation error;
- real or semi-real target sample efficiency;
- automatic correction of positive-slope rate or hazard-shape shift. The safe
  gate deliberately preserves the source model in those cases.

Partial factor reversal, mixed stable and changed contexts, 20% calibration
label flips, and all 18 typed contexts are now evaluated in the
[local target-adaptation stress benchmark](target-adaptation-stress-benchmark.md).
That experiment supports correct-axis local repair at adequate calibration
size, but exposes small-sample false activations and stable-subgroup harm under
an unrepresentable XOR interaction. Semi-real longitudinal histories remain
unevaluated.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Small labeled calibration can detect the benchmark's exact typed-factor reversal. | Negative slopes in 50/50 reversal decisions and 0/200 non-reversal decisions, with outcome-independent splits. | Supported only for the paired synthetic mechanisms. |
| Sign gating repairs held-out reversal performance. | At six labels, NLL/C-index/IBS change from 3.506/0.183/0.627 to 1.191/0.817/0.129, with 10/10 paired wins. | Supported only for the exact affine reversal. |
| Sign gating is harmless under any real distribution shift. | Only four synthetic order-preserving alternatives were tested. | Unsupported and must not be claimed. |
| OpenProp now has semi-real adaptation evidence. | No official longitudinal target dataset was used. | Unsupported and must not be claimed. |
