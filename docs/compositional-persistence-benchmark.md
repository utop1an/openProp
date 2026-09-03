# Compositional persistence benchmark

This experiment is the first novelty-oriented extension beyond the original
two-context persistence smoke test. It asks whether a context-conditioned model
can predict state persistence for a **held-out combination** of familiar typed
features, then use that estimate to make the final entity-grounding decision.

## Research question

The training data contains location episodes described by five typed fields:

```text
property, subject type, relation predicate, context object, scene
```

Every individual field value used by validation and test appears in training,
but the complete validation and test tuples do not. The split therefore tests
compositional reuse of known factors rather than exact-context lookup.

The synthetic hazard generator is log-factorial in subject, relation and scene.
It creates censored, irregular-duration histories for 18 contexts:

- 12 training contexts and 960 default training episodes;
- 3 validation contexts and 240 default validation episodes;
- 3 test contexts and 240 default test episodes.

Group identifiers are disjoint across partitions. Hidden current truth remains
separate from entity observations in the grounding cases.

## Baselines

- `global-exponential`: one maximum-likelihood hazard over all training rows;
- `per-context-exponential`: one smoothed MLE per complete training tuple, with
  an explicit global backoff for unseen tuples;
- `factorized-exponential`: an L2-regularized log-linear hazard over the five
  typed categorical factors, trained with the same censoring likelihood and
  calibrated once on validation;
- `neural-compositional`: the existing categorical factor encoder and MLP hazard
  head, calibrated once on the validation partition;
- no decay and a fixed four-hour half-life for final grounding.

The per-context baseline deliberately reduces to the global model on the OOD
test tuples. This makes the difference between memorising a context and
composing previously observed factors explicit.

## Run

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_compositional_persistence.py
python scripts/evaluate_compositional_multiseed.py
```

The commands write:

- `artifacts/compositional_persistence_results.json`;
- `artifacts/compositional_persistence_multiseed_results.json`.

## Verified five-seed result

The default seeds are 31, 41, 53, 67 and 79. Values below are population mean
and standard deviation across those runs.

| Model | Test NLL lower is better | C-index higher is better | Integrated Brier lower is better | Grounding Top-1 higher is better |
|---|---:|---:|---:|---:|
| Global exponential | 1.795 +/- 0.033 | 0.500 +/- 0.000 | 0.243 +/- 0.003 | 0.000 +/- 0.000 |
| Factorized exponential | 1.239 +/- 0.069 | 0.749 +/- 0.014 | 0.079 +/- 0.010 | 1.000 +/- 0.000 |
| Neural compositional | 1.266 +/- 0.061 | 0.723 +/- 0.026 | 0.084 +/- 0.009 | 0.972 +/- 0.056 |

The factorized model achieved grounding Top-1 1.000 for every seed. It also
improved mean NLL, C-index and integrated Brier score over the neural model.
The neural model ranged from 0.861 to 1.000 grounding Top-1. No decay, fixed
decay, global exponential and per-context exponential scored zero on the
deliberately discriminative grounding cases in the verified single-seed
comparison.

## Supported interpretation

This experiment supports one narrow mechanism claim:

> When synthetic state dynamics factor across familiar typed context variables,
> explicitly factorized persistence models can reuse those variables on held-out
> combinations, whereas global decay and exact-context lookup cannot.

The strong baseline changes the method conclusion. This benchmark supports
typed factorization, but it does not support neural parameterization as the
source of the gain: the simpler log-linear model is better on all four reported
metrics. The downstream result still links survival quality to grounding where
a global half-life favours the wrong candidate.

## Component attribution

A frozen ten-seed, eight-condition ablation now separates subject, relation, and
scene effects. Removing scene, relation, or subject from the full model worsens
test NLL by 0.402 [0.364, 0.441], 0.156 [0.137, 0.176], and 0.052
[0.038, 0.066], respectively. These are family-wise simultaneous 95% paired-
bootstrap intervals across the three predeclared NLL comparisons; every row has 10/10 wins.
However, subject and relation removal do not change Grounding Top-1, while scene
removal collapses it. The survival result supports all three factors; the current
downstream construction supports only scene-dependent decision utility. See the
[typed-context component ablation](typed-context-component-ablation.md) for the
frozen protocol, full table, artifact hash, and claim limits.
A follow-up [component-balanced grounding benchmark](component-balanced-grounding.md)
holds non-tested context fixed and places an analytic confidence-age crossover
between factor hazards. On untouched confirmation seeds, full-context exceeds
the matching no-subject, no-relation, and no-scene models by 0.347, 0.327, and
0.500 probe Top-1, with paired intervals excluding zero. This supports all-axis
decision utility only for the controlled generator; it does not replace the
need for semi-real longitudinal grounding.


## Limits

This is still synthetic mechanism validation. It does not establish:

- generalisation to wholly unseen property or entity-type values;
- realistic state-transition probabilities;
- robustness to noisy language-to-query parsing;
- separation of state persistence from observation availability;
- learned event effects;
- robustness to changes in the meaning of typed context factors;
- real-world grounding accuracy.

The grounding cases are intentionally constructed around the factorised hazard
mechanism. They should be complemented by sampled trajectories, temporal
distribution shift and real or semi-real observation histories before use as a
main-paper benchmark.

## Observation-process follow-up

The first observation-process upgrade is now implemented: detected transitions
can retain the interval between the last negative and first positive inspection,
and training uses an interval-censored likelihood. The
[observation-process benchmark](observation-process-benchmark.md) shows that
this removes most false inspection-frequency dependence in a controlled
five-seed experiment. A subsequent joint hidden-state model handles missed
detections and informative inspection policies, and training-only EM estimates
its nuisance parameters when positively anchored. The
[latent-mechanism shift benchmark](latent-mechanism-shift-benchmark.md) adds a
typed Cox baseline and shows that all source-trained models fail when typed
factor effects reverse. These remain synthetic mechanism and failure analyses;
semi-real longitudinal validation is still required.
