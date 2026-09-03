# Typed-context component ablation

Date: 2026-08-26
Status: executed on the frozen protocol

## Reviewer question

The main compositional result compares a full typed factorization with global,
exact-context, and neural baselines, but it does not identify which typed context
axes cause held-out-combination generalization. A reviewer could therefore argue
that one dominant feature solves the generator and that the full method is
unnecessary.

## Frozen protocol

The ablation uses the existing `compositional_location_data` split. All values of
subject type, relation predicate, relation object, and scene occur in training,
while complete validation and test tuples remain held out. The ten seeds are
31, 41, 53, 67, 79, 83, 97, 109, 127, and 149, with 80 histories per context.

Eight additive log-hazard conditions are fixed before result inspection:

| Condition | Subject | Relation predicate + object | Scene |
|---|:---:|:---:|:---:|
| Intercept only | | | |
| Subject only | yes | | |
| Relation only | | yes | |
| Scene only | | | yes |
| Subject + relation | yes | yes | |
| Subject + scene | yes | | yes |
| Relation + scene | | yes | yes |
| Full context | yes | yes | yes |

Every condition uses the same optimizer, training rows, validation rows, test
rows, and grounding cases. Model effects are fit on training only. One scalar
hazard multiplier is fit on validation only. Test outcomes never select a
condition or tune a parameter.

The primary metric is test negative log-likelihood. C-index and integrated Brier
score test ranking and horizon calibration; grounding Top-1 tests whether a
survival difference changes the final entity decision. All condition-versus-full
comparisons are paired by seed. Deltas are oriented so positive means the full
context model is better, with a deterministic 20,000-resample seed bootstrap
interval and win/tie/loss counts.

## Predeclared interpretation

An axis is supported as an independently necessary component only if removing
it from the full model worsens mean test NLL and the paired 95% interval excludes
zero. A tied grounding result cannot support downstream necessity: it means the
current grounding construction is insensitive to that component, even if the
survival metrics differ. The experiment is synthetic mechanism attribution, not
real-world evidence.

Run:

    $env:PYTHONPATH = "src"
    python scripts/evaluate_typed_context_ablation.py

The frozen report path is
`artifacts/typed_context_component_ablation.json`.

## Verified results

The formal ten-seed run completed without changing the condition matrix,
training budget, or metrics. Values are population means across paired seeds.

| Condition | Test NLL lower is better | C-index higher is better | Integrated Brier lower is better | Grounding Top-1 higher is better |
|---|---:|---:|---:|---:|
| Intercept only | 1.863 | 0.500 | 0.277 | 0.000 |
| Subject only | 1.886 | 0.414 | 0.285 | 0.000 |
| Relation only | 1.679 | 0.747 | 0.219 | 0.000 |
| Scene only | 1.514 | 0.718 | 0.148 | 0.967 |
| Subject + relation | 1.664 | 0.719 | 0.219 | 0.000 |
| Subject + scene | 1.418 | 0.709 | 0.122 | 1.000 |
| Relation + scene | 1.314 | 0.747 | 0.095 | 1.000 |
| Full context | **1.262** | **0.747** | **0.083** | **1.000** |

The leave-one-group-out comparisons answer the predeclared necessity question.
All deltas are paired by seed and positive means the full model is better.

| Removed from full | NLL delta with simultaneous 95% interval | Wins / ties / losses | C-index delta | Integrated Brier delta | Grounding Top-1 delta |
|---|---:|---:|---:|---:|---:|
| Scene | 0.402 [0.364, 0.441] | 10 / 0 / 0 | 0.028 [0.013, 0.043] | 0.136 [0.125, 0.145] | 1.000 [1.000, 1.000] |
| Relation | 0.156 [0.137, 0.176] | 10 / 0 / 0 | 0.038 [0.017, 0.059] | 0.039 [0.037, 0.042] | 0.000 [0.000, 0.000] |
| Subject | 0.052 [0.038, 0.066] | 10 / 0 / 0 | 0.000 [0.000, 0.000] | 0.012 [0.010, 0.014] | 0.000 [0.000, 0.000] |

The NLL column uses one shared paired-seed resample and a maximum standardized
mean-deviation critical value across the three predeclared component tests.
Secondary-metric intervals remain comparison-wise diagnostics.

Artifact: `artifacts/typed_context_component_ablation.json`

SHA-256:
`830322504d5b1ae6d945c1530d27b212108d69559260ac5a9c73448d6b4e66f2`

## Interpretation

All three semantic groups satisfy the frozen NLL necessity criterion. Their
effects are not equal: scene is the largest contributor, relation is second,
and subject provides a smaller but consistent calibration gain. Thus the full
factorization is not merely a scene-only model on the survival task.

The downstream result has a narrower boundary. Removing subject or relation
does not change Grounding Top-1, while removing scene makes every case fail.
Subject also leaves C-index unchanged. The current grounding construction
therefore validates that contextual persistence can alter entity choice, but it
does not validate downstream necessity for every typed axis. This limitation
must remain explicit in the paper and motivates a component-balanced grounding
benchmark rather than a stronger claim from the existing cases.

## Claim-evidence map

Claim: each subject, relation, and scene group contributes to held-out-context
survival calibration. | Evidence: every leave-one-group-out NLL delta is positive
in 10/10 paired seeds and each 95% interval excludes zero. | Status: supported
only on the synthetic factorized generator.

Claim: every typed group is necessary for downstream grounding. | Evidence:
subject+scene and relation+scene both tie full-context Grounding Top-1 at 1.000.
| Status: contradicted by the current grounding construction.

Claim: the component ablation proves real-world typed dynamics. | Evidence: the
data generator defines the same factor groups fitted by the models. | Status:
unsupported and must not be claimed.

## Reviewer-style self-review

- **Contribution:** the ablation isolates the source of the existing
  compositional gain; it does not add a new model contribution.
- **Writing clarity:** relation predicate and relation object are grouped because
  the generator assigns one factor to their pair.
- **Experimental strength:** paired seeds, held-out tuples, validation-only
  calibration, and intervals support causal component attribution within the
  generator; ten synthetic seeds are not external evidence.
- **Evaluation completeness:** survival metrics cover likelihood, ranking, and
  horizon calibration, but the grounding cases are not component-balanced.
- **Method soundness:** all conditions share rows and optimization. The fixed
  generator aligns with the model factorization, so misspecification remains a
  separate stress-test question.
## Downstream follow-up

The original grounding cases remain insensitive to subject and relation, so the
negative result above is unchanged for that benchmark. A separately developed
and frozen [component-balanced grounding protocol](component-balanced-grounding.md)
now confirms axis-specific decision utility on untouched seeds by holding all
other typed context fixed and placing the confidence-age crossover between the
relevant hazards. This repairs the evaluation confound without retroactively
changing the original denominator.
