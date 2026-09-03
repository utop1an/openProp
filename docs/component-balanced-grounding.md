# Component-balanced typed grounding

Date: 2026-08-26
Status: confirmed on the single frozen confirmation run

## Reviewer question

The survival ablation shows that subject, relation, and scene each improve test
likelihood, but the original grounding cases are sensitive only to scene. This
protocol asks whether every typed context group can affect the final entity
decision when the decision boundary is made identifiable rather than dominated
by one feature.

## Analytic case construction

For two candidates that both match every query constraint, OpenProp ranks the
location evidence by `confidence * exp(-hazard * age)`. An older, more confident
observation and a newer, less confident observation exchange rank at

`hazard = log(old_confidence / new_confidence) / (old_age - new_age)`.

The benchmark uses this equation directly. Both plausible candidates in one
case share subject, queried relation, and scene. The probed factor varies only
across cases, while the confidence ratio fixes an explicit hazard threshold.
The generator's declared hazard determines whether the old or new entity is the
evaluation target. Current truth remains outside entity properties.

Three probe families are fixed:

| Probe | Fixed context | Values varied | Central threshold per hour |
|---|---|---|---:|
| Subject | relation=`on shelf`, scene=`busy` | book, cup, tool | 0.16 |
| Relation | subject=`tool`, scene=`quiet` | inside cabinet, on shelf, on table | 0.09 |
| Scene | subject=`cup`, relation=`on shelf` | quiet, busy | 0.12 |

Each central threshold is multiplied by 0.90, 0.95, 1.00, 1.05, and 1.10. This
creates 40 unique cases and exactly 20 old-target and 20 new-target decisions.
The thresholds lie strictly between the relevant low- and high-hazard groups;
they were derived from the declared generator factors, not fitted to model
outputs.

## Split protocol and frozen criteria

Development uses seeds 31, 41, 53, 67, 79, 83, 97, 109, 127, and 149. These
seeds were already inspected in the survival component ablation and cannot
support a fresh claim. Confirmation seeds 157, 163, 173, 181, 191, 199, 211,
223, 227, and 239 remain untouched until the analytic cases pass development.

All conditions use identical train, validation, test, and grounding rows:
intercept-only, no-subject, no-relation, no-scene, and full-context. Effects are
fit on training only and the single hazard scale on validation only.

Before confirmation, success requires full-context development Top-1 of at
least 0.95 on every probe and a positive mean paired advantage over the matching
axis ablation. Confirmation supports downstream necessity for an axis only when
the full model again reaches at least 0.95 mean probe Top-1 and the 95% paired
seed-bootstrap interval for its advantage excludes zero. All failures remain in
the denominator.

## Development gate result

The analytic construction passed on all ten previously inspected development
seeds without changing thresholds or cases.

| Condition | Overall Top-1 | Subject probe | Relation probe | Scene probe |
|---|---:|---:|---:|---:|
| Intercept only | 0.495 | 0.333 | 0.653 | 0.500 |
| No subject | 0.772 | 0.613 | 0.780 | 1.000 |
| No relation | 0.802 | 0.807 | 0.667 | 1.000 |
| No scene | 0.565 | 0.333 | 0.840 | 0.500 |
| Full context | **0.990** | **0.987** | **0.987** | **1.000** |

Full-context exceeds the corresponding missing-axis condition by 0.373 on the
subject probe, 0.320 on the relation probe, and 0.500 on the scene probe. Every
paired comparison is a 10/10 win. This clears the development requirement, so
the already declared confirmation seed set is now frozen for one execution.

Development artifact:
`artifacts/component_balanced_grounding_development.json`

SHA-256:
`39aaa3182a0f3248533243404ff94f5cd68deeedef9ecc4d0615d7c85f2ffc5a`

The hash above binds the case definition, model matrix, seeds, and development
outputs before any confirmation result is observed.

Commands:

    $env:PYTHONPATH = "src"
    python scripts/evaluate_component_balanced_grounding.py --phase development
    python scripts/evaluate_component_balanced_grounding.py --phase confirmation

Artifacts are written separately as
`artifacts/component_balanced_grounding_development.json` and
`artifacts/component_balanced_grounding_confirmation.json`.

## Frozen confirmation result

The single confirmation execution passed every predeclared criterion. Values
below are means over the ten untouched confirmation seeds.

| Condition | Overall Top-1 | Subject probe | Relation probe | Scene probe |
|---|---:|---:|---:|---:|
| Intercept only | 0.498 | 0.333 | 0.660 | 0.500 |
| No subject | 0.802 | 0.627 | 0.847 | 1.000 |
| No relation | 0.810 | 0.827 | 0.667 | 1.000 |
| No scene | 0.548 | 0.333 | 0.793 | 0.500 |
| Full context | **0.988** | **0.973** | **0.993** | **1.000** |

The axis-specific paired comparisons are:

| Probe | Matching ablation | Full Top-1 advantage with simultaneous 95% interval | Wins / ties / losses |
|---|---|---:|---:|
| Subject | No subject | 0.347 [0.329, 0.364] | 10 / 0 / 0 |
| Relation | No relation | 0.327 [0.313, 0.340] | 10 / 0 / 0 |
| Scene | No scene | 0.500 [0.500, 0.500] | 10 / 0 / 0 |

The three intervals share each paired seed resample and use a family-wise
maximum standardized mean-deviation critical value.

The denominator retains estimation failures. Seed 223 reaches only 0.733 on the
subject probe and 0.900 overall; seed 199 reaches 0.933 on the relation probe.
These failures lower the aggregate rather than being filtered, while all three
mean probe accuracies remain above the frozen 0.95 requirement.

Confirmation artifact:
`artifacts/component_balanced_grounding_confirmation.json`

SHA-256:
`ef9aa2217f939f1984aeb80c8d65c08d88a00d6cba7dce2c9d64f7239ef233bb`

## Interpretation

The fresh result repairs the evaluation confound exposed by the original
grounding cases. When each axis receives an analytically identified decision
boundary and all other typed context is held fixed within a probe, subject,
relation, and scene estimates each change the final entity ranking. The evidence
therefore supports all-axis downstream decision utility on this controlled
factorized generator.

The result does not turn the benchmark into naturalistic evidence. Targets are
defined from the known generator hazard and confidence-age crossover; the cases
test whether learned persistence transfers that mechanism into a decision, not
whether real users pose the same boundary cases. Official longitudinal data
remain necessary for the paper's external-validity claim.

## Claim-evidence map

Claim: every typed context group can be decision-useful rather than merely
improving survival likelihood. | Evidence: on untouched confirmation seeds,
full-context exceeds the matching missing-axis condition by 0.347, 0.327, and
0.500 Top-1 on subject, relation, and scene probes, with all intervals excluding
zero and 10/10 wins. | Status: supported on the controlled synthetic generator.

Claim: the learned model perfectly recovers all component decisions. | Evidence:
full-context overall Top-1 is 0.988, with seed 223 at 0.900. | Status: contradicted;
finite-sample estimation failures remain.

Claim: the balanced cases establish semi-real temporal grounding. | Evidence:
cases and truth labels are analytically generated from the same factorized hazard
family. | Status: unsupported and must not be claimed.

## Reviewer-style self-review

- **Contribution:** the protocol converts a survival component claim into a
  verified decision-level claim; it is an evaluation repair, not a new model.
- **Writing clarity:** every probe holds non-tested typed context fixed within a
  case and names the omitted model explicitly.
- **Experimental strength:** development and confirmation seeds are disjoint,
  criteria are frozen, all failures remain, and paired intervals exclude zero.
- **Evaluation completeness:** the cases cover three axes and both old/new target
  directions, but only one analytic age-confidence family and one generator.
- **Method soundness:** truth remains evaluation-only and candidate order is
  tested; model and generator factorization remain deliberately aligned.

This remains synthetic mechanism validation. Analytic case balance repairs an
evaluation confound; it does not supply semi-real longitudinal evidence.
