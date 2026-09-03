# Multi-entity visual association benchmark

## Purpose

This benchmark checks whether a localized visual property detection is assigned
to one existing entity, rejected as unresolved, or absorbed by the `null/new`
alternative. It validates the association and abstention mechanism on synthetic
scores. It is not evidence about real VLM calibration or real-world visual
grounding.

The benchmark exists because copying one uncertain detection onto every
query-compatible entity creates correlated false state. The robust path instead
keeps entity identity uncertain until a complete candidate distribution passes
predeclared acceptance and margin gates.

## Protocol

The generator creates three visually candidate cups per case:

- two red cups that are equally compatible with the typed language query;
- one blue cup that checks whether query evidence suppresses an irrelevant
  candidate;
- one localized `motion_state=moved` detection whose identity evidence follows
  one of four conditions.

| Condition | Association evidence | Desired safe behavior |
| --- | --- | --- |
| strong | visual and track evidence agree on the true red cup | commit one update |
| ambiguous | two red cups have nearly equal visual affinity | abstain |
| misleading | the wrong red cup has stronger visual and track affinity | abstain |
| null | no existing candidate has meaningful affinity | select no entity |

Each localized detection represents one physical target. Multiple physical
objects must produce multiple detections. `target_entity_id` is stored only on
the benchmark case and is never passed to `MultiEntityAssociator`. Entity
properties contain no `target` or `current_truth` marker.

Calibration and test groups are disjoint. Acceptance and margin thresholds are
searched on the calibration split only. The default run evaluates 80 calibration
cases and 160 untouched test cases. The selection gate requires calibration
false-update rate 0. Test outcomes do not select the policy.

Every test case is also replayed with:

- reversed candidate and entity order;
- a different query surface form with identical typed constraints.

## Run

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_association.py
```

The generated report is
[`artifacts/association_benchmark.json`](../artifacts/association_benchmark.json).

## Verified result

The 2026-08-31 default run selected:

| Policy | Value |
| --- | ---: |
| acceptance threshold | 0.90 |
| top-versus-runner-up margin | 0.25 |
| searched policies | 30 |
| feasible under zero false-update gate | 30 |

Calibration committed 20/80 cases, all correctly, with zero false updates.
The untouched test result was:

| Metric | Result |
| --- | ---: |
| correct updates | 40 / 160 |
| false updates | 0 / 160 |
| abstentions | 120 / 160 |
| selective accuracy | 1.000 |
| target recall | 0.333 |
| null false-positive rate | 0.000 |
| candidate-order invariance | 1.000 |
| query-paraphrase invariance | 1.000 |

All 40 strong cases were committed correctly. All 40 ambiguous, 40 misleading,
and 40 null cases abstained. Abstentions remain in the main denominator; the
0.250 correct-update rate cannot be presented as universal association
accuracy.

## Interpretation and limits

The result validates fail-closed plumbing:

- query compatibility alone does not copy a detection onto both red cups;
- `null/new` competes in the same normalization as existing entities;
- strict pre-event snapshots prevent an update from reinforcing its own
  identity association;
- accepted confidence keeps detection, value, identity posterior, and source
  reliability separate;
- candidate order and query wording do not affect decisions when typed
  constraints are unchanged.

The synthetic affinity distributions are deliberately separated. The benchmark
does not establish that VLM affinities are calibrated, that the chosen
thresholds transfer across cameras or properties, or that systematic
high-confidence identity errors will abstain. It also uses conservative
collision rejection rather than a global multi-detection assignment solver.

The next evidence step is captured-response replay from one or more real VLMs,
with identity labels annotated independently of the query. Thresholds must be
fit on model/source/property-specific validation data, followed by untouched
tests of expected calibration error, false-update risk, source shift, track
breaks, duplicate detections, and global assignment.
