# Evidence-constrained property selection on ALFRED

Date: 2026-08-26

## Research question

The controlled ontology improves values only after a property has been selected.
The remaining external failure is selection: on the original 40-case sample,
llama3.2 selects `type` in only 2/40 cases and produces many unsupported state
properties. This experiment asks whether explicit query spans can repair
selection while preserving the rule that missing evidence is not negative
evidence.

The claim remains language-layer specific. There are no candidate entities,
visual observations, persistence predictions, or matcher outputs in this
experiment.

## Method

The evidence selector receives only the query, the 53 object and 24 receptacle
labels fitted from ALFRED train PDDL parameters, the fixed property schema, and
the parsed frame. It never receives task type, validation PDDL parameters,
candidate entities, or a target answer.

Every added property requires a logged token span:

- `type`: an exact or uniquely shortened train-vocabulary object mention;
- `location`: a unique train-vocabulary receptacle mention bound to a preceding
  destination preposition;
- `cleanliness` or `thermal_state`: an unambiguous predeclared state cue.

Ambiguous object labels, unbound receptacles, conflicting hot/cold cues, and
queries without a recognizable mention add nothing. Added values then pass
through the already frozen controlled ontology. The selector never overwrites an
existing model value.

The final conflict gate removes one state family only when a positive query cue
explicitly supports the other state family. It does not remove a property merely
because no cue is present. This distinction matters: an aggressive absence gate
regressed valid implicit goals and violates OpenProp's missing-evidence rule.

## Development diagnosis and result

The original frozen sample uses annotation 0 from the first 10 sorted
valid-unseen trajectories per supported task. Before evidence fusion, gemma
selects type/location in 36/40 and 35/40 cases; llama selects them in 2/40 and
31/40. Llama also has 14 cleanliness and 6 thermal-state false positives.

| Model | Ontology property F1 | Fused property F1 | Ontology value recall | Fused value recall | Exact frames before / after |
|---|---:|---:|---:|---:|---:|
| gemma3:4b | 0.787 | 0.870 | 0.508 | 0.625 | 0.200 / 0.350 |
| llama3.2 | 0.545 | 0.794 | 0.400 | 0.675 | 0.000 / 0.425 |

All three metrics improve with no case-level regressions under the final
positive-conflict policy. These are development results because the method was
designed after inspecting this sample.

## Pre-frozen confirmation protocol

Before new model calls, `artifacts/alfred_selection_confirmation_manifest.json`
froze annotation 1 from sorted trajectory positions 10–19 within each of the
four task types. The resulting 40 cases have zero task-ID overlap and zero query
overlap with the development sample. The manifest contains hashes and paths but
no gold labels. Neither query text nor model output was inspected before the
manifest was written.

All request and parse failures remain in the denominator. Raw responses are
stored once and replayed through ontology-only and evidence-fused strategies.

| Model | Parse success | Property F1 before / after | Paired F1 delta (95% CI) | Value recall before / after | Paired value delta (95% CI) | Exact frames before / after |
|---|---:|---:|---:|---:|---:|---:|
| gemma3:4b | 0.775 | 0.648 / 0.765 | +0.117 [0.083, 0.152] | 0.354 / 0.508 | +0.154 [0.108, 0.200] | 0.025 / 0.225 |
| llama3.2 | 0.800 | 0.467 / 0.743 | +0.276 [0.217, 0.337] | 0.292 / 0.579 | +0.287 [0.217, 0.358] | 0.000 / 0.275 |

The shared-case, task-stratified two-model deltas are:

- property F1: +0.196, 95% CI [0.163, 0.231];
- strict canonical-value recall: +0.221, 95% CI [0.175, 0.267];
- exact-frame accuracy: +0.238, 95% CI [0.138, 0.350].

Gemma has 18/0 F1 and value improvements and 8/0 exact-frame improvements.
Llama has 31/0 F1, 26/0 value, and 11/0 exact-frame improvements. This confirms
the direction of the development result on untouched task and annotation text,
although the sample and model family remain small.

## Selection-policy ablation

The confirmation ablation separates addition from two gating policies.

| Model and policy | Precision | Recall | F1 | Value recall | Exact frame |
|---|---:|---:|---:|---:|---:|
| gemma ontology only | 0.717 | 0.604 | 0.648 | 0.354 | 0.025 |
| gemma add only | 0.731 | 0.758 | 0.740 | 0.508 | 0.150 |
| gemma add + positive conflict gate | 0.775 | 0.758 | 0.765 | 0.508 | 0.225 |
| llama ontology only | 0.579 | 0.429 | 0.467 | 0.292 | 0.000 |
| llama add only | 0.644 | 0.742 | 0.680 | 0.579 | 0.100 |
| llama add + positive conflict gate | 0.763 | 0.742 | 0.743 | 0.579 | 0.275 |

Addition drives recall and value gains. Positive-evidence conflict gating raises
precision and exact frames without reducing value recall. The rejected
absence-based gate produced a value regression on the development sample; it is
retained only as a failure ablation and is not the reported method.

## Full validation evidence audit

A model-independent audit evaluates only which properties have explicit span
evidence across all 945 supported validation descriptions.

| Split | Cases | Micro precision | Micro recall | Micro F1 | Exact property set |
|---|---:|---:|---:|---:|---:|
| valid-seen | 487 | 0.999 | 0.711 | 0.831 | 0.405 |
| valid-unseen | 458 | 0.999 | 0.746 | 0.854 | 0.465 |

The near-perfect precision is a mechanism property, not an end-task result:
type and location are present in every supported gold frame, and conservative
rules decline many implicit or ambiguous mentions. Valid-unseen type recall is
0.823, location recall 0.648, cleanliness recall 0.814, and thermal-state recall
0.755. One thermal false positive remains among 458 descriptions.

## Limitations and adversarial interpretation

The strongest reviewer objection is that deterministic lexical extraction is a
conventional hybrid parser rather than a novel learned model. The contribution
is therefore not the existence of rules; it is the auditable typed boundary:
model semantics are retained when present, additions require positive spans,
ambiguity stays unknown, and ontology consistency remains separate from
selection. The paper should frame this as evidence that unconstrained structured
generation is an unreliable selector and that explicit evidence boundaries are
necessary for open-property grounding.

A subsequent train-only BM25 study shows that these two local LLMs are not
strong baselines for this task. The retrieval-plus-positive-evidence method
reaches 0.984 property F1, 0.896 value recall, and 0.675 exact frames on the
same confirmation cases. See `docs/alfred-retrieval-baseline.md`; the original
frozen LLM result remains documented here rather than being retroactively hidden.

Remaining limitations are substantial:

- state cues and destination prepositions are predeclared English rules;
- only two small local model families and one generation per query are tested;
- exact canonical values remain affected by verified ALFRED label conflicts;
- generative-parser confirmation exact-frame accuracy is only 0.225 and 0.275;
- ALFRED lite provides no longitudinal visibility or state history.

## Reproducibility

    python scripts/freeze_alfred_language_sample.py --root artifacts/external/alfred/json_2.1.0 --split valid_unseen --trajectories-per-task 10 --trajectory-offset 10 --annotation-index 1 --reference-offset 0 --reference-annotation-index 0 --output artifacts/alfred_selection_confirmation_manifest.json
    python scripts/evaluate_alfred_language.py --root artifacts/external/alfred/json_2.1.0 --model gemma3:4b --split valid_unseen --trajectories-per-task 10 --trajectory-offset 10 --annotation-index 1 --output artifacts/alfred_selection_confirmation_gemma3_4b.json
    python scripts/evaluate_alfred_language.py --root artifacts/external/alfred/json_2.1.0 --model llama3.2 --split valid_unseen --trajectories-per-task 10 --trajectory-offset 10 --annotation-index 1 --output artifacts/alfred_selection_confirmation_llama3_2.json
    python scripts/analyze_alfred_selection_ablation.py --root artifacts/external/alfred/json_2.1.0 artifacts/alfred_selection_confirmation_gemma3_4b.json artifacts/alfred_selection_confirmation_llama3_2.json --output artifacts/alfred_selection_confirmation_analysis.json
    python scripts/evaluate_alfred_selection_components.py --root artifacts/external/alfred/json_2.1.0 --inputs artifacts/alfred_selection_confirmation_gemma3_4b.json artifacts/alfred_selection_confirmation_llama3_2.json --output artifacts/alfred_selection_confirmation_component_ablation.json
    python scripts/audit_alfred_selection_evidence.py --root artifacts/external/alfred/json_2.1.0 --output artifacts/alfred_selection_evidence_audit.json

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Positive query evidence improves property selection over ontology-only parsing. | Untouched confirmation mean F1 delta +0.196, case-clustered 95% CI [0.163, 0.231], with 49/0 model-case improvements. | Supported on the confirmation protocol. |
| Evidence fusion improves downstream exact canonical frames. | Confirmation exact-frame delta +0.238, 95% CI [0.138, 0.350]. | Supported on this language-only protocol. |
| Absence of a cue is valid negative evidence. | Development absence gating causes value and F1 regressions on implicit clean goals. | Contradicted; absence gating is rejected. |
| Evidence fusion solves external language parsing. | Confirmation exact-frame accuracy remains 0.225/0.275 and parse failures remain. | Contradicted. |
| The result establishes temporal or visual grounding. | No candidates, observations, or persistence predictions are evaluated. | Unsupported; must not be claimed. |
