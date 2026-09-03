# ALFRED human-language external validation

Date: 2026-08-26

## Claim boundary

This experiment evaluates one external layer only: mapping independently authored
human task descriptions to typed goal-property frames. It does **not** evaluate
visual entity grounding, observation histories, persistence estimation, or
current-state prediction. The official ALFRED 2.1.0 lite trajectory archive
contains language, task metadata, and expert action plans, but not the replayed
frame-level visibility/state stream required by the main OpenProp temporal claim.

The data come from the [official ALFRED repository](https://github.com/askforalfred/alfred)
and its [dataset release instructions](https://github.com/askforalfred/alfred/blob/master/data/README.md).
The acquired `json_2.1.0.7z` archive is 35,748,602 bytes.

## Deterministic feasibility audit

The extracted archive contains 6,574 train, 251 valid-seen, 255 valid-unseen,
483 test-seen, and 488 test-unseen trajectory JSON files. The adapter defaults
to validation splits and does not load test data.

Four single-target task families map without candidate or simulator access to
typed goal frames:

- `pick_and_place_simple`: object type plus destination relation;
- `pick_clean_then_place_in_recep`: type, clean state, and destination;
- `pick_cool_then_place_in_recep`: type, cold state, and destination;
- `pick_heat_then_place_in_recep`: type, hot state, and destination.

Across both validation splits, this yields 945 human descriptions from 299
supported trajectories. Another 207 trajectories are retained as explicit
exclusions: 83 multi-entity light goals, 67 nested movable-receptacle goals,
and 57 multiple-target-instance goals. The audit artifact is
`artifacts/alfred_language_feasibility_audit.json`.

## Frozen-sample language protocol

Before calling a model, the evaluation selects the first annotation from the
first 10 sorted valid-unseen trajectories in each of the four supported task
families. The resulting 40-case sample is identical across models. Every request
or parse failure remains in the primary denominator. Raw responses are stored
once and replayed through strict, tolerant, and schema-repaired parsing.

Schema repair receives only the property schema and raw response. It cannot see
the PDDL labels, task family, candidates, scores, or any current truth.

| Model and strategy | Parse success | Property F1 | Strict canonical-value recall | Exact typed frame |
|---|---:|---:|---:|---:|
| gemma3:4b strict | 0.875 | 0.775 | 0.200 | 0.000 |
| gemma3:4b tolerant | 0.900 | 0.787 | 0.208 | 0.000 |
| gemma3:4b schema-repaired | 0.900 | 0.787 | 0.296 | 0.025 |
| gemma3:4b ontology-normalized | 0.900 | 0.787 | 0.508 | 0.200 |
| llama3.2 strict | 0.575 | 0.375 | 0.138 | 0.000 |
| llama3.2 tolerant | 0.875 | 0.545 | 0.171 | 0.000 |
| llama3.2 schema-repaired | 0.875 | 0.545 | 0.171 | 0.000 |
| llama3.2 ontology-normalized | 0.875 | 0.545 | 0.400 | 0.000 |

The two raw-response artifacts are `artifacts/alfred_language_gemma3_4b.json`
and `artifacts/alfred_language_llama3_2.json`; typed mismatch analysis is in
`artifacts/alfred_language_analysis.json`.

A later controlled ontology layer fitted a 53-object, 24-receptacle vocabulary
from train PDDL labels only and added fixed typed state/relation semantics. On
the same frozen responses its two-model paired value-recall delta is +0.221,
with case-clustered 95% CI [0.160, 0.281] and no case-level regressions. See the
[controlled ontology ablation](alfred-controlled-ontology-ablation.md).

A subsequent evidence-constrained selector adds properties only from logged,
unambiguous query spans and uses positive state evidence to remove conflicts.
On a pre-frozen, task-disjoint 40-case confirmation sample, its two-model mean
property-F1 delta is +0.196 with 95% CI [0.163, 0.231], value-recall delta is
+0.221 [0.175, 0.267], and exact-frame delta is +0.238 [0.138, 0.350]. See
[evidence-constrained property selection](alfred-evidence-selection.md).

The pre-ontology results are negative external evidence: both models frequently
select useful property names but fail exact canonical values, especially
relation predicates and arguments. Schema repair alone increases gemma recall
by 0.088 and recovers one exact frame, while llama triggers no useful repair.
Controlled ontology normalization establishes a separate paired value gain, but
does not establish complete language parsing because exact-frame accuracy
remains low.

## Label-alignment audit

Strict canonical matching is intentionally conservative, but it is not a clean
semantic-accuracy measure. A model-independent audit of all 945 descriptions
finds:

- exact PDDL object phrase present: 0.762;
- exact PDDL destination phrase present: 0.399;
- task-state lexical cue present: 0.721;
- explicit object-label conflict: 60 cases, or 0.063.

An explicit conflict means the PDDL object phrase is absent while another known
full object label is present. One verified example describes a `salt shaker`
while the trajectory label is `pepper shaker`. Other mismatches are plausible
aliases rather than contradictions, such as `bat` versus `baseball bat`.
Therefore the reported value metric is named **strict canonical-value recall**;
it must not be presented as semantic understanding accuracy. The alignment audit
is stored in `artifacts/alfred_label_alignment_audit.json`.

## What this changes

The experiment closes part of the language external-validity gap by replacing
handwritten templates with official human descriptions and expert trajectory
labels. It also falsifies the assumption that structured-output tolerance alone
solves external language grounding. Controlled ontology normalization now
improves exact values substantially, while positive span evidence now improves
selection on a task-disjoint confirmation set. Exact-frame accuracy nevertheless
remains only 0.225 and 0.275, so the next language step is broader model,
generation, and linguistic transfer rather than claiming the parser is solved,
while the temporal claim still requires replayed visibility and state histories
from another release or benchmark.
