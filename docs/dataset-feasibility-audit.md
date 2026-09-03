# Longitudinal dataset feasibility audit

Date: 2026-08-27

## Required benchmark contract

A main OpenProp benchmark needs all of the following:

1. stable entity identity across time;
2. timestamped observations and state changes;
3. explicit observation opportunities or visibility;
4. natural language that refers to objects or object states;
5. current state held separately from matcher inputs;
6. group- or scene-disjoint train, validation, and test partitions;
7. a license and release path that supports reproducibility.

## Candidate assessment

| Dataset | Identity and timeline | State and observation process | Language | Decision |
|---|---|---|---|---|
| TEACh | Stable AI2-THOR `objectId`, timestamped interactions, human trajectories, initial and final states | Official replay writes a state diff and egocentric image before every interaction; object metadata includes visibility and task-relevant state fields | More than 3,000 free-form human-human dialogues with corrections, ambiguity, and coreference | Primary semi-real benchmark |
| ALFRED | Simulator identities and expert trajectories | Long compositional tasks include irreversible object-state changes, but observation-policy diversity is weaker | 25,000 high- and low-level directives | Controlled fallback and language baseline |
| Ego4D | Real egocentric time and localized video tracks | Real state changes and last-seen queries exist, but long-horizon object identity, state labels, and observation opportunities are split across benchmark subsets | Free-form NLQ, visual queries, and narrations | External real-video validation after the TEACh benchmark |
| BEHAVIOR-1K | Replayable simulator trajectories and rich object state | Strong physics, many state-changing skills, RGB/depth, and human teleoperation | Episode language annotations | High-value second-stage expansion; current 2026 release is 1.44 TB raw and 3.27 TB processed |

Primary sources:

- [TEACh paper](https://arxiv.org/abs/2110.00534) and [official repository](https://github.com/alexa/teach)
- [TEACh interaction schema](https://github.com/alexa/teach/blob/main/src/teach/dataset/interaction.py), [replay writer](https://github.com/alexa/teach/blob/main/src/teach/replay/episode_replay.py), and [state-diff implementation](https://github.com/alexa/teach/blob/main/src/teach/utils.py)
- [ALFRED paper](https://arxiv.org/abs/1912.01734)
- [Ego4D Episodic Memory](https://ego4d-data.org/docs/benchmarks/episodic-memory/) and [Hands and Objects](https://ego4d-data.org/docs/benchmarks/hands-and-objects/)
- [BEHAVIOR-1K paper](https://proceedings.mlr.press/v205/li23a.html) and [2026 dataset format](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/docs/challenge/dataset.md)

## Why TEACh is the first target

TEACh is semi-real rather than fully synthetic in the dimensions most relevant
to OpenProp. Human pairs generate free-form language and human action
trajectories, including mistakes and corrections, while the simulator supplies
auditable object identities and state. The official replay path writes
`statediff.<time>.json` before each interaction and `statediff.end.json` at
completion. This supports an explicit separation:

- observable history: only properties of objects marked visible at each
  egocentric snapshot;
- observation gaps: invisible objects contribute no negative or persistence
  label;
- hidden current truth: the complete final simulator state is evaluation-only.

The local `teach_adapter.py` implements this boundary without importing the
legacy TEACh Python 3.7/3.8 runtime.

## Proposed benchmark layers

### Layer A: persistence estimation

Reconstruct each state snapshot from the episode initial state and its
initial-relative state diff. For each visible object and task-relevant property,
build an episode beginning at the first observation. Repeated equal visible
states extend the last-confirmed boundary. A different state after an invisible
gap creates an interval-censored transition. The last visible confirmation is
right censored.

Use only task-relevant state fields audited in TEACh, including open, dirty,
cooked, toggled, filled, broken, picked-up, and parent-receptacle state.

### Layer B: gold-query temporal grounding

Generate typed query frames from held-out final-state predicates. Candidate
entities share object type and plausible distractor properties. The matcher
receives only prior visible observations. Complete final state determines the
target and audits all candidates but is never copied into entity properties.

This layer measures persistence and matching without language-parser noise.

Primary accuracy is reported only where the target is identifiable from prior
visible evidence and recency can in principle distinguish stale distractors.
Cases whose target state was never observed, changed only after its last
observation, or is tied under matcher-visible evidence remain in explicit
coverage categories rather than being scored as ordinary model failures.
Final truth is used only to generate the query, target, and audit labels.

### Layer C: natural-dialogue temporal grounding

The executable `next-successful-object-v1` policy processes official game
interactions in recorded order. It associates accumulated dialogue with the
next successful typed object interaction only when a target object ID exists
and Commander text names exactly one known compatible type: the target's.
Failed object actions do not consume pending dialogue. Coreference-only,
Driver-only, ambiguous, empty, and targetless segments are rejected with
explicit counts rather than silently disappearing.

This layer jointly tests language parsing and temporal grounding. The alignment
policy must be audited on a deterministic manually labeled sample before it can
support a main-paper claim. The manual file is bound to the exact manifest hash,
policy, automatic population count, and current case IDs; label-level precision
is derived rather than accepted as a claimed scalar. The complete protocol is
documented in the [TEACh dialogue alignment audit](teach-dialogue-alignment-audit.md).

## Split and leakage policy

- Split by floor plan before creating cases; no floor plan may cross train,
  validation, and test.
- Keep all instances from one gameplay session in one partition.
- Fit vocabularies and schemas on training only.
- Calibrate hazards and decision thresholds on validation only.
- Do not use official test trajectories whose future actions or final states
  are withheld.
- Never use `final_state_diff`, `state_changes`, future actions, or
  `statediff.end.json` to construct matcher inputs.
- Retain parser failures, missing targets, and rejected alignments in reported
  coverage denominators.

## Feasibility gate before a result claim

A small official-data slice must first establish:

- number of replayable sessions and snapshots;
- unique objects and task-relevant property observations;
- exact, interval-censored, and right-censored episode counts;
- state-transition counts per property;
- candidate-set sizes and non-tied target margins;
- natural-dialogue alignment yield and manual precision;
- floor-plan-disjoint split sizes.

Until those statistics are generated from released TEACh files, this document
is a verified schema and protocol audit, not experimental evidence.

## Executable feasibility gate

Run:

    python scripts/audit_teach_dataset.py --manifest data/teach/manifest.jsonl

Each JSONL row contains episode_id, floorplan, initial_state,
state_directory, and final_timestamp. Relative paths resolve beside the
manifest. The timestamp must come from the official game record because
statediff.end.json has no timestamp. The report checks three-way
floorplan-disjoint split feasibility, constructs leakage-safe Layer B cases,
and applies predeclared Layer A/B/C pilot thresholds. It labels itself as a data
audit, never as model performance. Downstream automation can require a layer:

    python scripts/audit_teach_dataset.py --manifest data/teach/manifest.jsonl --require-ready layer-b

The command writes the report and exits with status 2 when the requested layer
fails. `main` additionally requires a separate natural-dialogue manual alignment
audit; missing alignment evidence always fails closed. Full thresholds,
construction invariants, and reviewer-facing limits are documented in
[the executable TEACh feasibility gate](teach-feasibility-gate.md).

Once Layer B passes, the frozen experiment path is:

    python scripts/evaluate_teach_layer_b.py --manifest data/teach/manifest.jsonl

It fits persistence on training floorplans, uses validation only for fixed
half-life selection, factorized-model scale calibration, and evaluation
horizons, then evaluates test floorplans once. No-decay, fixed, global, and
factorized strategies share the same held-out identifiable cases.

The 2026-08-27 executable access audit binds official repository commit
`903191e256da866a603d1bbfb21db34e0874392d` and its downloader hash. Both
documented S3 URL forms for both `all_games.tar.gz` and
`images_and_states.tar.gz` return 403, while official issue
[alexa/teach#37](https://github.com/alexa/teach/issues/37) remains open without a
maintainer response or replacement endpoint. The frozen
`artifacts/teach_access_audit.json` records the four HEAD probes, local archive
absence, and explicit false Layer A/B/C and performance-evidence flags. This is
an external data-access constraint, not evidence about benchmark feasibility or
model quality.

## Executed ALFRED fallback audit

The official ALFRED lite trajectory archive remained accessible and was
downloaded from the release URL documented by the project. The 35,748,602-byte
archive contains 6,574 train, 251 valid-seen, 255 valid-unseen, 483 test-seen,
and 488 test-unseen trajectories. OpenProp does not load the test splits by
default.

The validation data support a narrower external contract: human task language,
expert action trajectories, and deterministic task/PDDL goal labels. They do
not contain replayed frame-level visibility or state diffs, so they cannot
replace TEACh for persistence or observation-policy evaluation. Four
single-target task families yield 945 human descriptions; 207 multi-entity,
nested-receptacle, or multiple-instance trajectories are explicitly excluded.

The executable adapter, audit, frozen two-model parser evaluation, and label
alignment analysis are documented in
[ALFRED human-language external validation](alfred-language-external-validation.md).
