# TEACh longitudinal feasibility gate and gold-query case builder

Date: 2026-08-27

## Purpose

OpenProp must not promote a simulator adapter or synthetic fixture into
semi-real performance evidence. This gate makes the required evidence
machine-checkable before any TEACh model comparison is reported.

The implementation has three distinct readiness levels:

- Layer A checks whether replay data contain usable visible observation
  histories and state transitions.
- Layer B checks whether those histories yield nontrivial gold-query temporal
  grounding cases with held-out final truth.
- Layer C checks whether natural dialogue can be aligned to object interactions
  at manually audited precision.

`main_claim_ready` is true only when all three layers pass. Missing statistics
fail rather than becoming zeros that can be interpreted as negative evidence.

## Predeclared pilot-v1 thresholds

These thresholds are engineering sufficiency gates, not a power analysis and
not performance-tuned hyperparameters. They were declared without access to an
official TEACh slice.

| Layer | Check | Required minimum |
|---|---|---:|
| A | Sessions | 30 |
| A | Floorplans | 3 |
| A | Replay snapshots | 300 |
| A | Unique visible entities | 50 |
| A | Observation-history records | 100 |
| A | Interval-censored transitions | 10 |
| A | Properties with transitions | 3 |
| A | Nonempty floorplan-disjoint train/validation/test allocation | required |
| B | Gold-query grounding cases | 50 |
| B | Temporally discriminative cases | 10 |
| B | Minimum candidate-set size | 2 |
| B | Final-truth target ties | 0 |
| C | Automatically aligned dialogue cases | 50 |
| C | Manually labeled alignment cases | 50 |
| C | Manual alignment precision | 0.90 |
| C | Manual audit bound to current automatic cases, manifest, and policy | required |

The floorplan allocator is deterministic, keeps every session from one
floorplan in one partition, and reports the exact floorplans and session counts
assigned to train, validation, and test. It is a feasibility allocation, not a
license to tune thresholds on test outcomes.

## Layer B construction

`build_teach_gold_grounding_cases` receives a reconstructed replay whose final
snapshot is already separated from observation snapshots. It builds matcher
entities only from the latest egocentrically visible observation of each
object. Never-visible objects are not introduced as candidates from final truth.

For each object type and typed state property, a case is retained only when:

1. at least two same-type objects were observed;
2. the property exists for every candidate in final truth; and
3. exactly one candidate has the queried final value.

The query frame contains object type and the typed state value. Boolean values
remain booleans. The complete final values are stored only in the case's
evaluation-only `current_truth` mapping and never copied into entity
observations.

Every generated case is retained for coverage accounting, but the primary
grounding metric is restricted to identifiable cases. It excludes cases where
the target's queried state was never observed, changed after its last
observation, or is tied under all matcher-visible evidence. Those exclusions
prevent evaluation-only final truth from turning an impossible inference into
an apparent model error.

A case is `temporal-discriminative` only when the target has newer matching
evidence while at least one distractor has older evidence for the same queried
value. This is the subset on which persistence can resolve an otherwise
plausible stale distractor. Static-identifiable, unobservable-target-state,
input-evidence-tie, and recency-conflict cases remain reported separately.
The Layer B gate requires at least ten genuinely temporal-discriminative cases,
not merely ten cases involving some hidden state change.

## Official archive preparation

Before downloading or preparing data, freeze the current official access state:

    python scripts/audit_teach_access.py

The audit reads the downloader from a pinned official repository commit, parses
its bucket and archive inventory, issues HEAD requests only to both documented
URL forms for the two required archives, checks the public access incident, and
records any local files by size and SHA-256. Its output is
`artifacts/teach_access_audit.json`. The decision states are deliberately
separate: `official_download_endpoints_accessible`,
`blocked_by_official_host`, `access_unverifiable`, and
`local_archives_present_provenance_unverified`. A local filename alone never
authorizes manifest preparation, and no access state supplies Layer A/B/C or
performance evidence. Use `--require-accessible` when orchestration must stop
unless the official endpoints are currently reachable.

After extracting the official `all_games` and `images_and_states` archives into
one root, build the manifest without hand-selecting sessions:

    python scripts/prepare_teach_manifest.py --data-root <extracted-teach-root>

The command expects `games/<split>/*.game.json` and
`images/<split>/<game_id>/statediff.*.json`. For every selected split it:

1. includes every discovered game rather than silently selecting successful
   examples;
2. requires one unambiguous episode because replay output is keyed at game level;
3. verifies that numeric replay timestamps exactly match all interaction starts
   and that exactly one `statediff.end.json` exists;
4. derives the evaluation horizon as the maximum interaction start plus duration;
5. materializes the episode's official initial state separately from replay
   observations; and
6. records SHA-256 hashes of the game, initial state, replay inventory, and final
   manifest in `artifacts/teach_manifest_preparation.json`.

Any missing, duplicate, malformed, or ambiguous session aborts the full build.
The generated JSONL paths are relative to the manifest when possible, so moving
the extracted root and manifest together preserves the audit contract. Preparation
proves archive integrity and deterministic selection only; it is not dataset
feasibility or model-performance evidence.

## Command and failure semantics

    python scripts/audit_teach_dataset.py --manifest data/teach/manifest.jsonl

The JSON report always records every check, observed value, required value, and
failed check. CI or experiment orchestration can make readiness mandatory:

    python scripts/audit_teach_dataset.py --manifest data/teach/manifest.jsonl --require-ready layer-a
    python scripts/audit_teach_dataset.py --manifest data/teach/manifest.jsonl --require-ready layer-b
    python scripts/audit_teach_dataset.py --manifest data/teach/manifest.jsonl --require-ready main

Layer C accepts a separate frozen manual-audit JSON:

    python scripts/audit_teach_dataset.py --manifest data/teach/manifest.jsonl --dialogue-alignment-audit artifacts/teach_dialogue_alignment_audit.json --require-ready main

Before labeling, construct the current automatic population and deterministic
sample from manifest-linked official game files:

    python scripts/prepare_teach_dialogue_alignment_audit.py --manifest data/teach/manifest.jsonl --sample-size 50 --sample-seed 29

Every manifest row requires `game_file` for this step. The frozen policy scans
recorded interactions in order, retains only Commander text that names exactly
the target's compatible type before the next successful object interaction,
and reports every rejection category. Failed actions do not consume pending
dialogue. The generated template is deliberately incomplete (`is_correct` is
null) and cannot pass the gate before independent labeling. See the
[Layer C alignment audit](teach-dialogue-alignment-audit.md) for the full policy
and coverage contract.

The audit schema is intentionally label-level rather than accepting a claimed
precision scalar:

    {
      "alignment_policy_id": "next-successful-object-v1",
      "frozen_manifest_sha256": "<64 lowercase hex characters>",
      "aligned_cases": 120,
      "labels": [
        {"case_id": "example-001", "is_correct": true}
      ]
    }

Case IDs must be unique, boolean correctness labels are required, and the
manual sample cannot exceed the automatically aligned population. Precision is
derived from the labels, so a hand-entered aggregate cannot satisfy the gate.
At execution time, the manual file is also checked against a fresh automatic
reconstruction: manifest SHA-256, policy ID, total aligned population, and every
labeled case ID must match. The resulting `validated_against_automatic` check is
required for Layer C, so a valid-looking but stale standalone file still fails
closed.

An unmet requested layer exits with status 2 after writing the audit artifact.
This makes the result inspectable while preventing downstream training from
mistaking an inadequate slice for a benchmark.

## Layer B experiment protocol


After the gate passes, run the predeclared train/validation/test experiment:

    python scripts/evaluate_teach_layer_b.py --manifest data/teach/manifest.jsonl

The runner refuses to train unless Layer B is ready. It freezes a reviewer-facing
baseline and ablation matrix before official test evaluation: no decay,
validation-selected fixed decay, a single train-only global hazard, a smoothed
exact-context estimator with global OOD backoff, a property-only factorized
model, and nested property/state, property/subject/state, property/state/scene,
and full typed-factor models. The fixed half-life is selected from the declared
grid (0.25, 1, 4, 16, 64, 256 hours) by validation interval-aware negative
log-likelihood. Every factorized variant is fitted on training floorplans and
receives only one global hazard-scale calibration on validation floorplans.
Survival-evaluation horizons are also fixed from validation only. Test
floorplans are evaluated once after all choices are frozen.

The property-only row is the critical attribution control: a full-model gain is
not evidence for typed context when different properties having different
persistence rates can explain it. The result artifact records SHA-256 digests of
the input manifest and feasibility audit, the exact property set, and the active
features for every ablation. This binds later tables to the qualified input
rather than only to a user-provided path. It also reports validation/test
coverage for every typed feature value and complete context tuple relative to
training support, using identities only and never transition outcomes or current
truth. Thus an exact-context baseline's global backoff rate cannot remain hidden.

The report compares the complete frozen matrix on the same held-out test cases.
The exact-context row uses a declared one-hour global-prior exposure and global
backoff on unseen tuples. The runner uses the repository's standard survival metrics
and temporal-grounding Top-1/MRR/tag metrics. Primary grounding scores exclude
unobservable and matcher-input-tied cases, while the report preserves their
counts and the full generated-case coverage. Synthetic runner tests establish
protocol behavior only; they are not TEACh performance evidence.

## Current executed status

No official TEACh data are present locally. The 2026-08-27 executable access
snapshot binds official repository commit
`903191e256da866a603d1bbfb21db34e0874392d`, verifies that its downloader still
names the `teach-dataset` bucket and both required archives, and records HTTP
403 for both documented URL forms of both `all_games.tar.gz` and
`images_and_states.tar.gz`. Official issue
[alexa/teach#37](https://github.com/alexa/teach/issues/37) remains open; at the
snapshot time it had two confirming comments and no owner, member, or
collaborator response. The exact endpoints, status codes, downloader hash,
issue metadata, and false evidence flags are frozen in
`artifacts/teach_access_audit.json` and separately bound as an external audit in
the reproducibility manifest.

Therefore:


- no official feasibility counts exist;
- strict archive discovery, game/replay pairing, timestamp validation, portable
  manifest hashing, adapter, case-construction, split, and gate behavior are
  unit-tested;
- no Layer A, B, or C readiness claim is made;
- no TEACh model performance result is reported.

Synthetic fixtures prove program invariants only. They do not pass the default
pilot thresholds and are not dataset evidence.

## Reviewer-facing failure boundary

The strongest remaining risk is no longer hidden: the primary external dataset
is currently unavailable from its documented host. Even if access returns,
Layer C still requires an independently labeled alignment sample. Layer B alone
can validate persistence and matching with gold queries, but it cannot support
a claim about language-conditioned temporal grounding.

The minimum defensible paper sequence is:

1. pass Layer A and publish the audit artifact;
2. pass Layer B and compare no-decay, validation-selected fixed, global, and
   factorized persistence on exactly the same held-out identifiable cases while
   reporting excluded-case coverage;
3. freeze the dialogue alignment heuristic;
4. manually label the preselected Layer C sample and pass its precision gate;
5. only then report the joint language-and-temporal result.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| TEACh replay outputs can be converted without exposing final truth to matcher entities. | Unit tests cover initial-relative reconstruction, visibility filtering, final-truth rejection from observations, and separate Layer B truth. | Supported as an implementation invariant. |
| Layer B creates identifiable temporal grounding cases. | Synthetic fixtures yield typed, two-candidate, uniquely labeled cases, separate impossible or tied cases from the primary metric, and identify stale-distractor cases that recency can resolve. | Supported only as mechanism validation. |
| The released TEACh data pass the OpenProp feasibility gate. | No official slice is locally available; documented archive URLs return 403. | Unknown and explicitly not claimed. |
| Extracted official archives can be converted into a deterministic, complete manifest without manual session selection. | Tests cover full-split discovery, exact game/replay timestamp pairing, duplicate and ambiguity rejection, portable paths, and content hashes. | Supported as an implementation invariant; not yet executed on official archives. |
| OpenProp improves semi-real temporal grounding. | No official TEACh performance comparison exists. | Unsupported and must not be claimed. |
