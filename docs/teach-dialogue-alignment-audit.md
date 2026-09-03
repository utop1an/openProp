# TEACh dialogue-to-object alignment audit

Date: 2026-08-27

## Claim boundary

Layer C converts recorded TEACh dialogue and object interactions into candidate
language-grounding cases. It is a data-construction and audit protocol, not a
model result. Synthetic fixtures validate the implementation only. No official
TEACh alignment yield, precision, or grounding performance is currently claimed.

## Frozen policy

The policy ID is `next-successful-object-v1`. For each episode, interactions are
processed in their recorded order. Dialogue accumulates until the next
successful action whose official action type is `ObjectInteraction`; failed
object actions do not consume the pending dialogue.

A candidate is retained only when all of the following hold:

1. the successful interaction has a recorded target object ID;
2. the pending segment contains at least one Commander turn;
3. Commander text explicitly names the target's compatible object type; and
4. Commander text names no other known object type in the scene.

The object vocabulary is the union of initial-state object types and types seen
in successful object interactions. Type matching preserves the typed boundary:
object identity remains the recorded `oid`, while only a normalized compatible
type is used for conservative lexical evidence. Coreference-only, ambiguous,
Driver-only, empty, and targetless segments are rejected rather than guessed.

Every successful object interaction contributes to the denominator. Reports
retain counts for `missing_target_object_id`, `no_dialogue_segment`,
`no_commander_utterance`, `target_type_not_mentioned`, and
`ambiguous_object_type`; malformed utterances are also counted. This makes
coverage loss visible.

## Manifest contract

Layer C uses the same JSONL manifest as Layers A and B, with one additional path
per row:

```json
{
  "episode_id": "...",
  "floorplan": "...",
  "initial_state": ".../initial.json",
  "state_directory": ".../episode_states",
  "final_timestamp": 123.4,
  "game_file": ".../game.json"
}
```

`game_file` must contain the authoritative TEACh game structure: definitions,
tasks, episodes, and ordered interactions. Layer C refuses to freeze a sample
unless every manifest row has a game file. The exact manifest bytes are hashed
with SHA-256 and carried by both the automatic case report and manual label file.

## Reproducible manual audit

Generate all candidates plus the deterministic, intentionally incomplete label
template:

```powershell
$env:PYTHONPATH = "src"
python scripts/prepare_teach_dialogue_alignment_audit.py `
  --manifest data/teach/manifest.jsonl `
  --output-cases artifacts/teach_dialogue_alignment_cases.json `
  --output-label-template artifacts/teach_dialogue_alignment_labels.json `
  --sample-size 50 `
  --sample-seed 29
```

Sampling is order invariant: cases are ranked by a seeded hash of their stable
case ID. The template writes `is_correct: null`; it cannot accidentally satisfy
the gate. A human must label each sampled case true only when the dialogue
unambiguously instructs the recorded target interaction.

After labeling, bind the audit to a fresh automatic reconstruction:

```powershell
python scripts/audit_teach_dataset.py `
  --manifest data/teach/manifest.jsonl `
  --dialogue-alignment-audit artifacts/teach_dialogue_alignment_labels.json `
  --require-ready main
```

The executable rejects mismatched manifest hashes, policy IDs, population
counts, duplicate IDs, non-boolean labels, labels outside the current automatic
case set, and samples larger than the population. Layer C also requires the
derived `validated_against_automatic` marker; a standalone manual file with
self-reported counts cannot make the main gate pass. Precision is recomputed
from label-level booleans and is never accepted as an aggregate input.

## Verified implementation invariants

Repository tests cover:

- official-format definitions, agents, actions, tasks, episodes, and interactions;
- failed object interactions preserving pending dialogue;
- acceptance of unambiguous Commander references;
- rejection of coreference-only, multi-type, Driver-only, and empty segments;
- nondecreasing finite timestamps and unique episode selection;
- manifest provenance and missing-game accounting;
- order-invariant sampling and incomplete label templates; and
- rejection of forged hashes, policies, counts, duplicate or unknown case IDs.

These are software invariants. The default pilot gate still requires at least
50 automatic cases, 50 independently labeled cases, and precision of at least
0.90 on official data before a natural-language temporal-grounding claim.

After the main gate passes, the [Layer C language-grounding protocol](teach-layer-c-grounding.md)
constructs matcher candidates only from evidence visible strictly before each
target interaction. It compares an official target-type oracle against replayed
predicted frames while retaining input-coverage and parse failures in the main
denominator. The evaluator binds complete automatic case contents—not only IDs—
to the passed manual/automatic feasibility audit.

## Remaining evidence requirement

The official archives are not present locally. The 2026-08-27 executable
`artifacts/teach_access_audit.json` binds the official downloader commit and
hash, records HTTP 403 for both URL forms of both required archives, and binds
open issue #37 with no maintainer response. This remains infrastructure evidence
only, so automatic yield and manual precision are unknown. Once verified data
access is restored, the required sequence is: freeze the manifest, generate
candidates, label the preselected sample without changing policy or thresholds,
rerun the bound audit, and publish both automatic coverage and manual precision
artifacts.
