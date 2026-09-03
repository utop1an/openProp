# TEACh Layer C language-grounding protocol

Date: 2026-08-26

## Claim boundary

Layer C measures how language-frame prediction affects object ranking over
strictly pre-action, egocentrically observed memory. It can test a rich frame
only after three independent annotations pass the frozen resolver. The automatic
alignment knows the official interaction target and target type, but it does
not know which additional attributes the Commander intended as referential.
Accordingly, the only automatic oracle is the official target object type.

This distinction prevents two invalid shortcuts: action effects are not copied
into matcher evidence, and final simulator state is not used to manufacture
oracle attributes. A richer property oracle remains a separate annotation
requirement. Its target/candidate/model-blind protocol is specified in
[TEACh Layer C rich-frame annotation](teach-layer-c-rich-annotation.md).

## Fail-closed execution gate

The performance command requires three mutually bound files:

1. the prepared manifest;
2. the complete automatic dialogue-alignment report; and
3. a fresh feasibility audit whose `main_claim_ready` gate passed after the
   independently labeled alignment sample was checked.

The evaluator requires exact agreement on manifest SHA-256, alignment policy,
automatic case count, ordered case IDs, and complete case contents. It also
requires the manual-binding, alignment-count, manual-label-count, and manual-
precision checks to have passed. Changing a query, target, timestamp, or case
population while retaining its ID therefore fails before model evaluation.

## Candidate and evidence construction

For each aligned dialogue segment, the target label is the object ID of the
recorded successful interaction. The matcher candidates are every entity that
was egocentrically visible at least once before that interaction. Each entity
contains only its last visible type, scene, and declared typed state values.

The time boundary is strict: a snapshot at the target action time is excluded.
This prevents the target action result—for example, `isPickedUp=true` after a
pickup—from identifying its own target. `statediff.end.json` is never read as a
matcher observation. Targets never observed before the action remain in the
primary denominator with no recoverable rank.

The oracle frame contains one constraint:

```text
type == official target object type
```

Reports separate type-unique cases, same-type ambiguity, target-type support,
single-candidate trivial cases, and target-unobserved coverage failures. The
same deterministic matcher and no-decay persistence setting are used for all
language conditions so the comparison isolates frame prediction rather than
temporal-model selection.

## Oracle and predicted conditions

Without rich labels, the executable reports four conditions on identical cases:

- `gold`: official target-type oracle;
- `llm-strict`: strict typed response parsing;
- `llm-tolerant`: invalid constraints are retained as explicit validation
  failures where possible;
- `llm-schema-repaired`: schema-only redundant relation repair before parsing.

Raw responses are requested once per unique query and replayed across parser
conditions. Saved `raw_responses` can be supplied to reproduce the comparison
without another model request. Top-1, Top-3, MRR, property/value metrics, and
parse success all use every aligned case as the denominator. Conditional Top-1
is secondary and is never substituted for the all-case result.

With `--rich-annotation-files`, the report separates `type_oracle_report`,
`rich_oracle_report`, and `predicted_reports`. The resolver first verifies the
three complete independent files and the 0.80 pairwise semantic-agreement gate;
the candidate population and all-case denominator remain unchanged.

## Command

After the official manifest, automatic alignment report, manual labels, and
main feasibility audit have passed:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_teach_layer_c.py `
  --manifest data/teach/manifest.jsonl `
  --alignment-cases artifacts/teach_dialogue_alignment_cases.json `
  --feasibility-audit artifacts/teach_feasibility_audit.json `
  --output artifacts/teach_layer_c_results.json
```

Use `--responses-input <prior-result.json>` to replay prior raw responses. The
optional richer evaluation adds:

```text
--rich-annotation-files annotations/a.json annotations/b.json annotations/c.json
```

The result records SHA-256 digests for all three gate inputs, the exact property
set, coverage accounting, response source, and per-case failures.

## Verified implementation invariants

Repository tests establish that:

- target-action and final-state effects cannot enter entity properties;
- every observation timestamp is strictly earlier than the query action;
- unobserved targets and missing language responses remain in denominators;
- same-type ambiguity is reported rather than declared identifiable;
- candidate input order cannot change the deterministic result;
- manifest, policy, population count, IDs, full case contents, and manual gate
  all fail closed when changed.

These are protocol invariants demonstrated on synthetic fixtures. They are not
TEACh performance evidence.

## Remaining evidence requirement

Official archives and completed independent labels are still unavailable in
the workspace. The current type-only oracle is intentionally conservative and
may expose that many aligned instructions do not identify one object among
same-type candidates. The richer protocol and resolver now exist, but no rich
oracle exists until three target-blind label files pass resolution. Attributes
cannot be inferred from the recorded target or post-action state and then called
an oracle.

