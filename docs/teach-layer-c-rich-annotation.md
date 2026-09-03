# TEACh Layer C rich-frame annotation protocol

Date: 2026-08-26

## Purpose and claim boundary

This protocol adds independently labeled referential attributes to the frozen
Layer C text frame. It does not create a performance result, and it does not
infer attributes from the recorded target, candidates, action outcome, or final
simulator state. Formal grounding claims remain unavailable until three valid
annotation files are resolved and the official Layer C evaluation is run.

## Blind annotation view

Each annotator sees only:

- the frozen Commander text;
- the official target-type constraint; and
- the declared boolean state-property vocabulary.

The template excludes target object ID, candidate entities and properties,
observation timestamps, action result and final truth, model outputs, and other
annotators' labels. Its case population and visible contents are SHA-256 bound
to the frozen manifest. `type` is already fixed, while `scene` is constant
within an episode; neither is allowed as an additional annotation.

## Label rule

For every case, choose exactly one status:

- `type_only`: no additional referential attribute is explicit;
- `explicit_attributes`: one or more explicit boolean state attributes occur;
- `uncertain`: the text does not support a confident frame.

Every additional constraint must include exact start/end character offsets and
the matching evidence substring. Annotate an attribute only when it identifies
the referred object. Do not convert the requested action into object state:
`clean the mug` does not assert `isDirty=false`, whereas `the dirty mug` can
assert `isDirty=true`.

## Independent resolution

Exactly three annotators complete templates independently. The resolver checks
the full ordered population, exact blind view, valid spans, typed boolean
values, unique properties, and distinct annotator IDs. A deterministic two-of-
three majority selects each complete semantic label. A majority of `uncertain`
or a three-way disagreement is unresolved and cannot become gold.

The frozen quality gate is mean pairwise exact semantic agreement of at least
0.80 over all cases and all three annotator pairs. This is an annotation-quality
gate, not a model threshold. If it fails, do not lower it after seeing results:
revise the guidelines on separate development examples, freeze a new protocol,
and recollect the official labels.

After resolution, type and each explicit attribute receive equal normalized
relevance. Normalization happens after annotation, so annotators never tune
weights against candidates or outputs.

## Commands

Create the three blind templates after the main Layer C gate passes:

```powershell
$env:PYTHONPATH = "src"
python scripts/prepare_teach_layer_c_annotations.py `
  --manifest data/teach/manifest.jsonl `
  --alignment-cases artifacts/teach_dialogue_alignment_cases.json `
  --feasibility-audit artifacts/teach_feasibility_audit.json `
  --annotator-ids annotator-a annotator-b annotator-c `
  --outputs annotations/a.json annotations/b.json annotations/c.json
```

Resolve completed templates without running a model:

```powershell
python scripts/resolve_teach_layer_c_annotations.py `
  --manifest data/teach/manifest.jsonl `
  --alignment-cases artifacts/teach_dialogue_alignment_cases.json `
  --feasibility-audit artifacts/teach_feasibility_audit.json `
  --annotation-files annotations/a.json annotations/b.json annotations/c.json `
  --output artifacts/teach_layer_c_rich_frames.json
```

Run type-only, rich-oracle, and predicted-frame reports on the same all-case
population by supplying those same annotation files to
`scripts/evaluate_teach_layer_c.py --rich-annotation-files ...`.

## Verified and pending evidence

Tests verify target/candidate/time/model blinding, exact spans, full-population
checks, distinct annotators, deterministic majority resolution, uncertainty
rejection, normalized relevance, and the 0.80 agreement gate on synthetic
fixtures. Procedural independence between human annotators must still be
enforced during collection. No official annotation or TEACh performance
artifact is currently present in the workspace.
