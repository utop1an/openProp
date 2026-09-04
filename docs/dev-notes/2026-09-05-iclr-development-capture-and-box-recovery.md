# ICLR development capture and invalid-box recovery

Date: 2026-09-05

## Readiness decision

Conditional go for development engineering, not calibration/test inference or
submission. Synthetic mechanisms and ALFRED language parsing are supported;
integrated longitudinal visual grounding remains unverified. Older paper and
submission reviews still name TEACh; the active execution route is AI2-THOR
followed by independent real-video confirmation. Neither route currently
supplies a verified integrated result.

Before calibration/test, complete multi-object actions, camera changes, partial
occlusion, development episode manifests, real parser/VLM capture, model/prompt
freezing, and candidate/viability gates. Freeze minimum query coverage, minimum
practical Top-1 effect, maximum false-update increase, and sample-size rules.
Zero accepted errors obtained by abstaining on everything do not prove utility.

## Capture implementation

- CLI and Slurm defaults now cover move_receptacle, open, toggle, dirty, fill,
  cook, slice, and break. Placement uses deterministic coordinate selection
  with PlaceObjectAtPoint; cook/slice/break are one-way state changes.
- Pre-capture settling requires consecutive stable Pass observations using
  displacement and isMoving: defaults 12 maximum steps, two stable steps,
  and 0.005 metres position tolerance.
- Captured status requires action success and the expected target state.
  Target and non-target changes remain evaluation-only; non-target changes
  are explicitly not causal labels.
- New receipts require consistent settling and semantic-success audits;
  legacy schema-v2 bundles remain readable.

This is implementation and deterministic test evidence, not a successful live
eight-family audit. Placement, slice lineage, break fragments, and settling
adequacy still need real Unity verification.

## Job 136673 preparation failure

The user supplied:

    ValueError: AI2-THOR instance box is outside the image

Default candidate extraction checked visible IDs against box keys, not valid
boxes. It therefore passed an unusable box into the strict normalizer.
The repair requires four finite coordinates with positive area inside the
declared image dimensions. Invalid boxes are omitted without clipping; their
entities remain in truth and the visible-object denominator, and appear in
unanchored_visible_entity_ids. Explicit candidate lists remain strict.

The exact offending coordinates and their upstream cause were unavailable
locally. No successful remote retry was reported in this conversation, and
the traceback does not establish successful coverage of all eight families.

After synchronizing code, reuse existing captures with CPU-only preparation:

    cd ~/openProp
    SCENE=FloorPlan1 sbatch hpc/ai2thor_prepare.slurm

Preserve source captures and hashes. Only preparation-report.json is the
completion marker; partial input/truth files are not completion evidence.

## Validation

- Eight-family implementation: 529 deterministic tests passed.
- Invalid-box repair: 530 tests passed in 77.677 seconds.
- Claim verifier: 12 claims, 13 artifacts, 309 checks passed.
- Frozen visual protocol check passed, digest
  d401dbd85b75311bc422b4fee07841cb887e364c41cda7fc3815bdaf7f6f7be7.
- Slurm Bash syntax, Python compilation, and whitespace checks passed.
- Computational snapshots were refreshed; release_ready remains false.

The full-suite snapshot equality test compares newly observed Git revision and
dirty status against the saved manifest. Committing changes that observation:
pre-commit green tests are not a post-commit clean-revision release audit.
Content/runtime verification and release-grade revision binding are separate.

## Follow-ups

1. Retry preparation and inspect all family status/coverage denominators,
   including whether intended targets remain anchored.
2. Audit live eight-family receipts and non-target changes.
3. Complete development scene factors and immutable episode IDs/manifests.
4. Capture real parser/VLM responses and pass candidate/viability gates.
5. Freeze calibration, then run untouched confirmation once.
