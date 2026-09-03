# ADR 2026-09-03: AI2-THOR candidate eligibility requires a visual anchor

## Status

Accepted

## Context

AI2-THOR can mark an object visible in event metadata without emitting an
instance-detection box for it. Conversely, the segmentation output can contain
static scene geometry that the metadata visibility predicate does not accept.
Requiring a box for every metadata-visible object made valid captures fail,
while inventing boxes or treating missing boxes as negative evidence would
violate the visual grounding contract.

## Decision

For the oracle-box simulator lane, default VLM candidates are the intersection
of metadata-visible entity IDs and valid instance-detection IDs. Objects
outside this intersection remain in evaluation-only truth but are omitted from
the VLM input candidate set.

Preparation reports the visible-object denominator, anchored-candidate
numerator, coverage, and unanchored visible IDs for each frame. Explicit caller
candidate lists retain strict visibility and box validation.

## Consequences

- Missing boxes become auditable missing candidate evidence, not negative
  property evidence.
- Candidate recall is measured with all eligible truth objects in the
  denominator, so intersection filtering cannot silently improve downstream
  scores.
- Simulator identity and raw current truth remain outside VLM and matcher
  inputs.
- The detected/tracked-box experiment lane remains separate from this
  oracle-box mechanism-validation lane.
