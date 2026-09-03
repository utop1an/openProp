# ADR: Optional global one-to-one visual assignment

Date: 2026-09-01
Status: accepted for experimental comparison

## Context

A VLM can emit several property detections for one frame. Independent
association can send two detections to the same entity-property key. The legacy
batch path rejects every member of that collision, which is safe but can lose
valid updates when two similar objects both changed.

Copying one detection to several plausible entities is not acceptable. It
creates correlated false history and makes later persistence inference appear
more certain than the evidence supports.

## Decision

Keep conservative collision rejection as the default compatibility behavior.
Add GlobalOneToOneAssociator as an explicit experimental alternative.

Within each frame-and-property group:

- each detection chooses one candidate entity or null;
- null is reusable because several detections may be unsupported;
- a real entity can receive at most one detection for that property and frame;
- exact dynamic programming computes the maximum-probability joint assignment;
- the partition function supplies per-detection assignment marginals;
- acceptance, margin, source, modality, and combined-confidence gates are
  reapplied to the assigned marginal;
- a symmetric solution remains an abstention even when the deterministic MAP
  tie-break selects an assignment internally;
- detection order and entity ID presentation order cannot alter the result.

The exact solver fails closed above its declared candidate limit, currently 12.
This limit must be logged as an exclusion/failure, not silently replaced by a
greedy assignment.

## Consequences

The observation ledger still receives only admitted point updates. Alternative
entities and their marginals remain in the association audit. There is no
soft-update fan-out.

Global assignment can recover multiple valid updates that the collision
rejection baseline loses. It cannot make uncalibrated VLM affinity
probabilistic. The words probability and posterior remain operational names
until captured-response calibration supports them.

The ICLR experiment must compare:

- independent association with collision rejection;
- global one-to-one assignment;
- global assignment without null;
- global assignment without the margin gate.

Thresholds and the candidate limit are frozen on calibration scenes. The test
split reports false updates, correct updates, coverage, ECE, Brier, NLL, and
crowded-scene slices with all failures retained.

## Rejected alternatives

### Update every entity above a threshold

Rejected because one physical detection can corrupt several entity histories.
A threshold does not enforce identity exclusivity.

### Greedy highest-edge assignment

Rejected because detection order changes the outcome and local choices can
block a better joint assignment.

### Always replace collision rejection

Rejected until captured multi-change episodes establish calibration and
utility. The new solver is opt-in so existing conservative behavior and
benchmarks remain stable.

### Store the full identity distribution as ordinary properties

Rejected because identity uncertainty is audit/proposal state, not an observed
entity property.
