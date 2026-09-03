# OpenProp architecture

OpenProp treats an entity as an extensible set of heterogeneous properties. It
does not assume that every value belongs in one embedding space.

## Matching pipeline

```text
language / sensor request
          |
          v
      QueryFrame          (from LLMQueryParser or application code)
          |
          v
   PropertySelector       (uses LLM-produced relevance weights)
          |
          v
 PropertyRegistry.resolve
          |
          v
 ComparatorRegistry       (one comparator per value family or property)
          |
          v
 score + match + coverage + per-property evidence
```

`LLMQueryParser` uses the current dictionary and strict structured output to create a
`QueryFrame`. The deterministic `MentionBasedSelector` then consumes its calibrated
weights. This keeps semantic planning separate from entity scoring.

## Robust multi-entity visual association

```text
language query                              visual history
      |                                           |
LLMQueryParser                         VLM localized detection
      |                           value / detection confidence
PropertyConstraint                    / value confidence / track
      |                                           |
      +------------ MultiEntityAssociator --------+
                         |
        pre-event EntityObservationLedger snapshot
                         |
          candidate posterior + null/new entity
                         |
       acceptance + margin + batch-conflict gates
                    /             \\
       abstained hypothesis     committed update
                    |              |
          AssociationAuditLedger   |
                                   v
                         EntityObservationLedger
                                   |
                     persistence + deterministic matcher
                                   |
                         rank + coverage + audit
```

The robust path does not let the VLM decide final identity. It emits a localized
`VisualPropertyDetection` with separate detection, value, visual-association,
and optional track confidence. `MultiEntityAssociator` scores the complete
candidate set against a strictly pre-event snapshot and normalizes candidates
together with a `null/new` alternative.

One localized detection represents one physical target. It becomes a
`PropertyUpdateProposal` only after its top posterior passes acceptance and
top-versus-runner-up margin gates. Ambiguous detections remain auditable and are
not copied onto every candidate. Multiple objects require multiple localized
detections; same-frame detections that collide on one entity property fail
closed. Source and time still come only from trusted `VisualFrame` metadata.

VisualUpdateOrchestrator now materializes base facts, accepted observations,
and entity events through one EntityStateStore; processes frames in timestamp
order; supplies a strictly pre-event snapshot to every same-frame detection;
records every hypothesis; and commits accepted proposals atomically per frame.
Opaque candidate IDs may be bound to trusted normalized candidate regions
from a detector or simulator instance boxes. Region anchors are metadata, not
simulator property truth.
Each localized detection carries its own normalized region independently of
candidate regions and identity affinity. Evaluation-only truth regions are
matched to predictions by exact maximum-cardinality, maximum-IoU assignment.
This produces separate property and association analysis units while retaining
misses, duplicates, false positives, unlocalized outputs, and candidate misses.
Rejected hypotheses retain their raw decision identity for calibration replay.

Missing query evidence is neutral during visual association. When a candidate's
matcher coverage is zero, OpenProp omits the query factor from the association
product instead of multiplying by a zero score; this preserves the invariant
that `unknown` is missing evidence rather than negative evidence. Once query
evidence exists, its typed matcher score participates normally.

For crowded frames, GlobalOneToOneAssociator is an opt-in replacement for
collision rejection. It solves same-frame, same-property detections jointly:
null is reusable, real entities are one-to-one, and exact assignment marginals
feed the same acceptance and margin gates. The solver fails closed beyond its
declared candidate limit. Alternative identity mass remains in audit state and
is never copied into several entity histories.

## Compatibility direct-update boundary

```text
                   Shared PropertyRegistry
                name / type / comparator / policy
                   /                      \\
       language query                  visual history
             |                              |
       LLMQueryParser                VLMPropertyUpdater
             |                              |
    PropertyConstraint            PropertyUpdateProposal
 desired value + relevance       observed value + confidence
                                  + trusted source + time
                   \\                      /
                    EntityObservationLedger
                              |
                history + persistence inference
                              |
                  deterministic EntityMatcher
                              |
                  rank + coverage + audit
```

The VLM emits proposals rather than mutating entities. Each proposal must refer
to a registered property, a candidate entity, and a supplied frame ID. Capture
time and provenance are copied from the trusted `VisualFrame` metadata and are
not model-generated. `PropertyUpdatePolicy` can disable visual updates, require
a confidence floor, or restrict accepted sources. The append-only ledger keeps
all accepted proposals and materializes current or `as_of` entity snapshots;
these snapshots enter the existing deterministic matcher unchanged.

Final query output has its own null-aware decision boundary. Matcher scores are
normalized with a predeclared null weight, ranked deterministically, and gated
by probability, margin, and evidence coverage. Association admission and final
query admission are calibrated separately on calibration rows because they
have different analysis units and error costs. Target identity is attached
only after a query decision is frozen.

Both frozen admission policies can rescale the null mass from already captured
probability rows and optionally make it proportional to
`candidate_count ** power`. Scale, power, threshold, and margin are selected
jointly using calibration rows only. A nonzero count power is identifiable only
when calibration contains multiple candidate-count levels; deployment counts
outside the frozen support abstain rather than extrapolate. Reports retain
explicit association/query slices by candidate count.

The multiplicative detection/value/association/source score is retained as a
raw feature, not asserted to be calibrated. After identity/null admission is
frozen, `FrozenCombinedConfidenceCalibration` fits a Laplace-smoothed isotonic
map on calibration decisions, with source-specific maps only above a declared
support floor and a global fallback otherwise. Application is target-blind and
revoke-only: it may reject an already admitted update below the property safety
floor but can never introduce a new update. Evaluation reports raw and
calibrated ECE, Brier, NLL, and reliability bins.

`calibrate_visual_pipeline.py` is the experiment-level calibration orchestrator.
It freezes association/null admission first, combined confidence second, and
final-query admission third, then writes a single result plus a content-addressed
audit of policies, calibration populations, and execution order. Individual
calibrators remain available only for controlled ablations.

Both frozen admission policies can rescale the null mass from already captured
probability rows and optionally make it proportional to
`candidate_count ** power`. Scale, power, threshold, and margin are selected
jointly using calibration rows only. A nonzero count power is identifiable only
when calibration contains multiple candidate-count levels; deployment counts
outside the frozen support abstain rather than extrapolate. Reports retain
explicit association/query slices by candidate count.

Evaluation attachment is a separate boundary. A truth-free replay outcome is
created first; only then may `evaluate_visual_replay` load frame events and the
query target to build property, association, and query records. Per-case audits
hash the safe input, safe case, captured response, evaluation truth, and output.
The matrix combiner requires identical query and expected-property populations
across systems while retaining model-specific false positives and misses.
Malformed model responses are converted into an empty detection set only at the
model-output boundary; their expected property events and final query remain in
the evaluation denominator and are marked malformed in both records and audit.

Real-video preparation uses the same boundary. A content-addressed,
evaluation-only collection manifest is verified first, then emits separate
`inputs/`, `cases/`, and `truth/` trees. Media hashes, source rights/consent,
room/person cluster splits, candidate identity coverage, frame annotations, and
the frozen independent-annotation agreement gate fail closed before replay.

Candidate generation is an explicit upstream stage rather than an implicit
property of `VisualFrame`. `CandidateTracker` consumes localized detector
proposals, links them globally one-to-one against active tracks using typed
category, IoU, and optional external continuity IDs, and competes every proposal
against a new-track alternative. Empty frames, low-confidence rejections,
capacity overflow, track expiry, and open-world track birth remain explicit.
Only opaque internal track IDs and normalized boxes cross the VLM boundary;
semantic detector labels and external tracker IDs do not.

`CandidateAwareVisualOrchestrator` creates blank entities for newly born tracks
without inventing property evidence, then invokes the ordinary detector,
association, ledger, and query path. Candidate truth is attached only after
tracking freezes. Its separate metrics include candidate recall/precision,
query-target recall, ID switches, fragmentation, track purity, rejected
proposals, and capacity failures, with every frame retained.

Proposal-confidence, link-score, and tolerated-gap parameters are selected only
from paired calibration input/truth artifacts under minimum candidate-recall
and maximum identity-switch-rate gates. Tracking and evaluation CLIs consume the
frozen policy directly and bind its SHA-256 in their audits; test-time command
defaults therefore cannot silently replace the calibrated policy.

Candidate-system inference is exactly paired by `(cluster_id, record_id)` and
requires identical truth-population hashes, source, query frame, query target,
and IoU evaluation threshold. Point estimates pool counts across episodes;
95% intervals resample scene or room/person clusters and recompute pooled
numerators and denominators. The report also retains exact episode-level sign
tests and raw denominators. Unpaired episodes or evaluation-definition drift
fail closed before a paper table or plot can be generated.

## Core design decisions

### Open property dictionary

`PropertyRegistry` stores a name, natural-language description, value type,
aliases and comparator configuration. A new property is added only when
resolution does not find a sufficiently similar existing definition. The
baseline resolver uses aliases and string similarity; embedding or LLM schema
resolution can replace it without changing entities or matchers.

In production, property creation should require a confidence threshold and
usually human review. Otherwise synonyms such as `location`, `position`, and
`spatial relation` can fragment the schema.

### Typed values and comparators

All properties share an interface, but values keep their native structure:

- semantic labels use semantic similarity;
- categories and entity references use identity-aware equality;
- numbers use a distance function and units;
- vectors use cosine similarity;
- relations retain `(predicate, arguments)` and compare both parts;
- a property can override the default comparator by name.

The included semantic comparator is deliberately a dependency-free token
baseline. An embedding-backed comparator can be registered under a new name,
then selected through `PropertyDefinition.comparator`.

### Missing values

`unknown` means there is no evidence. It contributes neither a positive nor a
negative similarity. OpenProp reports:

- `match_score`: agreement over observed evidence;
- `coverage`: relevance-weighted evidence availability;
- `score`: `match_score * coverage^coverage_power`.

Thus an unobserved color does not become a color mismatch, but a fully observed
candidate normally ranks above an otherwise identical uncertain candidate.
`not_applicable` is represented separately so future policies can distinguish
inapplicability from sensor occlusion.

### Persistence model boundary

`EntityMatcher` depends on the `PersistenceModel` protocol. The default
`ExponentialPersistenceModel` applies configured half-lives. An optional neural
survival implementation learns a context-conditioned hazard from transitions
and censored observations. Observation history remains separate from the
property dictionary.
### Temporal evidence

A stateful property may define a `TemporalPolicy`. At match time, observation
confidence is multiplied by half-life decay and by relevant entity events that
occurred after the observation. This affects coverage and final score while
preserving the original value and semantic similarity. Stable properties have
no temporal policy and retain legacy behavior.
## Current research priorities

The static, interference, survival, temporal-grounding, compositional OOD, and
observation-process diagnostics are implemented. Synthetic results show
that typed factorization, not neural parameterization, explains the current
compositional gain; interval censoring reduces inspection-frequency bias; a
shared Weibull shape detects non-exponential dynamics; a joint hidden-state
likelihood separates state-dependent inspection from missed detection; and
training-only EM can estimate observation nuisance parameters when positively
anchored. A Cox baseline now closes the strong-baseline gap on source data, and
paired latent-mechanism tests expose catastrophic failure when typed factor
effects reverse. A calibration-only sign gate now detects the benchmark's exact
reversal and repairs it while preserving the source model on order-stable
shifts. Typed group gates extend that repair to simple local subject or scene
changes, while the stress benchmark shows that small calibration groups can
false-activate and that a single-axis gate harms stable contexts under an XOR
interaction. A hierarchical adapter now searches a fixed global/main/pairwise/
three-way
typed family using identity-disjoint discovery and confirmation data,
likelihood-ratio e-values with family-wise control, a discovery BIC/MDL screen,
child-sign heterogeneity vetoes, and predictive parent-child closure. It repairs
XOR and a genuine 3-by-3-by-3 Latin shift without changing stable contexts at
adequate support. Target-only typed values remain typed; if any value was absent
from target calibration, prediction fails closed to the source model before
partition routing. This prevents unsupported extrapolation but cannot recover a
shift confined to the unsupported value. Calibration noise reduces structural
reliability and can still false-activate controls. A source-misspecification audit adds an opt-in general predictive
scope with global-first and parent-child closure. It repairs typed scene and joint
permutations that global affine calibration cannot, recovers the subject-cycle
partition in 9/10 clean seeds, sometimes over-refines global calibration errors, and does
not dominate target-only fitting. These experiments still validate mechanisms
rather than real-world grounding or universal adaptation safety.
The joint observation model now also represents false-positive emissions and
can estimate specificity from training sequences. Its frozen stress test finds
a clear benefit at false-positive rate 0.10 but no simultaneous NLL separation
at 0.02 or 0.05, exposing rather than hiding the finite-sample power boundary.
A property-specific reversible binary CTMC now relaxes the single-transition
assumption. Its joint forward-backward EM estimates forward/return rates and
observation nuisance parameters from logged outcomes only; the matcher fails
closed for unsupported value types.
An irregular-time extension consumes the elapsed interval before every logged
opportunity and evaluates its exact CTMC transition matrix. A generalized EM
rate update preserves likelihood monotonicity without interpolating absent
opportunities onto a fabricated regular grid.

1. Collect or annotate real or semi-real longitudinal histories with natural missingness and independently audited current truth.
2. Expand the completed evidence-constrained ALFRED selector across model families, repeated generations, and linguistic variation while keeping span evidence and the controlled ontology separately auditable.
3. Stress the completed support-aware higher-order adapter under non-affine
   local errors and semi-real histories.
4. Extend the estimated observation process to informative opportunity timing, multi-valued recurrence, correlated/adversarial sources, and source churn.
5. Add calibrated learned property relevance and controlled embedding-based schema resolution.
6. Resolve relation arguments through entity identity and type rather than raw strings.

See [the development-note index](dev-notes/README.md), [ALFRED external-language validation](alfred-language-external-validation.md), [controlled ontology ablation](alfred-controlled-ontology-ablation.md), [evidence-constrained selection](alfred-evidence-selection.md), [compositional benchmark](compositional-persistence-benchmark.md), [interval benchmark](observation-process-benchmark.md), [informative-observation benchmark](informative-observation-benchmark.md), [observation-parameter estimation benchmark](observation-parameter-estimation-benchmark.md), [latent-mechanism shift benchmark](latent-mechanism-shift-benchmark.md), [target-adaptation benchmark](target-adaptation-benchmark.md), [local adaptation stress benchmark](target-adaptation-stress-benchmark.md), and [typed interaction adaptation benchmark](target-interaction-adaptation-benchmark.md) for verified results and experiment boundaries.

The [source-misspecification benchmark](source-misspecification-adaptation-benchmark.md)
documents the generalized gate, paired results, structural audit, and limits.
The [open-world higher-order benchmark](open-world-higher-order-adaptation-benchmark.md)
documents novel-value support, fail-closed routing, the three-way Latin test,
sample efficiency, label noise, and the complexity-screen ablation.

