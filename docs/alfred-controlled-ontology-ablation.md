# Controlled ontology normalization on ALFRED

Date: 2026-08-26

## Research question

The external ALFRED audit showed that structured parsing often selects useful
properties but emits values at the wrong lexical granularity or with internally
inconsistent relation structure. This experiment tests whether a typed,
fail-closed ontology layer can improve exact canonical values without accessing
validation labels, candidate entities, matcher scores, or target identity.

The intended claim is narrow: controlled ontology consistency improves frozen
language-to-goal-frame outputs. This is not visual grounding, temporal
persistence, or real-world evidence.

## Method boundary

`fit_alfred_training_ontology` reads only `pddl_params.object_target` and
`pddl_params.parent_target` from the ALFRED `train` split. It obtains 53 object
labels and 24 receptacle labels. It does not read training annotation text and
does not open validation files during fitting. A separate evaluation-only audit
confirms 100% object- and receptacle-label coverage on valid-seen and
valid-unseen; this coverage statistic never changes the fitted vocabulary.

The normalizer applies four typed components:

1. object labels: exact or unique token-containment resolution;
2. relation arguments: the same conservative resolution over train receptacles;
3. relation predicates: a predeclared receptacle-class rule enforces `inside`
   for containers and `on` for surfaces;
4. semantic state aliases: predeclared cleanliness and thermal-state aliases.

The latter two are fixed schema semantics, not learned from validation data.
Ambiguous mappings preserve the original value. Unknown destinations preserve
both argument and predicate. Canonical inputs are idempotent. Relevance weights,
property selection, and value families are unchanged.

## Frozen paired protocol

Both models use the exact 40-case valid-unseen sample and raw responses captured
before this method existed. The comparison is ontology-normalized parsing minus
schema-repaired parsing. Model requests and parse failures remain in the
denominator. The primary value metric averages per-case strict canonical-value
recall; uncertainty resamples the 40 shared case clusters within the four task
strata. The aggregate bootstrap keeps the two model outputs for each case in the
same cluster.

| Model | Schema-repaired value recall | Ontology value recall | Paired delta | Improved / regressed cases | 95% stratified bootstrap CI | Exact frames before / after |
|---|---:|---:|---:|---:|---:|---:|
| gemma3:4b | 0.296 | 0.508 | +0.213 | 20 / 0 | [0.146, 0.279] | 0.025 / 0.200 |
| llama3.2 | 0.171 | 0.400 | +0.229 | 23 / 0 | [0.163, 0.296] | 0.000 / 0.000 |

The case-clustered two-model mean delta is +0.221 with 95% CI
[0.160, 0.281]. Per-model paired sign-test values are
`1.91e-6` and `2.38e-7`. Normalization acts on 0.525 of gemma cases and
0.600 of llama cases. These statistics establish a gain on this frozen sample,
not universal model transfer.

## Component ablation

The component analysis micro-averages all 110 gold constraints. Atomic policies
activate one component; leave-one-out policies start from the full normalizer.

| Policy | gemma value recall | llama value recall |
|---|---:|---:|
| schema-repaired, no ontology | 0.300 | 0.173 |
| type only | 0.327 | 0.173 |
| relation argument only | 0.345 | 0.209 |
| relation predicate only | 0.355 | 0.291 |
| state aliases only | 0.345 | 0.209 |
| full without type | 0.482 | 0.409 |
| full without relation argument | 0.427 | 0.327 |
| full without relation predicate | 0.418 | 0.245 |
| full without state aliases | 0.464 | 0.373 |
| full | 0.509 | 0.409 |

The components are complementary for gemma. Llama receives no measurable type
benefit because its frozen responses emit a usable `type` constraint in only a
small minority of cases; relation consistency accounts for most of its gain.
This is a useful failure localization rather than evidence that type
normalization is unnecessary.

## Adversarial interpretation

The strongest alternative explanation is that relation-predicate gains merely
restate the benchmark's receptacle semantics. That criticism is partly correct:
this component is a deterministic typed consistency constraint, not learned
reasoning. Its value is that the constraint is declared once, cannot inspect the
target, and prevents a language model from representing `on drawer` or
`inside table` when the domain ontology rules those combinations out. The paper
must present it as structured inference, not model intelligence.

The method does not resolve four remaining problems:

- missing properties remain missing, so llama exact-frame accuracy stays zero;
- ambiguous labels remain unresolved by design;
- verified annotation/PDDL conflicts remain counted as strict mismatches;
- ALFRED lite still provides no frame-level observation history.

A later [evidence-constrained selector](alfred-evidence-selection.md) targets
property-selection recall while keeping this value layer fixed and confirms its
gain on a task-disjoint sample. The main-task experiment still requires
longitudinal observations and held-out current truth.

## Reproducibility

Run the frozen replays and paired analyses with:

    python scripts/evaluate_alfred_language.py --root artifacts/external/alfred/json_2.1.0 --model gemma3:4b --responses-input artifacts/alfred_language_gemma3_4b.json --output artifacts/alfred_language_gemma3_4b_ontology.json
    python scripts/evaluate_alfred_language.py --root artifacts/external/alfred/json_2.1.0 --model llama3.2 --responses-input artifacts/alfred_language_llama3_2.json --output artifacts/alfred_language_llama3_2_ontology.json
    python scripts/analyze_alfred_ontology_ablation.py artifacts/alfred_language_gemma3_4b_ontology.json artifacts/alfred_language_llama3_2_ontology.json
    python scripts/evaluate_alfred_ontology_components.py --root artifacts/external/alfred/json_2.1.0 --inputs artifacts/alfred_language_gemma3_4b.json artifacts/alfred_language_llama3_2.json

Primary artifacts are `artifacts/alfred_ontology_ablation_analysis.json` and
`artifacts/alfred_ontology_component_ablation.json`.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Controlled typed normalization improves strict canonical values on the frozen ALFRED sample. | +0.221 paired mean delta; case-clustered 95% CI [0.160, 0.281]; no case-level regressions in either model. | Supported for this protocol. |
| Every ontology component is necessary for every model. | Leave-one-out losses occur for relation and state components, but llama shows no type benefit because type selection is mostly absent. | Contradicted as a universal claim. |
| Ontology normalization solves external human-language parsing. | gemma exact-frame accuracy is 0.200 and llama remains 0.000. | Contradicted. |
| The result supports temporal or visual grounding. | ALFRED lite has no frame-level visibility/state history and this evaluation has no candidates. | Unsupported; must not be claimed. |
