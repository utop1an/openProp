# OpenProp

OpenProp is a research framework for **language-conditioned, property-guided
entity matching in an open world**.

An entity owns a flexible set of properties. Each property has a name,
description, value type, value, confidence and provenance. A query selects and
weights only relevant properties, and each value family uses an appropriate
comparison method. Missing observations remain unknown instead of being treated
as mismatches.

## Current project status

The static matcher, LLM/Ollama parsing, irrelevant-attribute stress tests,
temporal evidence, learned persistence pipeline, compositional held-out-context
benchmark, interval-censored observation process, Cox comparison,
latent-mechanism shift audit, and end-to-end temporal grounding benchmark are
implemented. The current evidence validates mechanisms on synthetic data; it is
not yet real-world grounding evidence.
See [the development-note index](docs/dev-notes/README.md) for current progress
and the next academic priorities. The [evidence-locked paper package](paper/README.md)
contains the frozen claim hierarchy, working manuscript, executable result-claim
checks, a content-addressed [computational reproducibility manifest](paper/reproducibility.md),
and the current adversarial submission review. The manifest records the exact
runtime and fails closed on source, protocol, artifact, or command drift.

## What the framework contains

- an extensible property dictionary with aliases and conservative resolution;
- typed observations and explicit `observed` / `unknown` / `not_applicable` states;
- registry-constrained VLM detections, multi-entity association, and audited updates;
- structured predicates that preserve argument identity;
- pluggable property selection and comparator interfaces;
- relevance-weighted matching with separate match and evidence coverage scores;
- an executable red-cup example and unit tests.

## Quick start

The core package has no runtime dependencies.

```powershell
$env:PYTHONPATH = "src"
python examples/red_cup.py
python -m unittest discover -s tests -v
```

Expected ranking:

```text
cup_red              score=1.000 match=1.000 coverage=1.000
cup_unknown_color    score<1.000 match=1.000 coverage<1.000
blue_bowl            score<1.000
```

## Minimal use

```python
from openprop import *

properties = PropertyRegistry()
properties.register(PropertyDefinition("color", "perceived surface color", ValueType.SEMANTIC))

query = QueryFrame("red object", (PropertyConstraint("color", "red", relevance=0.9),))
entities = [Entity("cup-1", {"color": Observation("red")})]

matcher = EntityMatcher(properties, default_comparators(), MentionBasedSelector())
result = matcher.match(query, entities)[0]
```

See [the architecture note](docs/architecture.md) for the design boundary and
research roadmap. The current selector and semantic comparator are transparent
baselines; they are extension points for LLM and embedding implementations.

## Multi-entity visual association

Localized VLM detections are associated against a complete candidate set plus
a `null/new` alternative before any entity history is mutated. Validation-only
threshold and margin calibration, false-update accounting, and candidate-order
and query-paraphrase audits are executable with:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_association.py
```

See [the multi-entity association benchmark](docs/multi-entity-association-benchmark.md)
for the protocol, verified synthetic result, and real-VLM evidence gap.
The [HPC capture guide](hpc/README.md) provides a pinned AI2-THOR SIF,
GPU/Vulkan preflight, persistent build cache, and Slurm pilot template.
For crowded frames, the opt-in GlobalOneToOneAssociator jointly assigns
same-property detections to distinct entities or reusable null; symmetric cases
still abstain. The controlled simulator and frozen evaluation workflow starts
with:

    python scripts/capture_ai2thor_pilot.py --scene FloorPlan1 --platform cloud
    python scripts/freeze_ai2thor_scene_split.py --check
    python scripts/verify_ai2thor_capture.py artifacts/ai2thor_pilot/FloorPlan1.capture-manifest.json
    python scripts/prepare_ai2thor_capture.py artifacts/ai2thor_pilot/FloorPlan1.capture-manifest.json --output-dir artifacts/ai2thor_prepared
    python scripts/prepare_real_video.py artifacts/real_video/manifest.json --output-dir artifacts/real_video_prepared
    python scripts/track_visual_candidates.py --help
    python scripts/calibrate_visual_candidates.py --help
    python scripts/evaluate_visual_candidates.py --help
    python scripts/aggregate_visual_candidates.py --help
    python scripts/compare_candidate_systems.py --help
    python scripts/build_candidate_experiment_artifacts.py --help
    python scripts/verify_vlm_replay.py artifacts/vlm_responses/model-a/FloorPlan1.open.json --input artifacts/ai2thor_prepared/inputs/FloorPlan1.open.json
    python scripts/replay_visual_case.py --help
    python scripts/evaluate_visual_case.py --help
    python scripts/combine_visual_results.py --help
    python scripts/compare_visual_systems.py --help
    python scripts/compare_primary_visual_systems.py --help
    python scripts/freeze_visual_experiment_protocol.py --check
    python scripts/calibrate_visual_acceptance.py --help
    python scripts/calibrate_visual_combined_confidence.py --help
    python scripts/calibrate_visual_query_acceptance.py --help
    python scripts/calibrate_visual_pipeline.py --help
    python scripts/evaluate_visual_results.py --help
    python scripts/build_visual_experiment_artifacts.py --help

The [AI2-THOR and real-video feasibility protocol](docs/ai2thor-vlm-feasibility.md)
defines the end-to-end orchestrator, simulator truth boundary, public and custom
real-video tiers, baselines, metrics, planned tables/plots, and ICLR evidence
gates.

## LLM-backed query parsing

OpenProp includes an OpenAI Responses adapter using strict Structured Outputs.
The LLM turns raw language into a validated `QueryFrame`; deterministic
comparators still perform entity scoring.

```powershell
python -m pip install -e ".[openai]"
$env:OPENAI_API_KEY = "..."
$env:OPENAI_MODEL = "your-structured-output-model"
$env:PYTHONPATH = "src"
python examples/llm_red_cup.py
```

Unknown properties are ignored by default. Pass
`allow_property_creation=True` to `LLMQueryParser` only when dictionary growth
is intended. See [the LLM integration note](docs/llm-integration.md) for the
contract and safeguards.
## Local Ollama testing

No Python dependency or API key is required for local Ollama:

```powershell
$env:PYTHONPATH = "src"
$env:OLLAMA_MODEL = "gemma3:4b"
python examples/ollama_red_cup.py
```

The example defaults to `gemma3:4b`. Set `OLLAMA_HOST` when the server is not at
`http://127.0.0.1:11434`. See [the Ollama test note](docs/ollama.md) for the
verified output and local-model behavior.
## Evaluation benchmark

The `core-v1` benchmark contains 30 bilingual cases across semantic, numeric,
relation, material, and owner properties.

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate.py --strategy gold-weighted
python scripts/evaluate.py --strategy gold-equal
python scripts/evaluate.py --strategy llm-weighted --model gemma3:4b --limit 5
```

See [the evaluation note](docs/evaluation.md) for metric definitions, current
results, and benchmark limitations.
## Irrelevant-attribute stress test

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate.py --dataset interference --strategy gold-weighted
python scripts/evaluate.py --dataset interference --strategy gold-equal
python scripts/evaluate.py --dataset interference --strategy llm-weighted --model gemma3:4b --limit 5
```

See [the interference benchmark note](docs/interference.md) for construction,
results, and limitations.
## Temporal evidence

Stateful properties can define a half-life and event invalidation policy.
Matching accepts an explicit `as_of` timestamp for deterministic replay:

```powershell
$env:PYTHONPATH = "src"
python examples/temporal_states.py
```

See [the temporal evidence note](docs/temporal-evidence.md) for the confidence
formula, event semantics, and calibration boundary.
## Learned persistence

A pluggable survival model can learn context-dependent state persistence from
observed transitions and censored traces:

```powershell
python -m pip install -e ".[ml]"
$env:PYTHONPATH = "src"
python examples/train_contextual_persistence.py
```

The included experiment learns different hazards for `on(cup, table)` and
`inside(cup, cabinet)`. See [the learned persistence note](docs/learned-persistence.md)
for the data contract, objective, verified result, and research limitations.
## Observation-history pipeline

Timestamped state episodes can now be stored as JSONL, split by entity, trained,
calibrated on validation data, evaluated at multiple horizons, and saved as a
reloadable model:

```powershell
$env:PYTHONPATH = "src"
python examples/train_persistence_pipeline.py
```

See [the observation-history pipeline note](docs/observation-history-pipeline.md)
for the schema, generated artifacts, verified metrics, and calibration limits.
## Observation-process bias

Detected changes can now retain the interval between the last confirmation and
the first changed observation. Statistical and neural survival training use the
interval likelihood instead of pretending the detection time is the transition
time:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_observation_process.py
```

See [the observation-process benchmark](docs/observation-process-benchmark.md)
for the five-seed protocol, verified bias reduction, and claim boundary.

## Observation-process effects on grounding

A frozen downstream confirmation now tests whether inspection-frequency bias
changes entity ranking, not only survival likelihood. Two scene strata share the
same latent hazard but use 0.5-hour and 4-hour inspection schedules. On 40
target-scene-balanced analytic cases across ten untouched seeds, detected-time
training reaches Top-1 0.550 +/- 0.150, while interval-aware training and the
true-hazard oracle both reach 1.000. The paired Top-1 gain is 0.450
[0.350, 0.500] with 9/1/0 wins/ties/losses; the target-scene gap falls from
0.900 to 0.000. This remains synthetic controlled decision evidence. See the
[observation-process grounding confirmation](docs/observation-grounding-benchmark.md).

## Informative inspection and missed detection

State-dependent inspection and false-negative detections are now represented as
a joint hidden-state observation process. The training likelihood marginalizes
latent transitions rather than treating missing or negative results as truth:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_informative_observation.py
```

See [the factorial benchmark](docs/informative-observation-benchmark.md) for the
five-seed protocol, opposing-bias failure case, verified results, and limits.

## Training-only observation parameter estimation

A scaled forward-backward EM now estimates the persistence hazard, state-specific
inspection probabilities, and detector sensitivity from training sequences
without validation or test outcomes. A paired sample-size stress test separates
numerical convergence from statistical identifiability:

```powershell
python scripts/evaluate_observation_parameter_estimation.py
python scripts/evaluate_observation_identifiability.py
```

See [the estimation and identifiability report](docs/observation-parameter-estimation-benchmark.md)
for parameter recovery, oracle gaps, sample-efficiency results, and failure cases.

### False-positive observation stress

The hidden observation model now supports imperfect specificity as an explicit,
opt-in emission parameter. A five-seed paired stress test estimates the false-
positive rate from training sequences only and retains the low-noise power
boundary rather than claiming universal benefit:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_false_positive_observation.py
```

At a 0.10 false-positive rate, estimating specificity reduces hazard MAE from
0.0284 to 0.0061 and improves independent exact-test NLL in 5/5 seeds, with a
family-wise simultaneous 95% interval of [0.00343, 0.00887]. At rates 0.02 and
0.05 the simultaneous NLL intervals cross zero. See the
[false-positive observation benchmark](docs/false-positive-observation-benchmark.md).

### Recurrent state transitions

The observation likelihood also has an explicit reversible binary-state option.
It estimates forward and return rates together with inspection, sensitivity,
and false-positive parameters from missing/negative/positive training logs; the
latent training state path is never exposed.

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_recurrent_observation.py
```

Across five paired seeds, the reversible model improves independent exact-state
NLL over the irreversible fit at return rates 0.15, 0.30, and 0.45/h by 0.0604,
0.0810, and 0.0826, with all three simultaneous 95% intervals above zero and
5/5 wins per condition. At zero return, mean NLL differs by only 0.00005. This
is matched synthetic mechanism validation, not real-world evidence. See the
[recurrent observation benchmark](docs/recurrent-observation-benchmark.md).

### Irregular observation timing

Logged recurrent episodes may also provide a distinct elapsed interval before
every opportunity. The exact-interval estimator uses each duration directly in
the CTMC forward-backward likelihood; it does not treat an absent opportunity as
a negative observation or interpolate a synthetic regular grid.

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_irregular_observation.py
```

Across five paired seeds, exact intervals improve current-state NLL over a
common-mean-grid fit by 0.00391, 0.01446, and 0.03140 at bursty gap contrasts
0.50, 0.75, and 0.90. All three simultaneous 95% intervals exclude zero and
each condition wins 5/5 seeds; the regular-grid control differs by only 0.00001.
This is fixed-follow-up synthetic mechanism validation. See the
[irregular observation benchmark](docs/irregular-observation-benchmark.md).
## Strong survival baselines and model misspecification

The held-out-context benchmark now includes factorized exponential and Weibull
statistical models. The factorized exponential model outperforms the neural
model on all four five-seed aggregate metrics, so neural necessity is not
claimed. A separate three-shape experiment verifies that the Weibull model
recovers decreasing and increasing hazards while remaining equivalent when the
exponential assumption is correct.
A frozen ten-seed component ablation further shows that removing scene,
relation, or subject worsens held-out test NLL by 0.402, 0.156, and 0.052,
respectively, with 10/10 paired wins and intervals excluding zero. Grounding is
less complete in the original cases: subject and relation ablations tie the full
model at Top-1 1.000. A separate analytically balanced benchmark then holds all
other context fixed and confirms subject, relation, and scene decision utility
on untouched seeds: full-model probe Top-1 is 0.973/0.993/1.000 and exceeds the
matching missing-axis models by 0.347/0.327/0.500, with all paired intervals
excluding zero. See the [typed-context component ablation](docs/typed-context-component-ablation.md)
and [component-balanced grounding confirmation](docs/component-balanced-grounding.md).

A paired duration-shift experiment adds a piecewise baseline and compares 24 h
in-distribution training with 6/12/24 h train/validation/test follow-up while
holding test rows fixed. All models remain stable under this non-informative
shift, and Weibull retains the best NLL.

Run `scripts/evaluate_weibull_misspecification.py` and
`scripts/evaluate_temporal_shift.py`; see the [misspecification](docs/weibull-misspecification-benchmark.md) and [duration-shift](docs/temporal-shift-benchmark.md) reports.

## Cox and latent-mechanism shift

A typed Cox proportional-hazards baseline with a Breslow cumulative baseline is
now evaluated beside the parametric models. A paired five-condition experiment
### Source-model misspecification audit

A paired 560-run follow-up holds target data and outcome draws fixed while the
deployed source model is globally miscalibrated or receives explicit typed
subject and scene permutations. The interaction gate now has an opt-in
`any_predictive_gain` scope with global-first closure and multiplicity-controlled
parent-to-child descent; the original reversal-only behavior remains the default.

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_source_misspecification_adaptation.py
```

With 288 clean calibration labels, controlled general adaptation improves
scene-swap NLL/C-index from 2.034/0.480 to 1.482/0.733 and joint-permutation
NLL/C-index from 2.126/0.435 to 1.558/0.692. It beats unrestricted global
affine calibration on C-index in 10/10 seeds for both typed shifts; NLL wins are
8/10 for each and the joint NLL interval crosses zero. Clean correct-source runs
remain inactive in 10/10 seeds. The method does not
dominate target-only estimation, detects the subject-cycle shift in 9/10
clean seeds, sometimes over-refines global calibration errors, and
false-activates once under noisy correct-source calibration. See the
[source-misspecification benchmark](docs/source-misspecification-adaptation-benchmark.md)
for the protocol, paired uncertainty, structural audit, and claim limits.

changes global transition rate, hazard shape, or typed factor effects while
holding latent and censoring draws fixed. It uses same-test generator-oracle
regret because raw NLL and Brier scores are not comparable across target
mechanisms with different entropy.

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_latent_mechanism_shift.py
```

Cox reports C-index and integrated Brier score, not a fabricated continuous
event-time NLL for its stepwise baseline. See the
[latent-mechanism shift benchmark](docs/latent-mechanism-shift-benchmark.md) for
the verified five-seed results and catastrophic typed-factor-reversal failure.

## Leakage-safe target adaptation

A ten-seed sample-efficiency experiment now addresses the typed-factor-reversal
failure without target-test leakage. Target calibration pools are selected by
a hash of group identity only, calibration sizes are nested, and every size is
evaluated on the same group-disjoint target test. A two-parameter log-risk
adapter is activated only when its target-calibration slope is negative;
otherwise the source model is returned unchanged.

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_target_adaptation.py
```

Across 10 seeds, 5 calibration sizes, and 5 mechanisms, the gate activates in
all 50 typed-factor-reversal decisions and none of 200 order-preserving
decisions. With six total target labels, mean held-out NLL/C-index/IBS improve
from 3.506/0.183/0.627 to 1.191/0.817/0.129, with 10/10 paired wins for every
metric. These are synthetic results for an exact affine reversal, not a claim
of real-world few-shot adaptation. See the
[target-adaptation benchmark](docs/target-adaptation-benchmark.md) for the
protocol, target-only baseline, full sample-efficiency table, and limits.

### Local and noisy target-adaptation stress test

The follow-up stress test expands the target domain from 3 to all 18 typed
contexts, reverses only one subject or scene group, injects 20% calibration-only
event-status noise, and includes an unrepresentable subject-scene XOR shift.
Grouped sign gates use a predeclared typed factor and return the source model
unchanged for groups whose calibration slope is nonnegative.

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_target_adaptation_stress.py
```

With 144 total target labels, the correct subject gate changes held-out
NLL/C-index/IBS from 1.794/0.576/0.245 to 1.439/0.752/0.147, and the correct
scene gate changes them from 1.830/0.521/0.266 to 1.316/0.705/0.175. Stable
subgroups are exactly unchanged at this sample size, and gains remain under 20%
label flips. At 36 labels, however, the scene gate falsely activates a stable
group in 2/10 seeds. Under the XOR shift, a single-axis gate improves aggregate
ranking while significantly harming stable contexts. These synthetic results
establish a useful local-repair mechanism and a concrete interaction boundary,
not universal adaptation safety. See the [stress benchmark](docs/target-adaptation-stress-benchmark.md).

### Multiplicity-controlled typed interaction adaptation

A hierarchical follow-up searches a fixed family of 12 global, subject, scene,
and subject-by-scene groups without target-test selection. Candidate adapters
are fitted on an identity-disjoint discovery third and must pass a predictive
likelihood-ratio e-value test on the confirmation two-thirds with Bonferroni
family-wise control. A typed heterogeneity veto prevents a pooled parent repair
when finer cells have opposing calibration slopes.

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_target_interaction_adaptation.py
```

On the XOR shift with 288 clean calibration labels, the hierarchy changes
NLL/C-index/IBS from 1.983/0.504/0.272 to 1.500/0.732/0.157 while stable-context
C-index remains exactly 0.747. It beats the strongest single-axis gate on all
three metrics in 10/10 seeds, but is statistically tied with the target-only
per-context model. With 20% label flips it remains beneficial but exact
interaction recovery falls from 8/10 to 3/10. One of 40 noisy in-distribution
decisions false-activates, so neither zero false positives nor general noisy-label
robustness is claimed. See the [interaction benchmark](docs/target-interaction-adaptation-benchmark.md).

### Open-world value support and three-way adaptation

The hierarchy now preserves target-only typed values and searches a genuine
three-way subject-by-object-by-scene partition. Prediction fails closed to the
source model whenever any typed value was absent from target calibration, so a
coarse repair cannot silently transfer onto a test-only value.

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_open_world_adaptation.py
```

Across 200 paired runs, every hierarchical variant exactly matches the source
on the test-only novel-value slice. This is a safety boundary, not zero-shot
recovery: when the true change is confined to that unsupported value, the source
error remains. For a target-calibrated novel value at maximum support, the full
hierarchy improves NLL/C-index/IBS from 1.610/0.647/0.198 to
1.460/0.741/0.152 and selects the subject repair in 10/10 seeds.

On a 3-by-3-by-3 Latin shift, the three-way hierarchy selects the true partition
in 10/10 seeds and reaches 1.447/0.743/0.155 versus
1.490/0.721/0.163 for the pairwise hierarchy. The paired gains are 0.042 NLL,
0.023 C-index, and 0.009 IBS with 10/10 wins. Exact recovery falls to 1/10 at
the smallest calibration size, and noisy controls often activate, so neither
few-shot high-order recovery nor arbitrary-noise robustness is claimed. See the
[open-world higher-order benchmark](docs/open-world-higher-order-adaptation-benchmark.md).

### Local non-affine adaptation stress

A separate 240-run paired audit now tests errors that are nonlinear inside a
typed region: subject-local log-risk saturation, a scene-local folded ordering,
and a smooth subject-by-scene risk bump. Reports separate the affected contexts
from contexts whose deployed predictions are unchanged, and compare global
affine, controlled typed, BIC-screened typed, and target-only repair.

At maximum clean support, the current typed-affine gate reduces all-case NLL on
the smooth bump by 0.048 [0.029, 0.067], but its affected-subset C-index gain is
exactly zero. On the folded scene error it recovers only 0.031 [0.000, 0.093] of
affected C-index versus 0.192 [0.101, 0.279] for the correct-source reference.
With 20% calibration label flips, both typed variants false-activate in 5/10
correct-source controls. These results contradict arbitrary local-repair and
general noise-safety claims, and motivate a small robust nonlinear calibration
family rather than unrestricted flexibility. See the
[non-affine adaptation stress benchmark](docs/non-affine-adaptation-stress-benchmark.md).


A follow-up candidate audit tested that direction rather than assuming it would
work. Affine/quadratic/hinge expansion adds no reliable gain over a matched
sparse-affine control and increases noisy-control activation. A separate sparse
coverage closure reduces fresh-seed noisy correct-source activation from 4/10
to 0/10, but fails its predeclared acceptance rule because fold affected
C-index is unchanged versus the previous BIC gate and bump NLL is 0.003 worse.
Both candidates remain reproducible rejected ablations, not main-method
improvements. See the
[sparse candidate audit](docs/sparse-adaptation-candidate-audit.md).

### Repeated calibration evidence under label noise

A second candidate makes the extra evidence assumption explicit. For each
calibration identity, independent repeated status annotations estimate a
symmetric flip rate from pairwise disagreement; posterior confidence below 0.9
is treated as missing evidence. At 20% synthetic noise, five annotations on the
same 15 cases reduce decoded error from 20.0% to 0.84% and noisy correct-source
activation from 2/10 to 0/10. They require five times the annotation budget.
Equal-budget repetition loses almost all fold power.

The confidence variant improves bump all-case NLL by 0.037 [0.015, 0.060] and
fold affected C-index by 0.019 [0.000, 0.058] versus the deployed source, but
bump affected C-index falls by 0.036 [-0.109, 0.000]. A calibration-concordance
guard accepts the catastrophic held-out seed and does not repair this failure.
The protocol is retained as an identifiability and cost ablation, not a robust
adaptation claim. See the
[repeated-evidence audit](docs/repeated-evidence-label-noise-audit.md).

## TEACh longitudinal benchmark preparation

A dependency-free adapter reconstructs official TEACh replay state diffs,
extracts histories only from egocentrically visible objects, converts invisible
gaps into interval-censored transitions, and keeps the complete final state in
an evaluation-only container. The adapter is implemented and tested; no TEACh
performance result is claimed until an official data slice passes the documented
feasibility gate. A strict preparation command discovers every selected official game, pairs it
with the game-keyed replay directory, verifies exact interaction/state timestamps
and a unique final state, materializes the official episode initial state, and
records content hashes in a portable manifest. Missing or ambiguous sessions
abort preparation instead of being silently filtered. A manifest-driven audit
command then reports sessions,
floorplans, visible entities, property observations and transitions, and exact,
interval-censored, and right-censored records before model training. It also
constructs Layer B gold-query cases only from previously visible candidates,
separates identifiable primary cases from unobservable or matcher-input-tied
coverage cases, audits candidate sizes and unique hidden targets, creates
deterministic floorplan-disjoint partitions, and exposes CI-enforceable
`layer-a`, `layer-b`, and `main` readiness gates. A separate experiment runner
fits persistence on training floorplans, performs half-life selection,
calibration, and horizon selection on validation only, and evaluates test
floorplans once. Layer C now reconstructs official-format game interactions,
aligns only unambiguous Commander references to the next successful typed object
interaction, retains rejection denominators, and freezes an order-invariant
manual-label sample. Main readiness fails closed unless the completed labels are
bound to the current manifest hash, policy ID, automatic population count, and
case IDs; a self-reported aggregate cannot satisfy the gate.

Current official availability can be checked without downloading archive bodies:

```powershell
$env:PYTHONPATH = "src"
python scripts/audit_teach_access.py
```

The checked-in snapshot records four HTTP 403 responses and is explicitly a
non-performance external audit, not TEACh dataset evidence.

See [the longitudinal dataset audit](docs/dataset-feasibility-audit.md) for the
primary-source dataset comparison, split policy, three benchmark layers, and
audit manifest contract, and [the executable TEACh gate](docs/teach-feasibility-gate.md)
for thresholds, Layer B construction, current access status, and claim limits.
The [Layer C alignment audit](docs/teach-dialogue-alignment-audit.md) specifies
the frozen policy, coverage accounting, label protocol, and evidence boundary.

## Temporal grounding benchmark

The integrated benchmark measures whether temporal confidence changes the final
entity decision under stale locations, missing observations, invalidating
events, and irrelevant properties:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_temporal_grounding.py --learned-model artifacts/contextual_persistence.pt
```

See [the temporal grounding benchmark note](docs/temporal-grounding-benchmark.md)
for the hidden-current-truth contract, baseline comparison, verified results,
and research limitations.
