# End-to-end VLM grounding feasibility protocol

Date: 2026-09-01
Target: ICLR 2027
Status: implementation and pilot design; no real-VLM result is claimed here.

## Research question

Can an explicit typed observation ledger, conservative visual association, and
property-specific persistence improve language-conditioned current-entity
grounding when objects undergo partially observed state changes?

The primary unit is not VLM property classification in isolation. The final
unit is one delayed language query over a scene history with at least two
candidate entities. Perception, association, update safety, memory calibration,
and final ranking are reported separately so an end-to-end gain cannot hide an
upstream failure.

## Proposed claim-evidence ladder

These are candidate claims, not additions to paper/claims.json. They become
paper-eligible only after the frozen protocols run.

| Candidate claim | Required evidence | Forbidden shortcut |
| --- | --- | --- |
| E1: conservative association reduces false entity updates | AI2-THOR untouched scenes and independently annotated real videos; all detections in denominator | Reporting precision only on accepted updates |
| E2: typed temporal memory improves delayed current-entity queries | End-to-end Top-1/MRR at predeclared delays against latest-frame, no-decay, and direct-VLM baselines | Using simulator current truth as matcher input |
| E3: explicit abstention provides a favorable risk-coverage trade-off | calibration-only thresholds, untouched risk-coverage curve, null and target-missing cases | Selecting threshold on test |
| E4: the mechanism transfers beyond simulator appearance | room/person-disjoint real video plus Ego4D/ADT/EPIC perception slices | Calling simulator-to-simulator transfer real-world evidence |
| E5: multi-detection assignment is necessary in crowded scenes | controlled 1/2/3-event episodes and crowded real-video slice, compared with independent association | Showing only single-object updates |

The current paper blocker N2_REAL_WORLD_GROUNDING remains active until E2 and
E4 have independently held results.

## Evidence tiers

### Tier A: controlled causal simulator

AI2-THOR iTHOR is the primary controllable environment. Its official API
provides RGB frames, object metadata, instance masks and 2D boxes, and actions
for movement and object state changes. It contains 120 artist-designed scenes
across kitchens, living rooms, bedrooms, and bathrooms.

Sources:

- [AI2-THOR environment state and image/metadata API](https://ai2thor.allenai.org/ithor/documentation/environment-state/)
- [Object state-change actions](https://ai2thor.allenai.org/ithor/documentation/object-state-changes/)
- [Interactive movement and placement](https://ai2thor.allenai.org/ithor/documentation/interactive-physics/)
- [Scene distribution](https://ai2thor.allenai.org/ithor/documentation/scenes/)

ProcTHOR supplies procedurally generated layouts for appearance/layout OOD
testing. It is not merged with iTHOR test scenes; it is a separate transfer
column.

Source: [ProcTHOR](https://procthor.allenai.org/).

This tier supports exact interventions, timestamps, object identity, current
state, distractor count, visibility, camera pose, and query delay. It does not
support real camera noise, human motion, natural clutter, or genuine deployment
calibration.

### Tier B: public real egocentric data

No single public dataset provides all OpenProp requirements. Each dataset is
assigned a narrow role.

| Dataset | Useful supervision | OpenProp role | Important limitation |
| --- | --- | --- | --- |
| Ego4D Hands & Objects | pre/PNR/post frames, changed-object boxes, state-change labels | real state-change perception and changed-object association | short clips do not directly provide long-lived entity memory |
| Aria Digital Twin | real egocentric RGB, object trajectories, 2D/3D boxes, segmentation | real moved-object identity, tracking, occlusion, candidate recall | limited semantic state labels |
| EPIC-KITCHENS-100 + VISOR | unscripted kitchen video, action segments, masks and hand-object relations | appearance/source shift and active-object candidate generation | action labels are not complete current-state truth |
| Ego4D Episodic Memory | natural language questions over past egocentric video | query-language stress and retrieval comparison | task localizes past evidence rather than directly scoring current truth |

Sources:

- [Ego4D Hands & Objects](https://ego4d-data.org/docs/benchmarks/hands-and-objects/)
- [Aria Digital Twin](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset)
- [Aria Digital Twin data format](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/data_format)
- [EPIC-KITCHENS dataset](https://epic-kitchens.github.io/)
- [Ego4D Episodic Memory](https://ego4d-data.org/docs/benchmarks/episodic-memory/)

Public-video results are never relabeled as full current-entity grounding when
the necessary final-state truth is absent.

### Tier C: purpose-built real video

A small controlled collection closes the gap left by public datasets.

Pilot design:

- room- and recorder-disjoint train/calibration/test partitions;
- at least two visually similar instances per target episode;
- fixed identity tags visible only to annotators, never to the VLM;
- RGB video plus synchronized intervention log;
- pre-change, point-of-change, post-change, occluded, and delayed-query frames;
- single and multiple simultaneous changes;
- tripod, handheld, low-light, motion-blur, and partial-occlusion sources;
- position/receptacle, open/closed, on/off, filled/empty, clean/dirty, and
  whole/sliced properties where safe and visually meaningful;
- three independent identity/state annotations for the confirmation split.

Final sample size is set after a blinded pilot using the primary paired-effect
variance, not after inspecting test outcomes.

### Executable real-video contract

`openprop-real-video-v1` makes Tier B/C episodes enter the identical replay and
evaluation path as simulator episodes. Its evaluation-only manifest records
content-addressed PNG/JPEG/WebP keyframes, opaque candidate tracks, historical
initial observations, query frames, and separately held event/query truth. The
preparer emits three physical trees: only `inputs/` may be shown to the VLM,
`cases/` supplies non-current historical memory and language constraints to the
deterministic system, and `truth/` is loaded only after the decision is frozen.

The verifier rejects media drift, path traversal, incomplete frame annotation,
candidate/identity mismatch, nonchronological frames, and any room/person
cluster appearing in more than one split. Final confirmation manifests require
at least three annotators, adjudication, a named pre-adjudication agreement
metric, and a frozen agreement floor of at least 0.80. Source manifests also
record whether footage is self-recorded, public-dataset, or licensed-web plus
license, redistribution, and (for self-recorded material) consent basis.

    PYTHONPATH=src python scripts/prepare_real_video.py \
      artifacts/real_video/manifest.json \
      --output-dir artifacts/real_video_prepared \
      --report artifacts/real_video_prepared/preparation-report.json

The resulting input/case/truth triplets are consumed by
`evaluate_visual_case.py`, then combined with simulator results only after the
same paired-population gate. Public datasets with incomplete current-state truth
remain perception/transfer slices and cannot silently enter the end-to-end
query population.

## AI2-THOR episode construction

### Scene split

Use a fixed category-stratified, floorplan-disjoint split before VLM inference:

- development: 18 scenes per room category;
- calibration: 6 scenes per category;
- untouched test: 6 scenes per category.

Thus iTHOR contributes 72 development, 24 calibration, and 24 test scenes.
ProcTHOR houses are identity-disjoint and used only for OOD evaluation.

No test camera pose, object instance, action response, or query may select
prompts, source reliability, null weight, acceptance thresholds, margins, or
persistence parameters.

The executable split is frozen at
`artifacts/ai2thor_protocol/scene_split.json` under protocol
`openprop-ai2thor-scene-split-v1`. It contains all 120 iTHOR scenes, is exactly
balanced at 18/6/6 scenes per room category, and has split digest
`c99adcdc6f7118f11c2bfdd66143beb71279665668c406eb18c92bdfe2d80ea4`.
Regenerate only by an explicit protocol revision; ordinary runs verify it with:

    PYTHONPATH=src python scripts/freeze_ai2thor_scene_split.py --check

### Factorial episode axes

Each feasible action is crossed with:

- changed objects: 0, 1, 2, or 3;
- same-type distractors: 0, 1, 3, or 7;
- target visibility after change: visible, partially occluded, absent;
- camera: fixed, translated, rotated;
- history gap: immediate, short, medium, long simulator time;
- change type: move/receptacle, open, toggle, dirty, fill, cook, slice, break;
- query wording: canonical plus frozen paraphrases;
- candidate source: oracle instance boxes versus detected/tracked boxes.

Not every scene supports every action. The episode manifest records attempted,
successful, failed, and excluded actions. Failed simulator actions stay in the
construction denominator but do not become positive state-change examples.

### Frame and truth contract

For each action:

1. capture pre-action RGB and metadata;
2. execute the simulator action;
3. verify lastActionSuccess;
4. capture immediate post-action RGB and metadata;
5. optionally navigate, occlude, or advance time;
6. capture delayed query frame;
7. write VLM-visible frames and evaluation truth to separate files.

VisualFrame contains trusted source/time, opaque candidate IDs, and normalized
candidate regions derived from instanceDetections2D. Simulator property metadata
is never serialized into the VLM input artifact. The adapter derives changed
properties from separate before/after truth records.

### Query construction

Queries target the current entity, for example:

- Which red mug was moved?
- Find the cup that is now inside the cabinet.
- Which appliance is still on?
- Where is the previously filled mug now?
- Which of the two bowls is likely still clean?

Every query has typed gold constraints, independently authored surface forms,
target identity or explicit no-target truth, candidate set at query time,
required historical evidence, and an identifiability flag. Unidentifiable cases
remain in the denominator and should lead to abstention.

## End-to-end systems and baselines

All systems receive the same RGB frames, candidate proposals, and language
queries unless explicitly marked oracle.

| System | Purpose |
| --- | --- |
| Current-frame VLM answer | tests whether memory is needed |
| Direct VLMPropertyUpdater | compatibility baseline where the VLM chooses entity IDs |
| Latest accepted observation | memory without persistence |
| No-decay OpenProp | isolates temporal modeling |
| Fixed-decay OpenProp | transparent persistence baseline |
| Learned persistence OpenProp | full temporal model |
| Association without query score | tests language-conditioned identity evidence |
| Association without track evidence | tests temporal identity continuity |
| Association without null/margin | tests safety gates |
| Independent per-detection association | current collision-rejection baseline |
| Global one-to-one assignment | implemented opt-in crowded-scene method |
| Oracle boxes/properties/identity | separate upper bounds, never a main method |

At least two VLM model families are evaluated from frozen captured responses.
Model/source/property calibration uses calibration data only. Inference
failures, malformed responses, empty detections, and abstentions stay in the
primary denominator.

One frozen text LLM parser response per query is reused across visual systems.
A deterministic rule parser and oracle typed constraint are language diagnostics;
a second LLM parser is crossed with one VLM only for robustness. This prevents
query-parsing differences from being misattributed to the visual updater or
temporal matcher.

## Metrics

### Candidate generation

- target candidate recall;
- candidates per frame;
- box recall/IoU where boxes exist;
- target-missing rate;
- candidate-set latency.

Candidate recall is reported before association accuracy. Oracle-candidate
results cannot hide retrieval failure.

### VLM property detection

- detection precision/recall/F1 over all target events;
- typed value exact match and macro F1;
- duplicate-detection and missed-event rates;
- state-change temporal localization error where PNR exists.

### Entity association and update safety

- correct and false update rates over all detections;
- false updates per entity-property-hour;
- abstention rate and selective accuracy;
- risk-coverage curve and area under risk-coverage;
- identity Brier score, NLL, and ECE;
- null false-positive, collision, and duplicate rates;
- candidate-order and query-paraphrase invariance.

The normalized value currently called posterior is treated as an association
score until calibration tests justify probabilistic language.

### State memory

At each query horizon:

- current-property accuracy and macro F1;
- stale-positive and stale-negative rates;
- Brier score, NLL, and ECE;
- support coverage;
- accuracy conditional on visibility, source, and observation age.

### Final language-conditioned query

Primary endpoint:

- Top-1 entity accuracy over all query cases, including abstentions and failures.

Secondary endpoints:

- MRR and Top-k recall;
- answer coverage and selective Top-1;
- no-target false-positive rate;
- per-query latency and VLM call count.

All denominators are explicit.

## Statistical protocol

The machine-frozen matrix is
`artifacts/visual_protocol/experiment_protocol.json` under protocol
`openprop-iclr2027-visual-experiment-v2`, with content digest
`d401dbd85b75311bc422b4fee07841cb887e364c41cda7fc3815bdaf7f6f7be7`.
It binds the seven evidence tiers, five simulator seeds, factor axes, 18 system
variants, five primary comparisons, the frozen LLM/VLM factorization,
calibration safety gates, claim boundaries, and current implementation/external-data
gaps. Ordinary runs verify byte-level
stability with:

    PYTHONPATH=src python scripts/freeze_visual_experiment_protocol.py --check

- Split and thresholds are frozen before test inference.
- Resampling is paired and clustered by scene/episode, not frame.
- Report point estimates and 95% cluster-bootstrap intervals.
- Use simultaneous intervals for predeclared primary baseline comparisons.
- Aggregate at least five simulator seeds and show seed-level points.
- Real-video confirmation is participant/room clustered.
- Calibration metrics are fit on calibration only and evaluated once on test.
- Model comparison uses identical captured inputs and cached raw responses.
- Missing model output is a failure/abstention, never a dropped row.
- Zero-change and target-absent controls are reported separately.
- Candidate-order and paraphrase invariance are hard audit gates.

The four preregistered primary query baselines are evaluated in one shared
cluster-resampling run. Top-1 differences receive max-studentized family-wise
simultaneous 95% intervals and exact McNemar p-values with Holm adjustment; MRR
remains secondary with paired cluster-percentile intervals and exact sign tests:

    PYTHONPATH=src python scripts/compare_primary_visual_systems.py \
      --input artifacts/visual_results.jsonl \
      --main-system openprop_learned_global \
      --baselines current_frame_vlm direct_vlm_updater \
        latest_accepted_observation openprop_no_decay \
      --split test \
      --output artifacts/visual_pairwise/primary-query-family.json

## Planned paper tables

1. Main end-to-end result: systems by AI2-THOR test, ProcTHOR OOD, and custom
   real video, with Top-1, MRR, false-update rate, and coverage.
2. Perception/association decomposition: candidate recall, property F1,
   identity accuracy, ECE, null FPR, and duplicate rate.
3. Temporal horizon result: current-state accuracy/Brier and final Top-1 at
   each delay.
4. Component ablation: query score, region anchors, track evidence, null,
   margin, persistence, source reliability, and global assignment.
5. Domain/source transfer: simulator versus real calibration, property/source
   slices, and calibration sample efficiency.
6. Failure/safety table: ambiguous, misleading, target absent, occluded,
   multi-change, crowded, camera shift, and malformed-response slices.

Tables use booktabs, metric directions, consistent precision, complete
denominators, and captions stating split and evidence boundary.

## Planned plots

- reliability diagrams for identity and final-state confidence;
- selective risk-coverage curves;
- final Top-1 and Brier versus query delay;
- false-update rate versus same-type distractor count;
- candidate recall versus candidate-set size;
- calibration sample-efficiency curves;
- per-property/source forest plot with clustered intervals;
- typed-value confusion matrices;
- qualitative pre/change/post/query panels with boxes, hypotheses, and history.

Every plot is generated from content-addressed JSON with a deterministic check
mode.

## Feasibility gates

### Gate A: simulator extraction

Pass when AI2-THOR is pinned, every supported state family has at least one
successful before/after capture, RGB/boxes/metadata/action results agree, and
truth never appears in VLM serialization.

### Gate B: perception

Pass when calibration scenes show adequate target candidate recall, at least one
frozen VLM has nontrivial typed state-change recall, and malformed/duplicate
outputs are bounded and reported. Thresholds freeze before Gate C.

### Gate C: end-to-end simulator

Pass when untouched test results show either a paired Top-1 improvement over
direct-update and no-persistence baselines without higher false-update rate, or
a materially lower false-update rate at a predeclared coverage floor.

### Gate D: real-video confirmation

Pass when the primary safety/grounding effect direction is retained on
room/person-disjoint real video with independent annotation. Ego4D, ADT, and
EPIC slices support perception/transfer claims only.

## Implementation status

Completed in the first pass:

- EntityStateStore unifies base facts, accepted observations, and events;
- VisualUpdateOrchestrator provides chronological processing, pre-event
  snapshots, automatic audit, and atomic per-frame commit;
- trusted normalized candidate regions enter both VLM payloads;
- AI2-THOR typed property registry and metadata/frame adapter;
- evaluation truth held in separate frame/transition types;
- captured metadata preparation script producing separate input/truth files;
- deterministic tests independent of Unity;
- optional exact global one-to-one assignment with reusable null, assignment
  marginals, symmetric abstention, and fail-closed candidate limit;
- Linux/CloudRendering pilot capture runner for open, toggle, dirty, and fill;
- frozen, model-output-blind 72/24/24 iTHOR scene assignment with deterministic
  byte-level drift checking;
- schema-v2 portable capture bundles with per-file size/hash verification and
  physical `inputs/` versus evaluation-only `truth/` preparation;
- strict property/association/query JSONL evaluation records;
- normalized detection regions separated from candidate affinity;
- exact IoU-based truth matching that retains missed, duplicate, unlocalized,
  malformed, false-positive, and candidate-missing cases;
- deterministic Markdown/LaTeX table and four-panel PNG generation;
- truth-free captured-response replay through detection, association, ledger,
  deterministic matching, and null-aware final query decision;
- independent calibration-only acceptance policies for entity updates and
  final query answers;
- zero-coverage query evidence treated neutrally during association, preserving
  the `unknown`-is-missing invariant;
- schema-v1 real-video manifest verification and preparation with media hashes,
  provenance/consent fields, room/person-disjoint split enforcement, a
  three-annotator agreement gate, and replay-compatible truth separation;
- malformed VLM responses retained as empty detections so expected events and
  final queries remain in the predeclared denominator.
- calibration-only candidate-count-aware null rescaling for association and
  final-query decisions, including an identifiability gate and fail-closed
  behavior outside frozen candidate-count support.
- Laplace-smoothed monotone calibration of the combined update confidence,
  with support-gated source-specific mappings, global fallback, revoke-only
  application, and raw-versus-calibrated ECE/Brier/NLL artifacts.
- truth-free proposal-to-track candidate generation with one-to-one linking,
  explicit new-track competition, empty/occluded frames, capacity and rejection
  audit, open-world entity birth, and VLM replay-compatible opaque track IDs;
- candidate evaluation and multi-system paper artifacts covering recall,
  precision, query-target recall, identity switches, fragmentation, purity, and
  capacity failures over all frames.

AI2-THOR 5.0.0 is installed and its Python API imports successfully. A local
Windows smoke test on 2026-09-01 failed before scene launch because no Unity
build exists for the pinned commit/Windows architecture. This is a capture
infrastructure blocker, not a model result. Run capture on Ubuntu with
`CloudRendering` (or an approved hosted environment), then replay the portable
RGB/metadata/box artifacts on any platform. No VLM feasibility metric is yet claimed.

## Immediate execution order

1. Provision an Ubuntu/CloudRendering capture worker for pinned AI2-THOR 5.0.0.
2. Run the live capture runner for one kitchen scene and four state families.
3. Freeze an initial eight-scene development pilot with RGB/metadata/boxes.
4. Run captured-response VLM detection with opaque IDs plus region anchors.
5. Audit candidate recall, property F1, association calibration, and false
   updates before scaling.
6. Compare collision rejection and global assignment on identical captures.
7. Run deterministic JSON-to-table and JSON-to-plot scripts on frozen responses.
8. Start custom-video annotation after the simulator contract is stable.

Ubuntu headless capture:

    python scripts/capture_ai2thor_pilot.py \
      --scene FloorPlan1 \
      --families open toggle dirty fill \
      --platform cloud \
      --output-dir artifacts/ai2thor_pilot

Verify the copied bundle before any VLM replay or truth extraction:

    PYTHONPATH=src python scripts/verify_ai2thor_capture.py \
      artifacts/ai2thor_pilot/FloorPlan1.capture-manifest.json \
      --report artifacts/ai2thor_pilot/FloorPlan1.verification.json

Then generate physically separated, hash-bound replay inputs and evaluation
truth. Only files below `inputs/` may be passed to the VLM pipeline:

    PYTHONPATH=src python scripts/prepare_ai2thor_capture.py \
      artifacts/ai2thor_pilot/FloorPlan1.capture-manifest.json \
      --output-dir artifacts/ai2thor_prepared

### Portable capture bundle and current runtime status

The capture manifest is schema version 2. Every RGB image, metadata JSON, and
instance-box JSON is referenced by a bundle-relative path, exact byte count,
and SHA-256 digest. The verifier rejects absolute paths, parent traversal,
missing files, byte/hash drift, malformed metadata, contradictory action
status, duplicate families, and completed runs that omit a requested family.
This makes GPU-Linux or remote capture portable without trusting the transfer.
The manifest remains evaluation-only: verification reports artifact integrity
but does not expose metadata or target identity to the matcher.

The preparation step re-verifies the complete bundle before reading any
metadata. It writes VLM-visible frame history to `inputs/` and simulator state
and transition labels to a separate `truth/` directory. Both outputs and the
preparation report are bound to the exact capture-manifest digest. Input JSON
contains no object truth, target identity, transition label, or truth-file
pointer.

Captured VLM responses use a separate replay contract. Each response records
provider, model, system ID, request settings, episode ID, and the exact SHA-256
of its VLM input artifact. The writer recursively rejects simulator-truth keys
before accepting a response; the reader refuses input hash or episode drift.
This permits identical captured frames to be replayed through competing
association/calibration systems without making another model call:

    PYTHONPATH=src python scripts/verify_vlm_replay.py \
      artifacts/vlm_responses/model-a/FloorPlan1.open.json \
      --input artifacts/ai2thor_prepared/inputs/FloorPlan1.open.json

The truth-free execution command then parses the captured response, associates
detections against strictly pre-frame snapshots, commits admitted proposals,
materializes the delayed ledger, and produces a null-aware final query
distribution. It accepts no truth-file argument:

    PYTHONPATH=src python scripts/replay_visual_case.py \
      --input artifacts/ai2thor_prepared/inputs/FloorPlan1.open.json \
      --case artifacts/visual_cases/FloorPlan1.open.json \
      --response artifacts/vlm_responses/model-a/FloorPlan1.open.json \
      --assignment global \
      --output artifacts/visual_replays/model-a/FloorPlan1.open.global.json

For detector-proposal experiments, candidate tracking runs before this command.
The first command never accepts truth; the second reruns and freezes tracking
before loading its separate evaluation file:

    PYTHONPATH=src python scripts/calibrate_visual_candidates.py \
      --input artifacts/candidate_inputs/calibration-a.json \
      --truth artifacts/candidate_truth/calibration-a.json \
      --output artifacts/candidate_policy.json
    PYTHONPATH=src python scripts/track_visual_candidates.py \
      --input artifacts/candidate_inputs/FloorPlan1.open.json \
      --policy artifacts/candidate_policy.json \
      --output artifacts/tracked_inputs/FloorPlan1.open.json
    PYTHONPATH=src python scripts/evaluate_visual_candidates.py \
      --input artifacts/candidate_inputs/FloorPlan1.open.json \
      --truth artifacts/candidate_truth/FloorPlan1.open.json \
      --policy artifacts/candidate_policy.json \
      --system detector-external-track \
      --output artifacts/candidate_results/FloorPlan1.open.json

Candidate systems are aggregated and rendered independently of downstream VLM
quality:

    PYTHONPATH=src python scripts/aggregate_visual_candidates.py \
      --input \
        artifacts/candidate_results/FloorPlan1.open.json \
        artifacts/candidate_results/FloorPlan2.open.json \
      --split test \
      --output artifacts/candidate_test.json
    PYTHONPATH=src python scripts/compare_candidate_systems.py \
      --input \
        artifacts/candidate_results/baseline/FloorPlan1.open.json \
        artifacts/candidate_results/openprop/FloorPlan1.open.json \
      --baseline detector-framewise --system detector-openprop-track \
      --split test \
      --output artifacts/candidate_pairwise/openprop-vs-framewise.json
    PYTHONPATH=src python scripts/build_candidate_experiment_artifacts.py \
      --report artifacts/candidate_test.json \
      --comparison artifacts/candidate_pairwise/openprop-vs-framewise.json \
      --output-dir artifacts/candidate_paper

The pairwise gate requires the same `(cluster_id, record_id)`, exact
truth-population SHA-256, source, query frame, query target, and IoU threshold.
Point estimates pool metric numerators and denominators instead of averaging
episode percentages. Percentile 95% intervals resample scene or room/person
clusters and recompute both systems' pooled rates; exact episode-level sign
tests are retained as a secondary audit. Candidate recall/precision,
query-target recall, purity, identity switches per 100 matches, fragmentation
per 100 truth observations, capacity-failure rate, rejection load, and candidate
load all retain their explicit denominators. The artifact builder renders these
effects and intervals in Markdown, LaTeX, and PNG.

Association and final-query admission gates are fitted independently. Both
search calibration rows only and are then applied target-blind to every split.
The final-query gate has a separate safety limit on false accepted answers:

The frozen search includes a relative null scale and an optional
candidate-count exponent in addition to acceptance and margin. The exponent is
disabled unless calibration contains at least two candidate-count levels, and
any deployment count absent from calibration support forces abstention. This
tests the fixed-null failure hypothesis without reissuing VLM calls or
consulting test targets.

    PYTHONPATH=src python scripts/calibrate_visual_query_acceptance.py --help

After decisions are frozen, a separate evaluation command attaches frame-event
and query-target truth and emits all three metric record types. It also writes a
sidecar audit with SHA-256 for every source and output artifact:

    PYTHONPATH=src python scripts/evaluate_visual_case.py \
      --input artifacts/ai2thor_prepared/inputs/FloorPlan1.open.json \
      --case artifacts/visual_cases/FloorPlan1.open.json \
      --response artifacts/vlm_responses/model-a/FloorPlan1.open.json \
      --truth artifacts/visual_truth/FloorPlan1.open.json \
      --assignment global --system model-a-openprop-global \
      --output artifacts/visual_case_results/model-a/FloorPlan1.open.global.jsonl

Per-case outputs are combined only through the paired-population gate. Query
and expected-property populations must match exactly across required systems;
false positives and association misses remain model-specific records:

    PYTHONPATH=src python scripts/combine_visual_results.py \
      --input \
        artifacts/visual_case_results/model-a/FloorPlan1.open.independent.jsonl \
        artifacts/visual_case_results/model-a/FloorPlan1.open.global.jsonl \
      --require-systems model-a-openprop-independent model-a-openprop-global \
      --output artifacts/visual_results_raw.jsonl

Primary system differences are then estimated with paired cluster resampling,
using scene for simulator data and room/person for real video. Query Top-1 uses
an exact McNemar test, MRR uses an exact paired sign test, and all reported
deltas receive deterministic percentile cluster-bootstrap intervals. Association
inference is permitted only when both systems share exactly the same detection
population (for example independent versus global assignment on one captured
VLM response). Cross-model comparisons use `--query-only` rather than pretending
model-specific detections are paired:

    PYTHONPATH=src python scripts/compare_visual_systems.py \
      --input artifacts/visual_results_raw.jsonl \
      --baseline model-a-independent --system model-a-global \
      --split test --output artifacts/visual_pairwise/model-a-global.json

Controller initialization failures are also materialized before the capture
command re-raises the error. A zero-record failure therefore remains visible to
batch accounting instead of disappearing from the denominator. The verified
local failure artifact is
`artifacts/ai2thor_initialization_failure/FloorPlan1.capture-manifest.json`.

On the current Windows host, AI2-THOR 5.0.0 has no Unity build for the resolved
commit. Under WSL2, CloudRendering selected Mesa llvmpipe and crashed during
Vulkan physical-device initialization; Linux64 under WSLg also selected
llvmpipe, ran at roughly 0.02 FPS, and timed out during the 100-second
`Initialize` handshake. Forcing the Mesa D3D12 driver did not produce a usable
Unity process. These are environment feasibility results, not VLM or OpenProp
performance results. The next valid Tier-A evidence must come from a verified
schema-v2 bundle captured on a supported GPU-Linux renderer.

Frozen result aggregation and artifact generation:

The recommended entry point freezes all three calibration stages in the
required order and writes one SHA-bound audit containing every policy and
calibration population:

    python scripts/calibrate_visual_pipeline.py \
      --input artifacts/visual_results_raw.jsonl \
      --system openprop-global \
      --output artifacts/visual_results.jsonl \
      --policy-output artifacts/visual_calibration_pipeline.json

The lower-level equivalents remain available for component ablations:

    python scripts/calibrate_visual_acceptance.py \
      --input artifacts/visual_results_raw.jsonl --system openprop-global \
      --policy-output artifacts/visual_policy.json \
      --results-output artifacts/visual_results_association_calibrated.jsonl
    python scripts/calibrate_visual_combined_confidence.py \
      --input artifacts/visual_results_association_calibrated.jsonl \
      --system openprop-global \
      --policy-output artifacts/visual_confidence_policy.json \
      --results-output artifacts/visual_results_confidence_calibrated.jsonl
    python scripts/calibrate_visual_query_acceptance.py \
      --input artifacts/visual_results_confidence_calibrated.jsonl \
      --system openprop-global \
      --policy-output artifacts/visual_query_policy.json \
      --results-output artifacts/visual_results.jsonl
    python scripts/evaluate_visual_results.py \
      --input artifacts/visual_results.jsonl --split test \
      --output artifacts/visual_evaluation_test.json
    python scripts/build_visual_experiment_artifacts.py \
      --report artifacts/visual_evaluation_test.json \
      --output-dir artifacts/visual_paper

The artifact builder now emits the main Markdown/LaTeX result table and
four-panel grounding plot plus a separate raw-versus-calibrated combined
confidence table and reliability figure. Source and candidate-count slices are
retained in the frozen report even when a paper figure omits them for space.

These commands establish evaluation infrastructure only. They do not become
paper evidence until the input JSONL is bound to frozen captured responses and
an untouched split.

