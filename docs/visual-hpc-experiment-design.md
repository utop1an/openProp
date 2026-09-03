# Literature-grounded visual experiment design on Slurm

Date: 2026-09-02  
Target: ICLR 2027  
Status: preregistration and execution plan; this document reports no VLM result.

## 1. Decision

The visual study uses one immutable OpenProp SIF but two disjoint Slurm job
lanes:

1. **capture jobs** run AI2-THOR with one NVIDIA/Vulkan GPU and write only
   content-addressed frames, candidate regions, action receipts, and separately
   held truth;
2. **inference jobs** run the frozen language parser and VLM over the immutable
   VLM-visible inputs, save raw responses, and stop;
3. **replay/evaluation jobs** are CPU-only and compare every OpenProp baseline
   and ablation from the same saved parser/VLM responses.

The language model is required for the semantic query parser. The VLM is
required for visual property proposals. Neither model is allowed to rank final
entities directly in the main OpenProp system: typed parsing and visual
observation are frozen first, while final scoring remains deterministic. This
separation lets a model change be measured as a perception/input change instead
of silently changing the decision rule.

Unity capture and a local multimodal model must not share a GPU. API inference
can use a CPU node with outbound HTTPS; local-model inference uses a second GPU
job and a persistent model cache. The current `openprop-ai2thor.sif` already
contains Python, OpenProp, AI2-THOR, and the OpenAI client, so it supports capture,
API inference, and CPU replay. A second model-server SIF is needed only for the
open-weight VLM arm; it must be pinned after the exact model/runtime is selected.

## 2. What the literature changes in the protocol

| Evidence | Experimental consequence for OpenProp |
| --- | --- |
| Scene Graph Memory models a changing, partially observed scene rather than a static map. | Include delayed and target-absent queries; do not equate an unobserved object with a negative property. |
| DynaMem updates memory as objects move, appear, and disappear and evaluates real dynamic manipulation. | Compare against static/latest-observation memory under identical candidates and observations; include appear/disappear/move events and a real-video confirmation. |
| 3D-Mem shows the value of compact multi-view memory and active retrieval. | Include a current-frame VLM and a multi-frame/history baseline; report memory cost and calls, not accuracy alone. |
| Embodied VideoAgent couples persistent object identity with VLM state updates. | Decompose re-identification from property detection and test global multi-object assignment explicitly. |
| Ego4D Hands & Objects labels pre/PNR/post state changes and changed-object boxes. | Use it for real state-change detection and changed-object association, not as full current-state query truth. |
| Aria Digital Twin provides synchronized real RGB, object trajectories, boxes, masks, and 6DoF ground truth. | Use it for candidate recall, occlusion, re-identification, and ID-switch evaluation. |
| EPIC-KITCHENS VISOR tracks active objects through transformative interactions. | Use it for appearance shift and long/short-term candidate consistency; do not infer unlabeled current properties. |
| HOTA separates detection and association quality. | Report candidate detection and identity association separately in addition to an aggregate tracking score. |
| VLM hallucination and neural calibration studies show raw verbal confidence is not a probability. | Retain nonexistent/target-absent controls; calibrate on calibration scenes only; report Brier, NLL, ECE, and risk-coverage. |

OpenProp is not claiming a new mapper, object detector, tracker, VLM, or general
scene-memory architecture. Its claim is narrower: given supplied candidates and
typed observations of unequal reliability and age, conservative association and
property-specific persistence improve current-entity grounding.

## 3. Questions and falsifiable hypotheses

The primary unit is a delayed natural-language query over a scene history with
at least two candidate entities.

- **H1, end-to-end utility:** `openprop_learned_global` improves all-case Top-1
  over current-frame VLM, direct VLM update, latest observation, and no-decay
  memory on untouched AI2-THOR test scenes.
- **H2, update safety:** global one-to-one assignment reduces false entity
  updates relative to independent association when at least two objects change
  or at least three same-type distractors are present.
- **H3, temporal value:** the full method loses less Top-1 accuracy as the
  observation-query delay grows, without increasing stale-positive errors.
- **H4, selective reliability:** calibrated null/margin rules reduce false
  answers on absent or unidentifiable targets at a predeclared coverage floor.
- **H5, transfer:** the direction of H2 and the delayed-query gain in H1 holds
  on room/person-disjoint custom real video. Public datasets support only their
  annotated perception or tracking endpoints.

Failure to improve Top-1 is scientifically informative if oracle decomposition
shows that VLM property recall or candidate recall is the limiting stage. A
simulator-only improvement is mechanism validation, not real-world evidence.

## 4. Data and split design

### 4.1 AI2-THOR causal intervention benchmark

Keep the frozen 72/24/24 development/calibration/test scene split. Construct a
balanced, deterministic fractional factorial rather than the full Cartesian
product:

| Axis | Levels |
| --- | --- |
| changed objects | 0, 1, 2, 3 |
| same-type distractors | 0, 1, 3, 7 |
| target visibility | visible, partially occluded, absent |
| camera | fixed, translated, rotated |
| history gap | 0, 300, 3,600, 86,400 seconds |
| change family | move/receptacle, open, toggle, dirty, fill, cook, slice, break |
| query wording | canonical, paraphrase A, paraphrase B |
| candidates | oracle boxes, detected/tracked boxes |

For every scene and seed, require at least four successful episodes after
construction. Retain attempted actions and failed simulator actions in a
construction report, but never turn a failed action into a positive example.
Zero-change, wrong-object, target-absent, and ambiguous-query controls are
mandatory. Each episode contains pre-action, immediate post-action, and delayed
query frames. Simulator `current_truth` remains physically isolated from VLM
inputs and from the matcher.

Five fixed seeds are used for episode construction. Query paraphrases share the
same episode and are not counted as independent environmental samples.

### 4.2 ProcTHOR OOD

Sample identity-disjoint houses after the iTHOR protocol is frozen. Match the
supported property/event distribution as far as possible, but report unmatched
support explicitly. ProcTHOR tests layout and appearance transfer only; it is
still simulator evidence.

### 4.3 Custom real-video confirmation

Collect a minimum of eight test `room_person` clusters with at least six
episodes per cluster. Development, calibration, and test are disjoint by both
room and recorder. Each episode has at least two visually similar instances and
contains synchronized intervention logs plus pre/change/post/occluded/query
frames. Include tripod and handheld capture, illumination change, motion blur,
partial occlusion, and one-to-three object changes.

Identity markers may be visible to annotators but never to the VLM. Three
annotators label test identity and typed state; adjudication and the
pre-adjudication agreement statistic are recorded. The final sample size may be
increased only using blinded pilot variance before test inference.

### 4.4 Public real-data slices

- Ego4D Hands & Objects: changed-object box, PNR timing, state-change/no-change.
- Aria Digital Twin: target candidate recall, association, occlusion, ID switch,
  trajectory consistency.
- EPIC-KITCHENS VISOR: mask/box candidate quality, active-object association,
  transformative appearance and long-term consistency.
- Licensed web video: qualitative/adversarial source shift only unless rights,
  identity, and typed current-state annotations satisfy the real-video contract.

No public slice enters end-to-end Top-1 when it lacks current-state and target
identity truth.

## 5. Model protocol: yes, both LLM and VLM are part of the experiment

### 5.1 Language parser

Run one frozen text LLM parser for each unique query and cache its structured
`PropertyConstraint` response. Reuse that exact response across all visual
systems. The main paper reports:

- parser schema-valid rate;
- property-name/type/value exact match and macro F1;
- query relevance calibration where relevance is used;
- canonical/paraphrase consistency;
- oracle-parser upper bound.

Use a deterministic grammar/rule parser as a non-LLM baseline and the typed gold
constraint as an upper bound. This establishes whether an OpenProp gain depends
on privileged query parsing. Parser prompt, model revision, temperature, schema,
and raw response hash freeze before calibration/test.

### 5.2 Visual updater

Use at least two genuinely different VLM families, ideally one API model and one
open-weight model. Both receive the same ordered frame set, opaque candidate
IDs, and region anchors. They emit typed observed values plus source confidence;
they do not receive simulator IDs or truth fields.

For multiple detections, the VLM emits one observation hypothesis per visual
event/candidate region. OpenProp then scores every hypothesis-to-entity edge and
performs global one-to-one assignment with an explicit null option. It must not
copy the same high-confidence update to every red cup above a threshold. Soft
scores for non-selected candidates remain audit evidence, not ledger updates.

Raw responses are captured once per `(input_hash, model_revision,
request_settings)` and replayed for all association, persistence, and decision
ablations. Missing, malformed, duplicate, or timed-out responses remain in the
denominator as failures or abstentions.

### 5.3 Factorizing model variance

Use the following crossed analysis without paying for every possible model pair:

1. main matrix: one frozen parser × two VLM families × all systems;
2. parser robustness: second parser × one VLM × main plus oracle-parser system;
3. VLM robustness: oracle typed query × both VLMs;
4. deterministic replay: all 16 system variants over every cached response.

This distinguishes language errors, visual errors, and OpenProp algorithmic
effects. It also prevents repeated API calls from confounding ablations.

## 6. Systems, baselines, and upper bounds

Main baselines receive identical non-oracle inputs:

1. current-frame VLM directly answers the final query;
2. direct VLM updater chooses entity IDs;
3. latest accepted observation, without persistence;
4. OpenProp no decay;
5. fixed exponential decay;
6. learned property-specific persistence with global assignment.

Causal ablations remove query score, region anchors, track evidence, null,
margin, source-specific reliability, and global assignment one at a time.
Oracle candidate boxes, oracle typed properties, oracle identity, and oracle
query parsing are diagnostic upper bounds only.

External dynamic-memory systems should be reimplemented or adapted only if they
can consume the same observations/candidates and answer the same query set.
Published task-success numbers from DynaMem, 3D-Mem, or Embodied VideoAgent are
not comparable baselines.

## 7. Metrics and denominators

| Stage | Primary diagnostics |
| --- | --- |
| query parser | schema-valid %, typed exact/macro F1, relevance Brier/ECE, paraphrase consistency |
| candidates/tracking | target recall, precision, candidates/frame, IoU, HOTA components, ID switches, target-missing rate |
| VLM property | event precision/recall/macro F1, typed-value exact match, PNR error, hallucinated/duplicate event rate |
| association | correct update/all detections, false update/all detections, null FPR, collision rate, Brier, NLL, ECE |
| memory | current-property macro F1, stale-positive/negative rate, support coverage, Brier/NLL/ECE by delay |
| final query | **all-case Top-1**, MRR, Top-k, no-target FPR, coverage, selective accuracy, latency, calls, token/GPU cost |

`unknown` is missing evidence. It is not scored as a negative property. Every
table shows numerator/denominator and includes failures and abstentions unless a
selective metric explicitly conditions on answered cases.

## 8. Statistical analysis

- Freeze models, prompts, candidate settings, calibration maps, null weights,
  margins, and persistence parameters before untouched test inference.
- Pair systems by `(cluster_id, record_id)` and verify exact population equality.
- Cluster bootstrap by scene for AI2-THOR, house for ProcTHOR, room/person for
  custom video, and participant/video or sequence for public data.
- Report pooled point estimates, five seed-level points, and 95% clustered
  intervals. Do not average ratios with different denominators.
- For the four primary Top-1 comparisons, use shared resamples and
  max-studentized simultaneous intervals; use exact McNemar tests with Holm
  adjustment. Treat MRR and slice analyses as secondary.
- Reliability diagrams, Brier, and NLL are test-time evaluation only;
  temperature/isotonic calibration is fitted on calibration data.
- Report all prespecified slices, including null, malformed, low visibility,
  multi-change, crowded candidates, camera shift, source, property, and delay.

The design is successful only if the effect is both statistically compatible
with improvement and practically useful. Before test inference, set the minimum
effect of interest for Top-1, the maximum allowed false-update increase, and the
minimum coverage. A p-value alone is insufficient.

## 9. Slurm phases and resource envelope

These are starting requests to refine from development pilots, not universal
hardware requirements.

| Phase | Slurm shape | Starting resources | Output |
| --- | --- | --- | --- |
| SIF/preflight | interactive GPU | 1 GPU, 4 CPU, 16 GB, 30 min | NVIDIA Vulkan + real controller receipt |
| AI2-THOR pilot | one scene/job | 1 GPU with >=8 GB, 8 CPU, 32 GB, 2 h, 20 GB scratch | complete capture bundle |
| AI2-THOR full | scene/seed array | 1 GPU, 4-8 CPU, 24-32 GB; walltime from pilot | immutable development/calibration/test inputs |
| API LLM/VLM | sharded CPU array | 4 CPU, 16 GB, outbound HTTPS; no GPU | hashed raw model responses |
| open VLM 7B-13B | one model server/GPU | 1 GPU, normally 24-48 GB VRAM, 8 CPU, 64 GB RAM | hashed raw responses and model digest |
| larger open VLM | model dependent | 1-2 80 GB GPUs or validated quantization; benchmark first | robustness responses |
| replay/evaluation | CPU array | 8-16 CPU, 32-64 GB, no GPU | per-case JSONL, tables, plots |

Do not place API keys or model weights inside the SIF. Bind persistent directories
for AI2-THOR builds, Hugging Face/Ollama model weights, response cache, and job
scratch. Use one isolated model-server port per Slurm job. Record `sif_sha256`,
git/source snapshot, model weight digest, CUDA/driver, prompt hash, input hash,
seed, Slurm job ID, and command line in every run manifest.

Storage should be budgeted from a 20-episode pilot. Record raw frame bytes per
episode and multiply by the frozen episode count plus 30% headroom; do not guess
the final image budget from scene count alone.

## 10. Gates and stopping rules

1. **G0 infrastructure:** `nvidia-smi`, `vulkaninfo`, and a real CloudRendering
   controller pass inside the SIF. Otherwise stop all capture arrays.
2. **G1 truth isolation:** bundle verification proves that no current truth or
   simulator identity enters VLM inputs. All eight action families have at least
   one valid development episode before expanding the matrix.
3. **G2 candidates:** calibration target recall is at least 0.90 and identity
   switch rate at most 0.05. If not, improve candidates/tracking before studying
   the matcher.
4. **G3 model viability:** at least one frozen VLM has nontrivial typed
   state-change recall; malformed and duplicate rates are reported. If both VLMs
   fail, the project has established a perception bottleneck, not an OpenProp
   failure.
5. **G4 calibration:** null/margin/source-confidence policy is frozen using only
   calibration. Unsupported source/property cells use the global fallback.
6. **G5 untouched simulator:** run test once. Continue to real confirmation only
   if H1 or H2 meets its frozen practical criterion without violating safety.
7. **G6 real confirmation:** retain the effect direction on room/person-disjoint
   custom video. If this fails, restrict claims to simulator mechanism evidence.

## 11. Paper outputs

### Tables

1. end-to-end Top-1/MRR/false-update/coverage across iTHOR, ProcTHOR, and custom video;
2. query-parser, candidate, VLM-property, association, and memory decomposition;
3. performance by delay and visibility;
4. component ablation and oracle gap;
5. simulator-to-real calibration/sample-efficiency;
6. safety/failure slices with explicit denominators.

### Plots

1. Top-1 and stale error versus delay;
2. false-update versus distractor and simultaneous-change count;
3. identity/state reliability diagrams;
4. selective risk-coverage curves;
5. candidate recall versus set size;
6. calibration sample-efficiency;
7. per-property/source forest plot with clustered intervals;
8. typed-value confusion matrices and representative audit panels.

Every figure is generated from content-addressed JSONL in a deterministic
`--check` path. Qualitative panels are sampled by frozen rules, not hand-picked
after reading test results.

## 12. Immediate execution order

1. build and hash `openprop-ai2thor.sif` on the Slurm site;
2. run `hpc/preflight_ai2thor.py` and a `FloorPlan1` capture pilot;
3. implement and development-test the four pending state actions, multi-object
   actions, camera shifts, and occlusion;
4. measure episode runtime/storage, then freeze array size and walltime;
5. freeze one text parser and two VLM revisions and create hashed request
   manifests;
6. run development, then calibration; fit all thresholds/calibration maps;
7. run untouched iTHOR/ProcTHOR test once and render the complete table/plot set;
8. collect/prepare custom real video and download only the annotated public
   slices required for decomposition.

The current executable protocol is
`artifacts/visual_protocol/experiment_protocol.json`, version
`openprop-iclr2027-visual-experiment-v2`, digest
`d401dbd85b75311bc422b4fee07841cb887e364c41cda7fc3815bdaf7f6f7be7`.
Any change to factors,
systems, comparisons, or gates requires a versioned protocol revision before
test inference.
