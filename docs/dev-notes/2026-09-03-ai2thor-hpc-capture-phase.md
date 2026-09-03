# AI2-THOR HPC capture and preparation phase

Date: 2026-09-03

## Context

This phase established a portable Slurm execution path for the first real
AI2-THOR visual captures. It is infrastructure and mechanism evidence, not VLM
or end-to-end OpenProp performance evidence.

The HPC checkout is a Git clone at `~/openProp`. The immutable SIF is kept
outside the repository and the checkout is bind-mounted to `/work`, so source
updates use Git and do not require rebuilding the capture image.

## Container and cluster findings

- `openprop-ai2thor-v3.sif` is an Ubuntu 24.04 image with Python 3.12.3,
  AI2-THOR 5.0.0, Vulkan tools, and the required in-image NVIDIA ICD/EGL bind
  destinations for SingularityCE 3.5.
- The transferred SIF is 294,887,424 bytes with SHA-256
  `65224f5115e20f0bd95cc3dfd665178559350e973b0c88a7d3600bdf56c884ab`.
- `gpusrv-5` exposed CUDA/NVML but no NVIDIA Vulkan ICD or matching graphics
  userspace library. Job 136462 is the retained negative control.
- `mlcv2` exposed a usable NVIDIA Vulkan device. Job 136487 passed the
  preflight and one-family capture in 14 seconds.
- The capture launcher checks both the host ICD and
  `libGLX_nvidia.so.0`, binds the ICD to the path present in the SIF, and
  selects no Mesa software fallback.

## Four-family capture

Job 136541 ran on `mlcv2` and captured all four currently implemented state
families:

| Family | Intended target | Intended transition |
| --- | --- | --- |
| open | Cabinet | closed to open |
| toggle | CoffeeMachine | off to on |
| dirty | Bowl | clean to dirty |
| fill | Bottle | empty to filled |

The run produced four records and 24 content-addressed image, metadata, and
box artifacts. The capture manifest SHA-256 is
`a49fc8fceef944c4e2e6ef527840002e771a5d393a6be7512672b8c7c81e895f`.
Raw metadata and target identity remain evaluation-only and are not serialized
into matcher inputs.

## Preparation failure and repair

The first CPU preparation attempt, job 136577, failed after writing two partial
episodes:

```text
ValueError: missing instance detection for CounterTop|-00.08|+01.15|00.00
```

The captured data showed that AI2-THOR metadata visibility and rendered
instance detections are different predicates. In the dirty frames, 10 objects
were metadata-visible but two CounterTop objects lacked boxes. In the fill
frames, eight were visible but one CounterTop lacked a box. Detection output
also contained scene geometry that metadata did not mark visible.

The repaired default oracle-box candidate set is therefore:

```text
metadata-visible entities intersect valid instance-detection IDs
```

Unanchored visible objects stay in evaluation truth and count as missing
candidate evidence. They are not converted into negative observations and do
not receive fabricated boxes. Explicitly supplied candidate IDs remain
strictly validated. Preparation now reports per-frame visible counts, anchored
counts, coverage, and omitted IDs.

The reusable recovery entry point is:

```bash
SCENE=FloorPlan1 sbatch hpc/ai2thor_prepare.slurm
```

Only `preparation-report.json` is a completion marker; individual files can
remain after an interrupted attempt.

## Verified preparation result

Job 136580 completed successfully with exit code 0 in three seconds and
verified all 24 source artifacts. It produced all four input/truth episode
pairs with `truth_exposed_to_matcher=false`.

| Family | Before coverage | After coverage | Intended target anchored in both frames |
| --- | ---: | ---: | --- |
| open | 1.000 | 1.000 | yes |
| toggle | 1.000 | 1.000 | yes |
| dirty | 0.800 | 0.800 | yes |
| fill | 0.875 | 0.875 | yes |

Every transition also contains a DishSponge position and motion-state change.
The sponge falls between the before frame and the intervention step, so the
pilot mixes the intended property change with incidental physics settling.
This is useful as a multi-entity observation stress case but is not a clean
single-target causal episode.

## Validation and revisions

- The adapter repair passed the full test suite: 523 tests.
- The CPU preparation Slurm asset passed four focused HPC tests and Bash syntax
  validation.
- The refreshed reproducibility-manifest tests passed 6/6.
- Relevant revisions, in order, are `3e19f07`, `c72e4b1`, `3eada8b`,
  and `42c26f0`.

## Experiment readiness at phase close

- G0 infrastructure: passed on `mlcv2`.
- G1 truth isolation: passed for the four-family FloorPlan1 pilot, but the full
  eight-family gate is not met.
- G2 candidate/tracking: not measured on the frozen calibration split.
- G3 real VLM viability: not run; no captured real-VLM response exists.
- G4 calibration, G5 untouched simulator test, and G6 real confirmation: not
  started.

At the phase-close audit, the HPC login environment did not expose an API key.
ADT and VISOR were marked ready in the local acquisition status, but their
completion markers and data were not present on the HPC. Do not infer HPC
dataset readiness from the local status artifact.

## Follow-ups

1. Add a pre-capture physics-settling gate and separate intended intervention
   labels from incidental observed changes.
2. Implement and smoke-test move/receptacle, cook, slice, and break, then
   multi-object actions, camera changes, and partial occlusion.
3. Generate a deterministic development episode manifest and Slurm array.
4. Add captured-response LLM/VLM inference entry points; freeze provider,
   model revision, prompt/schema, and request settings before calibration.
5. Run development, then calibration, and only then the untouched test split.
6. Transfer ADT/VISOR completion-marker-bound subsets and collect the
   room/person-disjoint custom real-video confirmation set.
