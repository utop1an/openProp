# Visual dataset acquisition and transfer checklist

Date: 2026-09-03

The machine-readable registry is `data/visual/registry.json`. Initialize local
staging directories and emit a fail-visible status report with:

```bash
PYTHONPATH=src python scripts/prepare_visual_datasets.py --initialize \
  --require-ready ai2thor_ithor
```

The command never downloads licensed video and never records credentials. A
dataset is marked complete only by a `.openprop-dataset-complete.json` file that
enumerates byte counts and SHA-256 hashes. Use
`data/visual/completion-marker.example.json` as the template.

## Required for the first HPC experiment

The first development pilot needs only:

- `openprop-ai2thor.sif` and its SHA-256 receipt;
- the staged Linux AI2-THOR build cache, if compute nodes have no internet;
- `artifacts/ai2thor_protocol/scene_split.json`;
- `artifacts/visual_protocol/experiment_protocol.json`;
- the repository source and `hpc/ai2thor_capture.slurm`.

AI2-THOR 5.0.0 downloads the platform-specific Unity build separately from the
Python wheel. The cache is not called an experimental dataset: it is runtime
material. Extract the staged archive into `$OPENPROP_CACHE/ai2thor`; the Slurm
script binds that directory to the container user's `$HOME/.ai2thor` before an
offline compute job starts.

## Ego4D Hands & Objects

1. Accept the Ego4D license and obtain the temporary AWS credentials.
2. Install the official CLI in a data-transfer environment: `pip install ego4d`.
3. Download annotations first, inspect the FHO video IDs, then download selected
   clips only:

```bash
export EGO4D_AWS_PROFILE=ego4d
ego4d --output_directory "$DATA_ROOT/ego4d_hands_objects" \
  --datasets annotations --benchmarks FHO --version v2 \
  --aws_profile_name "$EGO4D_AWS_PROFILE" -y

ego4d --output_directory "$DATA_ROOT/ego4d_hands_objects" \
  --datasets clips --benchmarks FHO --version v2 \
  --video_uid_file selected_fho_video_uids.txt \
  --aws_profile_name "$EGO4D_AWS_PROFILE" -y
```

Do not download the full-scale release: the official documentation describes
the full videos as multi-terabyte data. Ego4D supports state-change perception
and changed-object association only, not OpenProp end-to-end current truth.

## Aria Digital Twin

1. Accept the ADT license and use Dataset Explorer to export a CDN JSON file.
2. Set `ADT_CDN_FILE` to the downloaded file; links expire, so never commit it.
3. Install `projectaria-tools[all]` in a separate environment and start with the
   approximately 500 MB official sample:

```bash
aria_dataset_downloader -c "$ADT_CDN_FILE" \
  -o "$DATA_ROOT/aria_digital_twin" \
  -l Apartment_release_golden_skeleton_seq100_10s_sample_M1292 \
  -d 0 1 2 3 4 5 6 7 8 9
```

For the experiment, select sequences with dynamic objects and initially request
RGB/VRS plus the smallest ground-truth package that includes timestamps,
instances, 2D boxes, and object trajectories. Do not download the approximately
3.5 TB full release before the sample adapter passes.

### Frozen OpenProp ADT pilot

The current pilot deliberately omits the large main VRS, depth, synthetic, and
MPS packages. It contains all 63 Apartment `clean`, `decoration`, and `meal`
sequences with main ground truth and RGB preview, plus an outcome-blind subset
of 18 segmentations (six per activity). The fixed sequence-level split is 40
train, 12 calibration, and 11 test. This cohort is not subject-disjoint, so it
must not be reported as subject-generalization evidence.

The credential-free frozen inputs and audit outputs are:

- `artifacts/adt_groundtruth_screening_plan.json`
- `artifacts/adt_sequence_screening.json` (descriptive analysis only)
- `artifacts/adt_pilot_selection.json`
- `artifacts/adt_pilot_download_audit.json`
- `artifacts/visual_dataset_status.json`

Reproduce the acquisition with `scripts/download_adt_screening.py`,
`scripts/rank_adt_sequences.py`, `scripts/build_adt_pilot_selection.py`, and
`scripts/download_adt_pilot_visuals.py`. Then run
`scripts/audit_adt_pilot_download.py --write-completion-marker` and pass the
parent of the `aria_digital_twin` directory to
`scripts/prepare_visual_datasets.py --data-root`. The completion marker binds
every downloaded file by byte count and SHA-256. Neither the ranking report nor
dataset readiness is performance evidence.

## EPIC-KITCHENS VISOR

Download the official VISOR annotation and sparse-frame release from the
University of Bristol DOI linked in the registry. Start with masks, relations,
and sparse frames. Only retrieve original EPIC-KITCHENS videos for selected
participant/video IDs after the adapter has identified the necessary clips.

VISOR is used for active-object candidate generation, tracking through
occlusion, and transformative appearance. It does not supply complete typed
current-state truth for the OpenProp end-to-end endpoint.

### Frozen OpenProp VISOR pilot

The current pilot contains all 115 train and 43 validation sparse annotation
JSON files, the official noun map, frame map, and README, plus sparse RGB ZIPs
for 44 selected videos. It deliberately omits dense interpolations, official
test images, full RGB-frame releases, and original EPIC-KITCHENS videos. The
selected media totals 5,468,340,556 bytes and contains 15,124 sparse RGB frames.

Selection applies the same minimum annotation-coverage rule before any model
run and then uses a fixed hash seed. The frozen roles are 24 train videos from
13 participants, 8 calibration videos from 4 participants, and 12 test videos
from 6 participants; participant IDs are disjoint across all three roles.
VISOR mask IDs must not be treated as persistent OpenProp entity IDs. This
pilot supports active-object candidate recall, hand-object association, and
appearance-shift analysis, while ADT remains the source for moved-object
identity and occlusion tracking.

The credential-free acquisition artifacts are:

- `artifacts/visor_screening_download.json`
- `artifacts/visor_pilot_selection.json`
- `artifacts/visor_pilot_media_plan.json`
- `artifacts/visor_pilot_download_audit.json`
- `artifacts/visual_dataset_status.json`

Use `scripts/download_visor_screening.py`,
`scripts/build_visor_pilot_selection.py`, and
`scripts/download_visor_pilot_media.py` to reproduce the subset. The media
downloader probes the complete frozen population before transfer, enforces an
explicit byte cap, resumes partial files, checks ZIP CRCs and member paths, and
writes a SHA-256 completion marker. Annotation screening and dataset readiness
are not performance evidence.

## Custom real video

Follow `openprop-real-video-v1` in `docs/ai2thor-vlm-feasibility.md`. Keep consent
records outside public artifacts, split by room and person, and place only
content-addressed media plus annotation manifests in the data staging root.
Three annotators and adjudication are required for the held-out confirmation
split.

## Transfer verification

After copying data to HPC, rerun `prepare_visual_datasets.py` against the HPC
data root. A missing file, changed size, changed SHA-256, path escape, missing
license acknowledgement, or wrong dataset ID fails closed. Dataset readiness
does not mean performance evidence; the status report always declares
`performance_evidence: false`.
