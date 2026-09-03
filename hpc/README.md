# OpenProp AI2-THOR on a Slurm HPC

This runtime follows the bind-mounted-repository pattern used by
`llm-action-extraction/hpc/apptainer.ranker.exp`, but deliberately excludes
Ollama. AI2-THOR uses the allocated GPU for Vulkan rendering; captured frames
can later be sent to a frozen API VLM or copied to a separate inference job.
All Python application and transitive dependency versions are exactly pinned in
`hpc/requirements-ai2thor.txt`; the realized environment is also shipped as
`dist/openprop-ai2thor-v3.packages.txt`. Version 3 uses a small Ubuntu 24.04
base, an isolated Python 3.12 environment, and in-image NVIDIA ICD/EGL bind
destinations required by the cluster's SingularityCE 3.5 runtime.

The complete literature-grounded execution and evaluation plan is in
[`docs/visual-hpc-experiment-design.md`](../docs/visual-hpc-experiment-design.md).
The experiment does require a text LLM for query parsing and a VLM for visual
updates. Run them as inference jobs over immutable captures; do not colocate a
local multimodal model and Unity on the same GPU. This SIF already contains the
OpenAI client for the API lane. An open-weight model-server SIF is intentionally
deferred until its exact model, serving runtime, quantization, and weight digest
are frozen; those choices materially determine CUDA and VRAM requirements.

## Build

Build once where Apptainer/Singularity image builds and downloads are allowed:

```bash
cd "$HOME/openProp"
apptainer build "$HOME/openprop-ai2thor-v3.sif" hpc/openprop-ai2thor.def
apptainer test "$HOME/openprop-ai2thor-v3.sif"
```

If unprivileged builds are disabled, use the site's remote/fakeroot service or
build elsewhere and copy the immutable SIF. Do not build on a GPU compute node
unless site policy permits it.

## Pilot

Edit only site-specific partition, account and notification directives, then:

```bash
cd "$HOME/openProp"
mkdir -p logs
SCENE=FloorPlan1 RUN_PREFLIGHT=1 sbatch hpc/ai2thor_capture.slurm
```

The capture job verifies the content-addressed bundle and then prepares
physically separated VLM inputs and evaluation-only truth. If capture already
succeeded but preparation must be retried after a code or schema fix, do not
allocate another GPU or recapture the scene:

```bash
cd "$HOME/openProp"
SCENE=FloorPlan1 sbatch hpc/ai2thor_prepare.slurm
```

Treat `prepared/FloorPlan1/preparation-report.json` as the completion marker;
individual input or truth files can exist after an interrupted failed attempt.

## Prepared local transfer bundle

The local build workflow writes four content-addressed files under `dist/`:

- `openprop-ai2thor-v3.sif`;
- `ai2thor-cloudrendering-f0825767-cache.tar.gz`;
- `openprop-ai2thor-v3.packages.txt`;
- `openprop-hpc-source.tar.gz`.

Verify them against `dist/HPC_TRANSFER_MANIFEST.json` after upload:

```bash
python scripts/build_hpc_transfer_manifest.py --directory dist --check
export OPENPROP_CACHE="${OPENPROP_CACHE:-$HOME/openprop-cache}"
mkdir -p "$OPENPROP_CACHE/ai2thor"
tar -xzf dist/ai2thor-cloudrendering-f0825767-cache.tar.gz \
  -C "$OPENPROP_CACHE/ai2thor"
```

The cache archive already contains the `releases/...` prefix expected below the
directory bound to the container user's `$HOME/.ai2thor`. The SIF was built
under WSL2 Linux, but the mandatory final
runtime test remains the Slurm GPU preflight because the target NVIDIA driver
and Vulkan ICD cannot be validated locally.

For a fresh upload, extract the source first and use the guarded installer:

```bash
mkdir -p "$HOME/openProp"
tar -xzf /path/to/uploaded/dist/openprop-hpc-source.tar.gz -C "$HOME/openProp"
bash "$HOME/openProp/hpc/install_uploaded_bundle.sh" \
  /path/to/uploaded/dist
```

The installer verifies every SHA-256 receipt, refuses to replace a different
existing SIF, installs the offline cache, and checks that the frozen AI2-THOR
dataset protocols are present.

The first controller start downloads roughly 500 MB into the persistent
`OPENPROP_CACHE/ai2thor` bind. If compute nodes have no outbound network,
pre-warm this directory in an interactive GPU job with permitted network or
stage a verified cache before the batch run.

## Initial resource request

| Resource | Pilot | Full capture starting point | Reason |
| --- | ---: | ---: | --- |
| NVIDIA GPU | 1 | 1 per array task | one Unity controller per task; Vulkan matters more than tensor throughput |
| GPU memory | >=8 GB | >=8 GB | 640x480 RGB plus instance segmentation is modest |
| CPU | 8 cores | 4-8 cores | Unity simulation, PNG and JSON serialization |
| RAM | 32 GB | 24-32 GB | safe headroom for Unity and Python |
| Local scratch | 20 GB | 50-100 GB per task/batch | SIF/cache plus images, metadata, boxes and logs |
| Walltime | 2 h | size from pilot | depends on supported actions and filesystem throughput |

An A100 is unnecessary for capture. A V100, A40, A100, L40, or similar NVIDIA
GPU is suitable only when the node exposes a functioning Vulkan driver/ICD via
`apptainer/singularity exec --nv`. `nvidia-smi` alone is not sufficient: preflight also
requires `vulkaninfo --summary` and a real CloudRendering controller.

Do not launch calibration or test scenes until the development pilot passes,
the remaining interventions are implemented, and the VLM request manifest is
frozen. Keep API keys outside the SIF and job script.

NCI Gadi uses PBS rather than this Slurm template. Translate the directives and
declare every required `/scratch` or `/g/data` filesystem in the PBS storage
request.

The image intentionally installs the Vulkan loader but not Mesa Vulkan drivers:
software `llvmpipe` must not silently compete with the host NVIDIA ICD.
The Slurm launcher binds only the host NVIDIA ICD and fails before container
startup when a selected node lacks `nvidia_icd.json` or `libGLX_nvidia.so.0`.
On the audited `cluster1` snapshot, `gpusrv-5` lacked both and is unsuitable for
AI2-THOR CloudRendering until the administrator installs the matching NVIDIA
graphics/Vulkan userspace components; `mlcv2` exposed them and is the pilot
node used to validate the version-3 image.

The first verified cluster runtime was Slurm job `136487` on `mlcv2` on
2026-09-03. It completed in 14 seconds: the container's `vulkaninfo --summary`
returned zero and selected the NVIDIA RTX 2080 Ti, the AI2-THOR 5.0.0
CloudRendering controller created a 320x240 RGB frame with nine instance boxes,
and the one-family `open` capture completed 1/1 with six content-addressed
artifacts. This is environment and mechanism validation, not performance
evidence. Failed job `136462` on `gpusrv-5` remains the recorded negative
control for a CUDA-visible node without a host Vulkan ICD.
