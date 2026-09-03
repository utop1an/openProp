#!/bin/bash
#SBATCH --job-name=openprop-thor
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log

set -euo pipefail

OPENPROP_BASE="${OPENPROP_BASE:-$HOME/openProp}"
OPENPROP_SIF="${OPENPROP_SIF:-$HOME/openprop-ai2thor-v3.sif}"
OPENPROP_CACHE="${OPENPROP_CACHE:-$HOME/openprop-cache}"
OPENPROP_OUTPUT="${OPENPROP_OUTPUT:-$OPENPROP_BASE/artifacts/ai2thor_hpc}"
OPENPROP_PREPARED="$OPENPROP_OUTPUT/prepared"
SCENE="${SCENE:-FloorPlan1}"
FAMILIES="${FAMILIES:-open toggle dirty fill}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_RUNTIME=singularity
else
    echo "Neither apptainer nor singularity is available" >&2
    exit 2
fi

mkdir -p logs "$OPENPROP_CACHE/ai2thor" "$OPENPROP_CACHE/python" "$OPENPROP_OUTPUT"

HOST_NVIDIA_ICD=""
for ICD_FILE in /usr/share/vulkan/icd.d/nvidia_icd.json \
    /etc/vulkan/icd.d/nvidia_icd.json
do
    if [ -s "$ICD_FILE" ]; then
        HOST_NVIDIA_ICD="$ICD_FILE"
        break
    fi
done
if [ -z "$HOST_NVIDIA_ICD" ]; then
    echo "Host $(hostname) has no NVIDIA Vulkan ICD; select a Vulkan-capable node" >&2
    exit 3
fi
if ! ldconfig -p 2>/dev/null | grep -q 'libGLX_nvidia.so.0'; then
    echo "Host $(hostname) has no NVIDIA Vulkan userspace library" >&2
    exit 3
fi

export SINGULARITYENV_VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export APPTAINERENV_VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

BIND_ARGS=(
    -B "$OPENPROP_BASE:/work"
    -B "$OPENPROP_CACHE/ai2thor:$HOME/.ai2thor"
    -B "$OPENPROP_CACHE/python:$HOME/.cache"
    -B "$OPENPROP_OUTPUT:/output"
    -B "$HOST_NVIDIA_ICD:/usr/share/vulkan/icd.d/nvidia_icd.json:ro"
)

echo "host=$(hostname) scene=$SCENE gpu=${CUDA_VISIBLE_DEVICES:-unset} icd=$HOST_NVIDIA_ICD"
nvidia-smi

if [ "$RUN_PREFLIGHT" = "1" ]; then
    "$CONTAINER_RUNTIME" exec --nv "${BIND_ARGS[@]}" "$OPENPROP_SIF" \
        bash -lc 'cd /work && python -u hpc/preflight_ai2thor.py'
fi

read -r -a FAMILY_ARGS <<< "$FAMILIES"
"$CONTAINER_RUNTIME" exec --nv "${BIND_ARGS[@]}" "$OPENPROP_SIF" \
    bash -lc "cd /work && python -u scripts/capture_ai2thor_pilot.py \
      --scene '$SCENE' --platform cloud --families ${FAMILY_ARGS[*]} \
      --output-dir /output"

"$CONTAINER_RUNTIME" exec "${BIND_ARGS[@]}" "$OPENPROP_SIF" \
    bash -lc "cd /work && python -u scripts/verify_ai2thor_capture.py \
      '/output/$SCENE.capture-manifest.json' \
      --report '/output/$SCENE.verification-report.json'"

"$CONTAINER_RUNTIME" exec "${BIND_ARGS[@]}" "$OPENPROP_SIF" \
    bash -lc "cd /work && python -u scripts/prepare_ai2thor_capture.py \
      '/output/$SCENE.capture-manifest.json' \
      --output-dir '/output/prepared/$SCENE'"

echo "capture complete: $OPENPROP_OUTPUT/$SCENE.capture-manifest.json"
echo "prepared inputs: $OPENPROP_PREPARED/$SCENE/inputs"
echo "evaluation truth: $OPENPROP_PREPARED/$SCENE/truth"
