#!/bin/bash
#SBATCH --job-name=openprop-prepare
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.log

set -euo pipefail

OPENPROP_BASE="${OPENPROP_BASE:-$HOME/openProp}"
OPENPROP_SIF="${OPENPROP_SIF:-$HOME/openprop-ai2thor-v3.sif}"
OPENPROP_OUTPUT="${OPENPROP_OUTPUT:-$OPENPROP_BASE/artifacts/ai2thor_hpc}"
SCENE="${SCENE:-FloorPlan1}"

if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_RUNTIME=singularity
else
    echo "Neither apptainer nor singularity is available" >&2
    exit 2
fi

mkdir -p logs "$OPENPROP_OUTPUT/prepared/$SCENE"

BIND_ARGS=(
    -B "$OPENPROP_BASE:/work"
    -B "$OPENPROP_OUTPUT:/output"
)

"$CONTAINER_RUNTIME" exec "${BIND_ARGS[@]}" "$OPENPROP_SIF" \
    bash -lc "cd /work && python -u scripts/verify_ai2thor_capture.py \
      '/output/$SCENE.capture-manifest.json' \
      --report '/output/$SCENE.verification-report.json'"

"$CONTAINER_RUNTIME" exec "${BIND_ARGS[@]}" "$OPENPROP_SIF" \
    bash -lc "cd /work && python -u scripts/prepare_ai2thor_capture.py \
      '/output/$SCENE.capture-manifest.json' \
      --output-dir '/output/prepared/$SCENE'"

echo "prepared inputs: $OPENPROP_OUTPUT/prepared/$SCENE/inputs"
echo "evaluation truth: $OPENPROP_OUTPUT/prepared/$SCENE/truth"
