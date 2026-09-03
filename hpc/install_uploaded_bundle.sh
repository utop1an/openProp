#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 /absolute/path/to/uploaded/dist" >&2
    exit 2
fi

BUNDLE_DIR="$1"
OPENPROP_BASE="${OPENPROP_BASE:-$HOME/openProp}"
OPENPROP_CACHE="${OPENPROP_CACHE:-$HOME/openprop-cache}"
OPENPROP_SIF="${OPENPROP_SIF:-$HOME/openprop-ai2thor.sif}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "python interpreter not found: $PYTHON_BIN" >&2
    exit 2
fi

case "$BUNDLE_DIR" in
    /*) ;;
    *) echo "bundle directory must be absolute: $BUNDLE_DIR" >&2; exit 2 ;;
esac

if [ ! -d "$OPENPROP_BASE" ] || [ ! -f "$OPENPROP_BASE/scripts/build_hpc_transfer_manifest.py" ]; then
    echo "extract openprop-hpc-source.tar.gz into $OPENPROP_BASE first" >&2
    exit 2
fi

cd "$OPENPROP_BASE"
"$PYTHON_BIN" scripts/build_hpc_transfer_manifest.py --directory "$BUNDLE_DIR" \
    --output "$BUNDLE_DIR/HPC_TRANSFER_MANIFEST.json" --check

if [ -e "$OPENPROP_SIF" ]; then
    current_hash=$(sha256sum "$OPENPROP_SIF" | awk '{print $1}')
    incoming_hash=$(sha256sum "$BUNDLE_DIR/openprop-ai2thor.sif" | awk '{print $1}')
    if [ "$current_hash" != "$incoming_hash" ]; then
        echo "refusing to overwrite a different SIF: $OPENPROP_SIF" >&2
        exit 2
    fi
else
    install -m 0444 "$BUNDLE_DIR/openprop-ai2thor.sif" "$OPENPROP_SIF"
fi

mkdir -p "$OPENPROP_CACHE/ai2thor"
tar -xzf "$BUNDLE_DIR/ai2thor-cloudrendering-f0825767-cache.tar.gz" \
    -C "$OPENPROP_CACHE/ai2thor"

"$PYTHON_BIN" scripts/prepare_visual_datasets.py --initialize \
    --require-ready ai2thor_ithor

echo "bundle verified"
echo "sif=$OPENPROP_SIF"
echo "cache=$OPENPROP_CACHE/ai2thor"
echo "next: cd $OPENPROP_BASE && SCENE=FloorPlan1 RUN_PREFLIGHT=1 sbatch hpc/ai2thor_capture.slurm"
