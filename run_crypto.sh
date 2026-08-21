#!/usr/bin/env bash

# Portable entry point for the bundled daily cryptocurrency dataset.
# Override any value from the shell, for example:
#   EPOCH=1 MC_SAMPLES=2 CORRUPTION_REPEATS=1 DIFFUSION_OPTIONS="1" ./run_crypto.sh

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

export PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
export DATASET_NAME="crypto"
export TIME_SEQ="${TIME_SEQ:-50}"
export Y_SEQ="${Y_SEQ:-20}"
export BATCH_SIZE="${BATCH_SIZE:-1024}"

exec "$SCRIPT_DIR/run_diffusions.sh"
