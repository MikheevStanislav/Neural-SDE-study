#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/data/mzh/conda_envs/sde_stas/bin/python}"
GPU="${GPU:-0}"
TESTS_DIR="${TESTS_DIR:-$PROJECT_DIR/tests_res}"
DATASET_NAME="${DATASET_NAME:-crypto}"
TIME_SEQ="${TIME_SEQ:-50}"
Y_SEQ="${Y_SEQ:-20}"
BATCH_SIZE="${BATCH_SIZE:-1024}"

TESTS_DIR="$TESTS_DIR/$DATASET_NAME"

mkdir -p "$TESTS_DIR"

LAST_RUN="$(
  find "$TESTS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | sed -nE 's/^([0-9]+).*/\1/p' \
    | sort -n \
    | tail -n 1
)"
LAST_RUN="${LAST_RUN:-0}"

RUN_NUMBER=$((10#$LAST_RUN + 1))
RUN="$(printf '%03d' "$RUN_NUMBER")"
RUN_DIR="$TESTS_DIR/${RUN}_${DATASET_NAME}_naivesde"

mkdir -p "$RUN_DIR"
date -Is > "$RUN_DIR/started_at.txt"
touch "$RUN_DIR/.started"

CMD=(
  "$PYTHON_BIN" -u "$PROJECT_DIR/ODEvsSDE.py"
  --ode_model ncde_forecasting
  --model naivesde
  --intensity false
  --method euler
  --epoch 200
  --mc_samples 20
  --mc_seed 12345
  --dataset_name "$DATASET_NAME"
  --batch_size "$BATCH_SIZE"
  --time_seq "$TIME_SEQ"
  --y_seq "$Y_SEQ"
  --missing_rate 0.3
  --test_missing_rates 0.3 0.7
  --input_noise_levels 0 0.1
  --corruption_repeats 5
  --corruption_seed 24680
  --missing_pattern random
  --step_mode valloss
  --explosion_threshold 100
)

{
  printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU"
  printf ' %q' "${CMD[@]}"
  printf '\n'
} > "$RUN_DIR/command.txt"

echo "Results directory: $RUN_DIR"

set -o pipefail
CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" 2>&1 | tee "$RUN_DIR/output.txt"
STATUS=${PIPESTATUS[0]}

printf '%s\n' "$STATUS" > "$RUN_DIR/exit_code.txt"
date -Is > "$RUN_DIR/finished_at.txt"

if (( STATUS != 0 )); then
  echo "Experiment failed with exit code $STATUS"
  echo "Log: $RUN_DIR/output.txt"
  exit "$STATUS"
fi

RESULT_FILE="$(
  find "$PROJECT_DIR/results/$DATASET_NAME/ODEvsSDE" \
    -maxdepth 1 -type f -newer "$RUN_DIR/.started" \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
)"

if [[ -n "$RESULT_FILE" ]]; then
  cp "$RESULT_FILE" "$RUN_DIR/result.json"
  echo "Result copied to: $RUN_DIR/result.json"
else
  echo "WARNING: new result file was not found"
fi

echo "Experiment completed: $RUN_DIR"
