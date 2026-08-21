#!/usr/bin/env bash

# Run the 23 non-zero diffusion families one after another. Each invocation
# trains its ODE and SDE once, then evaluates the established corruption grid.
# Environment variables below may be overridden without editing this file.

set -u
set -o pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/data/mzh/conda_envs/sde_stas/bin/python}"
GPU="${GPU:-0}"
EPOCH="${EPOCH:-200}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MC_SAMPLES="${MC_SAMPLES:-20}"
CORRUPTION_REPEATS="${CORRUPTION_REPEATS:-5}"
SDE_INPUT_OPTION="${SDE_INPUT_OPTION:-1}"
DIFFUSION_OPTIONS="${DIFFUSION_OPTIONS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23}"
MIXTURE_OPTIONS="${MIXTURE_OPTIONS:-16 23 6}"
# Dataset selection: crypto (bundled daily archive), mujoco, or physionet.
DATASET_NAME="${DATASET_NAME:-crypto}"
TIME_SEQ="${TIME_SEQ:-50}"
Y_SEQ="${Y_SEQ:-20}"

TESTS_DIR="${TESTS_DIR:-$PROJECT_DIR/tests_res/$DATASET_NAME}"
RESULTS_DIR="$PROJECT_DIR/results/$DATASET_NAME/ODEvsSDE"

declare -a DIFFUSION_LABELS=(
  [0]="zero"
  [1]="scalar_constant"
  [2]="scalar_time"
  [3]="scalar_state"
  [4]="diagonal_constant"
  [5]="diagonal_time"
  [6]="diagonal_state"
  [7]="holder_sqrt_abs_state"
  [8]="cubic_state"
  [9]="sigmoid_state"
  [10]="relu_state"
  [11]="time_state"
  [12]="linear_time"
  [13]="linear_time_times_state"
  [14]="linear_joint_time_state"
  [15]="linear_joint_time_state_times_state"
  [16]="mlp_time"
  [17]="mlp_time_times_state"
  [18]="mlp_joint_time_state"
  [19]="mlp_joint_time_state_times_state"
  [20]="log1p_abs_state"
  [21]="exp_state"
  [22]="linear_time_times_linear_state"
  [23]="linear_time_plus_linear_state"
  [24]="mixture3_rms"
)

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

fail_usage() {
  echo "ERROR: $*" >&2
  exit 2
}

require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || fail_usage "$name must be a non-negative integer; got '$value'."
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  require_nonnegative_integer "$name" "$value"
  (( 10#$value >= 1 )) || fail_usage "$name must be at least 1; got '$value'."
}

next_run_number() {
  local path base prefix number
  local maximum=0

  shopt -s nullglob
  for path in "$TESTS_DIR"/*; do
    [[ -d "$path" ]] || continue
    base="${path##*/}"
    if [[ "$base" =~ ^([0-9]+) ]]; then
      prefix="${BASH_REMATCH[1]}"
      number=$((10#$prefix))
      (( number > maximum )) && maximum="$number"
    fi
  done
  shopt -u nullglob

  printf '%03d' "$((maximum + 1))"
}

maximum_numeric_result() {
  local path base number
  local maximum=-1

  [[ -d "$RESULTS_DIR" ]] || {
    printf '%d' "$maximum"
    return
  }

  shopt -s nullglob
  for path in "$RESULTS_DIR"/*; do
    [[ -f "$path" ]] || continue
    base="${path##*/}"
    if [[ "$base" =~ ^[0-9]+$ ]]; then
      number=$((10#$base))
      (( number > maximum )) && maximum="$number"
    fi
  done
  shopt -u nullglob

  printf '%d' "$maximum"
}

newest_numeric_result_after() {
  local previous_max="$1"
  local path base number
  local newest_number=-1
  local newest_path=""

  [[ -d "$RESULTS_DIR" ]] || return 1

  shopt -s nullglob
  for path in "$RESULTS_DIR"/*; do
    [[ -f "$path" ]] || continue
    base="${path##*/}"
    if [[ "$base" =~ ^[0-9]+$ ]]; then
      number=$((10#$base))
      if (( number > previous_max && number > newest_number )); then
        newest_number="$number"
        newest_path="$path"
      fi
    fi
  done
  shopt -u nullglob

  [[ -n "$newest_path" ]] || return 1
  printf '%s' "$newest_path"
}

[[ -f "$PROJECT_DIR/ODEvsSDE.py" ]] || fail_usage "ODEvsSDE.py was not found in $PROJECT_DIR."
if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || fail_usage "Python is not executable: $PYTHON_BIN"
else
  python_command="$PYTHON_BIN"
  PYTHON_BIN="$(command -v "$python_command")" || fail_usage "Python command was not found: $python_command"
fi

require_positive_integer EPOCH "$EPOCH"
require_positive_integer BATCH_SIZE "$BATCH_SIZE"
require_positive_integer MC_SAMPLES "$MC_SAMPLES"
(( 10#$MC_SAMPLES >= 2 )) || fail_usage "MC_SAMPLES must be at least 2 to estimate variance."
require_positive_integer CORRUPTION_REPEATS "$CORRUPTION_REPEATS"
require_positive_integer TIME_SEQ "$TIME_SEQ"
require_positive_integer Y_SEQ "$Y_SEQ"
require_nonnegative_integer SDE_INPUT_OPTION "$SDE_INPUT_OPTION"
(( 10#$SDE_INPUT_OPTION <= 6 )) || fail_usage "SDE_INPUT_OPTION must be between 0 and 6."
SDE_INPUT_OPTION=$((10#$SDE_INPUT_OPTION))

read -r -a REQUESTED_OPTIONS <<< "$DIFFUSION_OPTIONS"
(( ${#REQUESTED_OPTIONS[@]} > 0 )) || fail_usage "DIFFUSION_OPTIONS must not be empty."
for index in "${!REQUESTED_OPTIONS[@]}"; do
  option="${REQUESTED_OPTIONS[$index]}"
  require_nonnegative_integer DIFFUSION_OPTIONS "$option"
  (( 10#$option <= 24 )) || fail_usage "Diffusion option must be between 0 and 24; got '$option'."
  REQUESTED_OPTIONS[$index]=$((10#$option))
done

read -r -a MIXTURE_OPTS <<< "$MIXTURE_OPTIONS"
(( ${#MIXTURE_OPTS[@]} == 3 )) || fail_usage "MIXTURE_OPTIONS must contain exactly 3 integers."
for mixopt in "${MIXTURE_OPTS[@]}"; do
  require_nonnegative_integer MIXTURE_OPTIONS "$mixopt"
  (( 10#$mixopt >= 1 && 10#$mixopt <= 23 )) || fail_usage "Mixture component option must be between 1 and 23; got '$mixopt'."
done

# Fail before creating run directories when only this launcher was copied to
# the server but the Python implementation is still an older revision.
PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" - "$PROJECT_DIR" "$SDE_INPUT_OPTION" "$DATASET_NAME" "$TIME_SEQ" "$Y_SEQ" "$BATCH_SIZE" <<'PY'
import sys

project_dir = sys.argv[1]
input_option = int(sys.argv[2])
dataset_name = sys.argv[3]
time_seq = int(sys.argv[4])
y_seq = int(sys.argv[5])
batch_size = int(sys.argv[6])
sys.path.insert(0, project_dir)

from parse import parse_args

sys.argv = [
    "run_diffusions_preflight",
    "--model",
    "diffusionsde",
    "--sde_input_option",
    str(input_option),
    "--sde_noise_option",
    "1",
    "--intensity",
    "false",
    "--dataset_name",
    dataset_name,
    "--time_seq",
    str(time_seq),
    "--y_seq",
    str(y_seq),
    "--batch_size",
    str(batch_size),
]
args = parse_args()
if (
    args.sde_input_option != input_option
    or args.sde_noise_option != 1
    or args.dataset_name != dataset_name
    or args.batch_size != batch_size
):
    raise RuntimeError("Diffusion CLI arguments were parsed incorrectly.")

import common_sde

configuration = common_sde.resolve_sde_config(
    model_name="diffusionsde",
    sde_input_option=input_option,
    sde_noise_option=1,
)
if configuration.model_name != "diffusionsde":
    raise RuntimeError("Generic diffusion model is not registered.")

import datasets

dataset = datasets.get_dataset(dataset_name)
input_features, output_features = datasets.feature_dimensions(dataset)
if not hasattr(dataset, "get_stress_test_dataloader"):
    raise RuntimeError(
        f"Dataset {dataset_name!r} has no stress-test loader."
    )
details = None
if hasattr(dataset, "validate_source"):
    details = dataset.validate_source(time_seq=time_seq, y_seq=y_seq)

print(
    "Preflight OK | model={} | input={} | diffusion={} | "
    "dataset={} ({}->{}) | horizons={}/{}{}".format(
        configuration.model_name,
        configuration.input_option,
        configuration.noise_option,
        dataset_name,
        input_features,
        output_features,
        time_seq,
        y_seq,
        " | windows={}".format(details["window_counts"])
        if details and "window_counts" in details
        else "",
    )
)
PY
preflight_status=$?
if (( preflight_status != 0 )); then
  fail_usage "Diffusion Python files are missing or out of sync. Copy/apply the complete implementation before rerunning."
fi

mkdir -p "$TESTS_DIR"

TOTAL=${#REQUESTED_OPTIONS[@]}
COMPLETED=0
FAILURES=0

echo "Diffusion sweep started: $(timestamp)"
echo "Project: $PROJECT_DIR"
echo "Python: $PYTHON_BIN"
echo "GPU: $GPU | batch: $BATCH_SIZE | dataset: $DATASET_NAME (time_seq=$TIME_SEQ, y_seq=$Y_SEQ) | input option: $SDE_INPUT_OPTION | diffusion options: ${REQUESTED_OPTIONS[*]}"

for option in "${REQUESTED_OPTIONS[@]}"; do
  label="${DIFFUSION_LABELS[$option]}"
  run_number="$(next_run_number)"
  printf -v option_padded '%02d' "$option"
  run_dir="$TESTS_DIR/${run_number}_${DATASET_NAME}_diffusion_i${SDE_INPUT_OPTION}_g${option_padded}_${label}"
  start_marker="$run_dir/.started"

  mkdir -p "$run_dir"
  timestamp > "$run_dir/started_at.txt"
  touch "$start_marker"
  result_max_before="$(maximum_numeric_result)"

  command=(
    "$PYTHON_BIN" -u "$PROJECT_DIR/ODEvsSDE.py"
    --ode_model ncde_forecasting
    --model diffusionsde
    --sde_input_option "$SDE_INPUT_OPTION"
    --sde_noise_option "$option"
    --intensity false
    --method euler
    --seed 12
    --epoch "$EPOCH"
    --h_channels 49
    --hh_channels 49
    --layers 4
    --lr 0.0001
    --weight_decay 0.00001
    --loss mse
    --reg l2
    --scale 0.01
    --mc_samples "$MC_SAMPLES"
    --mc_seed 12345
    --batch_size "$BATCH_SIZE"
    --dataset_name "$DATASET_NAME"
    --time_seq "$TIME_SEQ"
    --y_seq "$Y_SEQ"
    --missing_rate 0.0
    --test_missing_rates 0.0 0.7
    --input_noise_levels 0.0 0.1
    --corruption_repeats "$CORRUPTION_REPEATS"
    --corruption_seed 24680
    --missing_pattern random
    --step_mode valloss
    --explosion_threshold 100
  )
  if (( option == 24 )); then
    command+=(--sde_mixture_options "${MIXTURE_OPTS[@]}")
  fi

  {
    printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU"
    printf ' %q' "${command[@]}"
    printf '\n'
  } > "$run_dir/command.txt"

  echo
  echo "[$((COMPLETED + 1))/$TOTAL] Option $option ($label)"
  echo "Results directory: $run_dir"

  CUDA_VISIBLE_DEVICES="$GPU" "${command[@]}" 2>&1 | tee "$run_dir/output.txt"
  pipeline_status=("${PIPESTATUS[@]}")
  python_status="${pipeline_status[0]}"
  tee_status="${pipeline_status[1]}"
  status="$python_status"
  if (( status == 0 && tee_status != 0 )); then
    status="$tee_status"
  fi

  result_file=""
  if result_file="$(newest_numeric_result_after "$result_max_before")"; then
    if cp "$result_file" "$run_dir/result.json"; then
      echo "Result copied to: $run_dir/result.json"
    else
      status=74
      echo "ERROR: could not copy $result_file into the run directory." | tee -a "$run_dir/output.txt"
    fi
  elif (( status == 0 )); then
    status=66
    echo "ERROR: the run exited successfully but no new result JSON was found." | tee -a "$run_dir/output.txt"
  else
    echo "No result JSON was produced for the failed run." | tee -a "$run_dir/output.txt"
  fi

  printf '%s\n' "$python_status" > "$run_dir/python_exit_code.txt"
  printf '%s\n' "$tee_status" > "$run_dir/tee_exit_code.txt"
  printf '%s\n' "$status" > "$run_dir/exit_code.txt"
  timestamp > "$run_dir/finished_at.txt"

  if (( status != 0 )); then
    ((FAILURES += 1))
    echo "Option $option failed with exit code $status; continuing with the next option."
  else
    echo "Option $option completed successfully."
  fi
  ((COMPLETED += 1))
done

echo
echo "Diffusion sweep finished: $(timestamp)"
echo "Completed: $COMPLETED/$TOTAL | Failed: $FAILURES"

if (( FAILURES > 0 )); then
  exit 1
fi
