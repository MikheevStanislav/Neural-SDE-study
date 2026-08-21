# Crypto/MuJoCo/PhysioNet ODE/SDE forecasting benchmark

This directory contains the ANCDE forecasting baseline and the SDE
comparison/stress-test runner. `ODEvsSDE.py` trains one ODE and one SDE, then
reuses those weights for every test-only missing-data/noise condition. The SDE
prediction is estimated by Monte Carlo; training is not repeated inside the
stress grid.

Three datasets are supported through `--dataset_name`:

| Dataset | Source | Records | Grid | Channels | Recommended windows |
| --- | --- | --- | --- | --- | --- |
| `crypto` (sweep default) | bundled `datasets/crypto/crypto_history.zip` | 23 assets | daily, 2013–2021 | 6 → 6 | `--time_seq 50 --y_seq 20` (or `--y_seq 40`) |
| `mujoco` | `datasets/mujoco.npy` | 100 trajectories | 100 steps | 14 | `--time_seq 50 --y_seq 20` |
| `physionet` | PhysioNet/CinC Challenge 2012, set-a + set-b (auto-downloaded) | 8000 patients | 48 hourly bins | 41 | `--time_seq 24 --y_seq 12` |

## Cryptocurrency dataset

`datasets/crypto/crypto_history.zip` is a portable copy of the supplied
archive (SHA-256
`44b63683c0ef967adcf1bbc1f029e407cb372f1c526bff9226a0877d7562a933`).
It contains 23 daily OHLCV/market-cap series. No Desktop path and no network
download are needed on the server.

The model forecasts six stationary coordinates rather than incomparable raw
USD levels:

```text
log(Open_t / Close_t-1), log(High_t / Close_t-1),
log(Low_t / Close_t-1),  log(Close_t / Close_t-1),
delta log1p(Volume_t),    delta log(Marketcap_t).
```

Every coordinate is normalised separately per asset using a median and MAD
computed only from training dates, then clipped to `[-10, 10]`. Source zeros
in Volume/Marketcap are treated as unavailable; model inputs are filled only
from earlier observations, so input construction never consults the forecast
target. Windows spanning a missing calendar day are discarded.

All coins share one target-date split to prevent market-regime leakage:

- train target ends no later than 2021-01-31;
- validation target lies wholly in 2021-02-01 through 2021-04-30;
- test target lies wholly in 2021-05-01 through 2021-07-06.

For `time_seq=50`, the exact counts are:

| Forecast horizon | Train | Validation | Test |
| ---: | ---: | ---: | ---: |
| 20 | 31,254 | 1,610 | 1,104 |
| 40 | 30,794 | 1,150 | 644 |

Training samples are currently weighted per available window. Consequently,
older assets contribute more training examples than recently listed assets;
this choice is recorded in the result metadata. Validation and test use the
same recent calendar intervals for all 23 assets. If the scientific target is
an equal-weight mixture of assets rather than an equal-weight mixture of
asset-days, add inverse-frequency asset weighting before comparing the two
protocols.

As in the original forecasting code, the solver evolves over the observed
50-point context and the readout maps its last `y_seq` latent states to the
next `y_seq` targets. Thus a plotted “horizon variance” is an output-index
profile of this sequence-to-sequence forecast; it must not be described as a
Brownian path integrated for 20 or 40 additional calendar days.

The simplest launch is:

```bash
chmod +x run_crypto.sh run_diffusions.sh
./run_crypto.sh
```

Run one short end-to-end check before the full sweep:

```bash
EPOCH=1 MC_SAMPLES=2 CORRUPTION_REPEATS=1 \
DIFFUSION_OPTIONS="1" ./run_crypto.sh
```

For the 40-day forecast use `Y_SEQ=40 ./run_crypto.sh`. The spline/window
cache is versioned by the archive hash, preprocessing version, horizons,
missing rate and time augmentation under `datasets/processed_data`.

## PhysioNet 2012 notes

`--dataset_name physionet` builds forecasting windows from the public
PhysioNet/Computing in Cardiology Challenge 2012 training sets (set-a +
set-b = 8000 patients). On first use the archives are downloaded to
`datasets/physionet2012/raw` (PhysioNet account not required) and each
48-hour record is quantised onto a 1-hour grid of 48 steps; multiple
readings inside one bin are averaged. The 41 channels follow the
Latent-ODE convention: 36 time-series variables plus the five static
descriptors (Age, Gender, Height, ICUType, Weight) observed at t=0. The
parsed dense grid is cached in `datasets/physionet2012/processed`.

Two deliberate differences from the MuJoCo pipeline:

- Channels are z-scored with mean/std of the raw train-window slice
  (PhysioNet variables have wildly different scales); targets share the
  same transform, so training and evaluation stay in one space.
- Forecast targets keep their natural NaN entries (an unobserved future
  bin has no ground truth). Training losses and every target-dependent
  metric in `ODEvsSDE.py` are computed over observed entries only
  (masked MSE); prediction-only quantities (Monte Carlo variance,
  ODE-vs-SDE distances) are unchanged. For NaN-free targets the masked
  code paths reduce to the original dense computations.

Split, windowing, corruption helpers and cache layout are identical to
MuJoCo: sequential 70/15/15 over the flattened windows, spline
coefficients cached under `datasets/processed_data/physionet{time_seq}_{y_seq}_{missing_rate}`.
A stress run therefore looks like:

```bash
CUDA_VISIBLE_DEVICES=0 python -u ODEvsSDE.py \
  --ode_model ncde_forecasting \
  --model diffusionsde \
  --sde_input_option 1 \
  --sde_noise_option 7 \
  --dataset_name physionet \
  --time_seq 24 --y_seq 12 \
  --intensity false \
  --method euler \
  --epoch 200 \
  --mc_samples 20 \
  --test_missing_rates 0.0 0.3 0.5 0.7 \
  --input_noise_levels 0.0 0.1 0.2 \
  --corruption_repeats 5 \
  --missing_pattern random \
  --step_mode valloss \
  --explosion_threshold 100
```

With `--y_seq 12` every selected horizon index (`0,1,2,4,5,10,11`)
exists, matching the `--y_seq 20` MuJoCo analysis.

The historical names remain available as fixed presets:

| Model | Drift/input option | Diffusion option |
| --- | ---: | ---: |
| `staticsde` | 1 | 0 |
| `naivesde` | 1 | 18 |
| `neurallsde` | 2 | 16 |
| `neurallnsde` | 4 | 17 |
| `neuralgsde` | 6 | 17 |

Use `--model diffusionsde` with `--sde_input_option` and
`--sde_noise_option` to select an explicit configuration. Input options are
`0..6`; diffusion option `0` is the deterministic control and options `1..23`
are the non-zero diffusion families.

## Diffusion catalogue

In the formulas below, `y` is the latent state and time-dependent networks use
the bounded features `[sin(t), cos(t)]`. Scalar scales are broadcast over the
latent dimensions; diagonal scales have one learned value per dimension.

| Option | Stable name | Raw diffusion before the common output transform |
| ---: | --- | --- |
| 0 | `zero` | `0` (deterministic control) |
| 1 | `scalar_constant` | `exp(clamp(log_sigma, -20, 10))` |
| 2 | `scalar_time` | `exp(clamp(log_sigma, -20, 10)) * t` |
| 3 | `scalar_state` | `exp(clamp(log_sigma, -20, 10)) * y` |
| 4 | `diagonal_constant` | `exp(clamp(log_sigma_diag, -20, 10))` |
| 5 | `diagonal_time` | `exp(clamp(log_sigma_diag, -20, 10)) * t` |
| 6 | `diagonal_state` | `exp(clamp(log_sigma_diag, -20, 10)) * y` |
| 7 | `holder_sqrt_abs_state` | `sqrt(abs(y) + eps) - sqrt(eps)` |
| 8 | `cubic_state` | `clamp(y, -cube_limit(dtype), cube_limit(dtype)) ** 3` |
| 9 | `sigmoid_state` | `sigmoid(y)` |
| 10 | `relu_state` | `relu(y)` |
| 11 | `time_state` | `t * y` |
| 12 | `linear_time` | `linear_t([sin(t), cos(t)])` |
| 13 | `linear_time_times_state` | `linear_t([sin(t), cos(t)]) * y` |
| 14 | `linear_joint_time_state` | `linear_ty([sin(t), cos(t), y])` |
| 15 | `linear_joint_time_state_times_state` | `linear_ty([sin(t), cos(t), y]) * y` |
| 16 | `mlp_time` | `relu(mlp_t([sin(t), cos(t)]))` |
| 17 | `mlp_time_times_state` | `relu(mlp_t([sin(t), cos(t)])) * y` |
| 18 | `mlp_joint_time_state` | `relu(mlp_ty([sin(t), cos(t), y]))` |
| 19 | `mlp_joint_time_state_times_state` | `relu(mlp_ty([sin(t), cos(t), y])) * y` |
| 20 | `log1p_abs_state` | `log1p(abs(y))` |
| 21 | `exp_state` | `exp(clamp(y, max=10))` |
| 22 | `linear_time_times_linear_state` | `relu(linear_t([sin(t), cos(t)])) * relu(linear_state(y))` |
| 23 | `linear_time_plus_linear_state` | `relu(linear_t([sin(t), cos(t)])) + relu(linear_state(y))` |
| 24 | `mixture3_rms` | `s * sqrt(pi1*g1(t,y)^2 + pi2*g2(t,y)^2 + pi3*g3(t,y)^2), pi=softmax(alpha)` |

Every row above is subsequently transformed as

```text
g_effective(t, y) = tanh(sigmoid(theta) * g_raw(t, y)).
```

Consequently the diffusion actually passed to the solver is bounded. Results
must not be described as experiments with an unbounded raw formula alone.

When comparing the influence of the 23 diffusion functions, keep
`--sde_input_option` fixed (the sweep uses `1`). Changing it also changes the
drift architecture, so an observed difference could no longer be attributed
only to `g`.

### Mixture diffusion (option 24)

Option `24` combines three independently selected catalogue options into a
single RMS mixture. The raw diffusion is

```text
g_raw(t,y) = s * sqrt(pi1*g1(t,y)^2 + pi2*g2(t,y)^2 + pi3*g3(t,y)^2),
pi = softmax(alpha)
```

where `g1,g2,g3` are any options from `1..23` (each slot has its own
parameters), `alpha` is a learnable 3-vector initialized to zeros so `pi`
starts uniform, and `s = exp(clamp(log_s, -20, 10))` is a learnable positive
scale initialized to `1.0`. As with every other option, the effective
diffusion passed to the solver is `tanh(sigmoid(theta) * g_raw)`.

Select the components with `--sde_mixture_options`:

```bash
--model diffusionsde --sde_input_option 1 --sde_noise_option 24 \
  --sde_mixture_options 16 23 6
```

The default components are `16 23 6` (`mlp_time`,
`linear_time_plus_linear_state`, `diagonal_state`). Each component is
individually clamped before squaring so that extreme latent states cannot
overflow during the weighted sum. For the sweep script, set the environment
variable `MIXTURE_OPTIONS` (default `16 23 6`) when including `24` in
`DIFFUSION_OPTIONS`.

For the generic sweep, the vector field and forecasting head are initialized
from separate deterministic seed streams. This keeps shared drift/head weights
and the subsequent data/Brownian RNG state paired even though different
diffusion functions contain different numbers of parameters. The initialization
scheme and both seeds are stored in `sde_config.initialization`.

## One explicit ODE-versus-SDE run

The following reproduces the established 12-condition stress grid: four test
missing rates times three input-noise levels. Each trained SDE is sampled 20
times per condition, and each corruption condition is repeated five times.

```bash
CUDA_VISIBLE_DEVICES=0 /data/mzh/conda_envs/sde_stas/bin/python -u \
  /data/stas/Stable-Neural-SDEs/benchmark_forecasting/ODEvsSDE.py \
  --ode_model ncde_forecasting \
  --model diffusionsde \
  --sde_input_option 1 \
  --sde_noise_option 7 \
  --intensity false \
  --method euler \
  --epoch 200 \
  --mc_samples 20 \
  --mc_seed 12345 \
  --dataset_name crypto \
  --batch_size 1024 \
  --time_seq 50 \
  --y_seq 20 \
  --missing_rate 0.3 \
  --test_missing_rates 0.0 0.3 0.5 0.7 \
  --input_noise_levels 0.0 0.1 0.2 \
  --corruption_repeats 5 \
  --corruption_seed 24680 \
  --missing_pattern random \
  --step_mode valloss \
  --explosion_threshold 100
```

The saved JSON contains an `sde_config` block with the resolved input option,
diffusion option, stable name, raw formula, and effective formula.

Evaluation also performs one test-only drift ablation of the trained SDE. The
model is trained and selected with its learned diffusion active; only the
additional diagnostic forward uses `g_theta(z,t)=0`. Its three MSE distances
to the full-SDE Monte Carlo mean, the target, and the ODE prediction are saved
under `final_zero_diffusion_metrics`.

Detailed stochastic metrics are saved under
`final_sde_coordinate_metrics`: SDE-mean loss for every output coordinate,
Monte Carlo predictive variance by coordinate, variance by every forecast
horizon, and the full `[horizon, coordinate]` variance matrix. The console
highlights coordinates and horizons `0,1,2,4,5,10,11`; the complete arrays
remain available in JSON. With `--y_seq 20` every requested horizon exists.
The run command itself is unchanged; the extra drift-only solve costs one SDE
path per evaluated test view.

## Sequential sweep over all 23 functions

`run_diffusions.sh` derives the project directory from its own location. The
Python path can be overridden through `PYTHON_BIN`:

```bash
cd /data/stas/Stable-Neural-SDEs/benchmark_forecasting
chmod +x run_crypto.sh run_diffusions.sh
nohup ./run_crypto.sh > crypto_launcher.log 2>&1 &
echo $! > crypto_launcher.pid
```

The script runs options `1..23` sequentially and continues after an individual
failure. It exits non-zero at the end if any option failed. Every option gets a
new numbered directory under `tests_res/<dataset>`, containing `command.txt`,
`output.txt`, timestamps, exit-code files, and (after success) `result.json`.
The original numbered JSON is also retained under
`results/<dataset>/ODEvsSDE`.

Settings can be overridden without editing the script. This small smoke sweep,
for example, runs only three functions:

```bash
EPOCH=1 \
MC_SAMPLES=2 \
CORRUPTION_REPEATS=1 \
DIFFUSION_OPTIONS="7 20 23" \
GPU=0 \
./run_crypto.sh
```

Supported overrides are `PROJECT_DIR`, `PYTHON_BIN`, `GPU`, `EPOCH`,
`BATCH_SIZE`,
`MC_SAMPLES`, `CORRUPTION_REPEATS`, `SDE_INPUT_OPTION`,
`DIFFUSION_OPTIONS`, `DATASET_NAME`, `TIME_SEQ`, and `Y_SEQ`. Option `0`
may be supplied explicitly as a deterministic control, but it is not part
of the default 23-function sweep. For a PhysioNet sweep use
`DATASET_NAME=physionet TIME_SEQ=24 Y_SEQ=12`.

The full sweep retrains both models for every diffusion option and is therefore
long-running. Do not start two sweeps for the same dataset concurrently,
because each run copies the newly numbered result.

## Provenance

- Upstream ANCDE: `references/ANCDE`, commit
  `cce222f4602eae3dd2e0fbf069e20c6798dbd48e`.
- Local SDE path: `common_sde.py`, `models_sde/`, and `mujoco-sde.py`.
- The finite-horizon bounded-time features and outer `tanh` are benchmark
  stabilisation choices, not theorem-faithful replacements for every original
  LSDE/LNSDE/GSDE construction.
# Neural-SDE-study
