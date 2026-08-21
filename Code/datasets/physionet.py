"""PhysioNet/CinC Challenge 2012 forecasting dataset.

This module mirrors ``datasets/mujoco.py`` so the training runners
(``mujoco.py``/``mujoco-sde.py``), ``ODEvsSDE.py`` and the stress-test grid can
select it through ``--dataset_name physionet`` without any further changes.

Data: the public ``set-a`` and ``set-b`` training archives of the PhysioNet/
Computing in Cardiology Challenge 2012 (4000 patients each, 48-hour ICU
records). Every record is quantised onto a regular 1-hour grid of 48 steps;
the 41 channels follow the established Latent-ODE convention (36 time-series
variables plus the 5 static descriptors Age/Gender/Height/ICUType/Weight,
which are observed once at t=0). Unobserved bins are NaN, which the cubic
spline interpolation already used by this benchmark fills on the model side.

Differences to the MuJoCo pipeline, by design:

* Channels are z-scored with mean/std computed from the raw (uncorrupted)
  train-window slice, because PhysioNet variables live on wildly different
  scales (heart rate ~80, pH ~7.35). Targets are transformed with the same
  statistics, so all models train and are evaluated in the same space.
* Forecast targets keep their natural NaN entries (an unobserved future bin
  has no ground truth). The training losses and ``ODEvsSDE.evaluate`` mask
  them out; this module never fabricates target values.

The 70/15/15 train/val/test split, the windowing convention
(``range(trajectory_length - time_seq - y_seq)``), the corruption helpers and
the cache layout are identical to ``datasets/mujoco.py``.
"""

import pathlib
import tarfile
import urllib.request

import numpy as np
import torch

import controldiffeq
from . import common
from .mujoco import (
    _DEFAULT_MISSING_SEED,
    _add_scaled_input_noise,
    _append_time_channel,
    _apply_missingness,
    _finite_channel_std,
    _validate_corruption,
    _validate_window_lengths,
)


here = pathlib.Path(__file__).resolve().parent

NUM_FEATURES = 41
GRID_STEPS = 48  # 1-hour bins over the 48-hour challenge records.

PARAMS = [
    "Age", "Gender", "Height", "ICUType", "Weight",
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR",
    "K", "Lactate", "Mg", "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP",
    "NISysABP", "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2",
    "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC",
]
_PARAMS_DICT = {name: index for index, name in enumerate(PARAMS)}

# set-c is excluded to match the established Latent-ODE benchmark split; add
# its URL here (and its folder below) if the extra 4000 unlabelled patients
# are ever wanted.
_URLS = (
    "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download",
    "https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download",
)
_SET_NAMES = ("set-a", "set-b")

_RAW_DIR = here / "physionet2012" / "raw"
_PROCESSED_DIR = here / "physionet2012" / "processed"
_DENSE_CACHE = _PROCESSED_DIR / "physionet_1h.pt"


def _download_and_extract():
    """Fetch and unpack the challenge archives into ``_RAW_DIR``."""
    missing = [
        set_name
        for set_name in _SET_NAMES
        if not (_RAW_DIR / set_name).is_dir()
    ]
    if not missing:
        return

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    for url, set_name in zip(_URLS, _SET_NAMES):
        if set_name not in missing:
            continue
        archive_name = url.rsplit("/", 1)[-1].split("?")[0]
        archive_path = _RAW_DIR / archive_name
        print(f"Downloading {url} ...", flush=True)
        try:
            urllib.request.urlretrieve(url, archive_path)
        except Exception as error:
            raise RuntimeError(
                "Could not download the PhysioNet Challenge 2012 archives "
                f"automatically ({error}). Download set-a.tar.gz and "
                "set-b.tar.gz from https://physionet.org/content/"
                "challenge-2012/1.0.0/ manually and extract them so that "
                f"{_RAW_DIR}/set-a and {_RAW_DIR}/set-b contain the "
                "per-patient .txt files."
            ) from error
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(_RAW_DIR)
        archive_path.unlink()


def _parse_patient_file(txt_path):
    """Quantise one patient record onto the 48 x 41 hourly grid (NaN=missing)."""
    sums = np.zeros((GRID_STEPS, NUM_FEATURES), dtype=np.float64)
    counts = np.zeros((GRID_STEPS, NUM_FEATURES), dtype=np.float64)

    with open(txt_path) as handle:
        lines = handle.readlines()

    for line in lines[1:]:
        time_str, parameter, value = line.rstrip("\n").split(",")
        channel = _PARAMS_DICT.get(parameter)
        if channel is None:
            continue  # RecordID and any unexpected parameter.
        hours, minutes = time_str.split(":")
        time = int(hours) + int(minutes) / 60.0
        # Bin = elapsed hour; the rare readings at/after 48:00 fold into the
        # last bin.
        step = min(max(int(time), 0), GRID_STEPS - 1)
        sums[step, channel] += float(value)
        counts[step, channel] += 1.0

    grid = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
    return grid.astype(np.float32)


def _build_dense_cache():
    """Parse every patient text file once and cache the dense tensor."""
    _download_and_extract()

    grids = []
    record_ids = []
    for set_name in _SET_NAMES:
        set_dir = _RAW_DIR / set_name
        txt_files = sorted(set_dir.glob("*.txt"))
        if not txt_files:
            raise RuntimeError(
                f"No patient .txt files found in {set_dir}; the challenge "
                "archive may not have been extracted correctly."
            )
        for txt_path in txt_files:
            grids.append(_parse_patient_file(txt_path))
            record_ids.append(f"{set_name}/{txt_path.stem}")
        print(
            f"Parsed {len(txt_files)} patients from {set_name}.",
            flush=True,
        )

    values = torch.from_numpy(np.stack(grids))  # [patients, 48, 41]
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "values": values,
            "record_ids": record_ids,
            "params": PARAMS,
            "grid_steps": GRID_STEPS,
        },
        _DENSE_CACHE,
    )
    observed = torch.isfinite(values).float().mean().item()
    print(
        "PhysioNet dense grid: {} patients x {} steps x {} channels, "
        "observed fraction {:.3f}. Cached to {}.".format(
            values.size(0),
            values.size(1),
            values.size(2),
            observed,
            _DENSE_CACHE,
        ),
        flush=True,
    )
    return values


def _load_dense_records():
    if _DENSE_CACHE.exists():
        return torch.load(_DENSE_CACHE)["values"]
    return _build_dense_cache()


def _make_forecasting_windows(time_seq, y_seq):
    """Sliding windows with the repository's original MuJoCo convention."""
    values = _load_dense_records().contiguous()

    if values.ndim != 3:
        raise ValueError(
            "PhysioNet records must have shape [patients, time, channels]; "
            f"got {tuple(values.shape)}."
        )
    _, trajectory_length, _ = values.shape
    _validate_window_lengths(time_seq, y_seq, trajectory_length)

    window = time_seq + y_seq
    windows = values.unfold(1, window, 1)
    # Keep the original convention: range(trajectory_length - window),
    # without the trailing +1 window that unfold provides.
    windows = windows[:, :-1]
    if windows.size(1) != trajectory_length - window:
        raise RuntimeError("PhysioNet window slicing lost the convention.")

    inputs = windows[..., :time_seq]  # [patients, starts, channels, time]
    targets = windows[..., time_seq:]
    inputs = inputs.permute(0, 1, 3, 2).reshape(-1, time_seq, NUM_FEATURES)
    targets = targets.permute(0, 1, 3, 2).reshape(-1, y_seq, NUM_FEATURES)
    return inputs.clone(), targets.clone()


def _split_offsets(num_examples):
    """The shared sequential 70/15/15 split offsets."""
    return int(num_examples * 0.7), int(num_examples * 0.85)


def _train_channel_stats(train_inputs):
    """Z-score statistics over finite values of the raw train windows."""
    means = []
    stds = []
    for channel in range(train_inputs.size(-1)):
        values = train_inputs[..., channel]
        values = values[torch.isfinite(values)]
        if values.numel() < 2:
            means.append(torch.tensor(0.0, dtype=train_inputs.dtype))
            stds.append(torch.tensor(1.0, dtype=train_inputs.dtype))
        else:
            means.append(values.mean())
            stds.append(values.std(unbiased=False).clamp_min(1e-6))
    return torch.stack(means), torch.stack(stds)


def _zscore(windows, means, stds):
    return (windows - means.view(1, 1, -1)) / stds.view(1, 1, -1)


def _process_data(append_time, time_seq, missing_rate, y_seq):
    """Build the cached train/validation/test benchmark data."""
    _validate_corruption(missing_rate, 0.0, "random")
    input_windows, target_windows = _make_forecasting_windows(time_seq, y_seq)

    train_len, _ = _split_offsets(input_windows.size(0))
    means, stds = _train_channel_stats(input_windows[:train_len])
    input_windows = _zscore(input_windows, means, stds)
    target_windows = _zscore(target_windows, means, stds)

    input_windows = _apply_missingness(
        input_windows,
        missing_rate=missing_rate,
        missing_pattern="random",
        corruption_seed=_DEFAULT_MISSING_SEED,
    )

    final_indices = torch.full(
        (input_windows.size(0),),
        time_seq - 1,
        dtype=torch.long,
    )
    times = torch.linspace(1, time_seq, time_seq, dtype=input_windows.dtype)

    (
        times,
        train_coeffs,
        val_coeffs,
        test_coeffs,
        train_y,
        val_y,
        test_y,
        train_final_index,
        val_final_index,
        test_final_index,
        _,
    ) = common.preprocess_data_forecasting(
        times,
        input_windows,
        target_windows,
        final_indices,
        append_times=append_time,
    )

    return (
        times,
        train_coeffs,
        val_coeffs,
        test_coeffs,
        train_y,
        val_y,
        test_y,
        train_final_index,
        val_final_index,
        test_final_index,
    )


def _cache_location(append_time, time_seq, y_seq, missing_rate):
    suffix = "_time_aug" if append_time else ""
    return (
        here
        / "processed_data"
        / f"physionet{time_seq}_{y_seq}_{missing_rate}{suffix}"
    )


def get_data(
    batch_size,
    missing_rate,
    append_time,
    time_seq,
    y_seq,
    noise_std=0.0,
    loader_seed=0,
):
    """Return the cached loaders used for model training.

    Same contract as ``datasets.mujoco.get_data``; the stress/noise views are
    created test-only through ``get_stress_test_dataloader``.
    """
    del loader_seed  # Legacy loader draws from the runner-seeded global RNG.
    if noise_std != 0.0:
        raise ValueError(
            "get_data() no longer applies noise to validation/test data. "
            "Use get_stress_test_dataloader(..., noise_level=...) after "
            "training so model weights and validation selection stay fixed."
        )
    cache_root = here / "processed_data"
    loc = _cache_location(append_time, time_seq, y_seq, missing_rate)

    if loc.exists():
        tensors = common.load_data(loc)
        times = tensors["times"]
        train_coeffs = (
            tensors["train_a"],
            tensors["train_b"],
            tensors["train_c"],
            tensors["train_d"],
        )
        val_coeffs = (
            tensors["val_a"],
            tensors["val_b"],
            tensors["val_c"],
            tensors["val_d"],
        )
        test_coeffs = (
            tensors["test_a"],
            tensors["test_b"],
            tensors["test_c"],
            tensors["test_d"],
        )
        train_y = tensors["train_y"]
        val_y = tensors["val_y"]
        test_y = tensors["test_y"]
        train_final_index = tensors["train_final_index"]
        val_final_index = tensors["val_final_index"]
        test_final_index = tensors["test_final_index"]
    else:
        (
            times,
            train_coeffs,
            val_coeffs,
            test_coeffs,
            train_y,
            val_y,
            test_y,
            train_final_index,
            val_final_index,
            test_final_index,
        ) = _process_data(append_time, time_seq, missing_rate, y_seq)

        cache_root.mkdir(parents=True, exist_ok=True)
        loc.mkdir(parents=True, exist_ok=True)
        common.save_data(
            loc,
            times=times,
            train_a=train_coeffs[0],
            train_b=train_coeffs[1],
            train_c=train_coeffs[2],
            train_d=train_coeffs[3],
            val_a=val_coeffs[0],
            val_b=val_coeffs[1],
            val_c=val_coeffs[2],
            val_d=val_coeffs[3],
            test_a=test_coeffs[0],
            test_b=test_coeffs[1],
            test_c=test_coeffs[2],
            test_d=test_coeffs[3],
            train_y=train_y,
            val_y=val_y,
            test_y=test_y,
            train_final_index=train_final_index,
            val_final_index=val_final_index,
            test_final_index=test_final_index,
        )

    return common.wrap_data(
        times,
        train_coeffs,
        val_coeffs,
        test_coeffs,
        train_y,
        val_y,
        test_y,
        train_final_index,
        val_final_index,
        test_final_index,
        "cpu",
        batch_size=batch_size,
    )


def get_stress_test_dataloader(
    batch_size,
    append_time,
    time_seq,
    y_seq,
    missing_rate,
    noise_level,
    corruption_seed,
    missing_pattern="random",
):
    """Create a corrupted view of the unchanged held-out test windows.

    Identical contract and determinism guarantees as
    ``datasets.mujoco.get_stress_test_dataloader``: the same raw windows, the
    same sequential split, the same train-slice noise scales, and targets are
    never modified. The only addition is the z-score normalisation with
    train-slice statistics, applied before any corruption so the training and
    stress views live in the same space.
    """
    _validate_corruption(missing_rate, noise_level, missing_pattern)
    input_windows, target_windows = _make_forecasting_windows(time_seq, y_seq)

    train_len, test_start = _split_offsets(input_windows.size(0))
    means, stds = _train_channel_stats(input_windows[:train_len])
    input_windows = _zscore(input_windows, means, stds)
    target_windows = _zscore(target_windows, means, stds)

    train_inputs = input_windows[:train_len]
    test_inputs = input_windows[test_start:]
    test_targets = target_windows[test_start:]

    channel_std = _finite_channel_std(train_inputs)
    test_inputs = _apply_missingness(
        test_inputs,
        missing_rate=missing_rate,
        missing_pattern=missing_pattern,
        corruption_seed=corruption_seed,
        global_window_offset=test_start,
    )
    test_inputs = _add_scaled_input_noise(
        test_inputs,
        channel_std=channel_std,
        noise_level=noise_level,
        corruption_seed=corruption_seed,
    )

    if append_time:
        test_inputs = _append_time_channel(test_inputs)

    times = torch.linspace(
        0,
        time_seq - 1,
        time_seq,
        dtype=test_inputs.dtype,
    )
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(
        times,
        test_inputs,
    )
    test_final_index = torch.full(
        (test_inputs.size(0),),
        time_seq - 1,
        dtype=torch.long,
    )

    test_dataset = torch.utils.data.TensorDataset(
        *test_coeffs,
        test_targets,
        test_final_index,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=min(batch_size, len(test_dataset)),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return times, test_dataloader
