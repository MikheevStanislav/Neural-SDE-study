"""Leakage-safe daily cryptocurrency forecasting dataset.

The bundled archive contains one CSV per asset.  The adapter turns the raw
OHLCV/market-cap levels into six stationary, dimensionless coordinates and
uses one global calendar split for every asset.  This is important: a
per-asset percentage split would allow a late market regime to be training
data for one coin whilst the same dates are test data for another coin.

Inputs and targets use the following coordinates::

    log(Open_t / Close_{t-1})
    log(High_t / Close_{t-1})
    log(Low_t / Close_{t-1})
    log(Close_t / Close_{t-1})
    log1p(Volume_t) - log1p(Volume_{t-1})
    log(Marketcap_t / Marketcap_{t-1})

Each coordinate is robustly normalised per asset using only observations in
the training period.  Non-positive volume/market-cap values are treated as
unavailable.  They remain NaN in targets (the benchmark loss masks them) and
are causally forward-filled only when used as model inputs.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import io
import math
import os
import pathlib
import stat
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


here = pathlib.Path(__file__).resolve().parent

NUM_FEATURES = 6
NUM_INPUT_FEATURES = 6
NUM_OUTPUT_FEATURES = 6
EXPECTED_ASSET_COUNT = 23

FEATURE_NAMES = (
    "open_log_return",
    "high_log_return",
    "low_log_return",
    "close_log_return",
    "volume_log_change",
    "marketcap_log_change",
)

RAW_COLUMNS = (
    "SNo",
    "Name",
    "Symbol",
    "Date",
    "High",
    "Low",
    "Open",
    "Close",
    "Volume",
    "Marketcap",
)

# All assets use the same market regimes.  A target block must fit completely
# inside exactly one of these intervals.  Past context may cross a boundary.
TRAIN_END = date(2021, 1, 31)
VAL_START = date(2021, 2, 1)
VAL_END = date(2021, 4, 30)
TEST_START = date(2021, 5, 1)
TEST_END = date(2021, 7, 6)

PREPROCESSING_VERSION = "crypto_daily_returns_v1"
ROBUST_CLIP = 10.0
_DEFAULT_MISSING_SEED = 56_789
_NOISE_SEED_OFFSET = 1_000_003
_DEFAULT_ARCHIVE = here / "crypto" / "crypto_history.zip"
_PROCESSED_ROOT = here / "processed_data"


@dataclass(frozen=True)
class _AssetSeries:
    name: str
    symbol: str
    dates: Tuple[date, ...]
    values: np.ndarray


def _resolved_archive_path(archive_path=None):
    if archive_path is None:
        archive_path = os.environ.get("CRYPTO_DATA_ARCHIVE", _DEFAULT_ARCHIVE)
    path = pathlib.Path(archive_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Cryptocurrency archive was not found at {}. Copy the supplied "
            "archive to datasets/crypto/crypto_history.zip or set "
            "CRYPTO_DATA_ARCHIVE.".format(path)
        )
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_csv_members(archive):
    members = []
    for info in archive.infolist():
        member = pathlib.PurePosixPath(info.filename.replace("\\", "/"))
        unix_mode = info.external_attr >> 16
        if (
            member.is_absolute()
            or ".." in member.parts
            or stat.S_ISLNK(unix_mode)
        ):
            raise ValueError(
                f"Unsafe path or symbolic link in crypto archive: {info.filename!r}."
            )
        if info.is_dir():
            continue
        if member.suffix.lower() != ".csv":
            raise ValueError(
                f"Unexpected non-CSV member in crypto archive: {info.filename!r}."
            )
        members.append(info)
    if not members:
        raise ValueError("The cryptocurrency archive contains no CSV files.")
    return sorted(members, key=lambda item: item.filename)


def _parse_member(archive, info):
    with archive.open(info, "r") as binary:
        text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
        reader = csv.DictReader(text)
        columns = tuple(reader.fieldnames or ())
        missing_columns = [column for column in RAW_COLUMNS if column not in columns]
        if missing_columns:
            raise ValueError(
                f"{info.filename} is missing required columns {missing_columns}; "
                f"found {list(columns)}."
            )

        parsed_dates = []
        parsed_values = []
        asset_name = None
        symbol = None
        previous_date = None

        for row_number, row in enumerate(reader, start=2):
            try:
                current_date = datetime.fromisoformat(row["Date"]).date()
                current_values = np.asarray(
                    [
                        float(row["Open"]),
                        float(row["High"]),
                        float(row["Low"]),
                        float(row["Close"]),
                        float(row["Volume"]),
                        float(row["Marketcap"]),
                    ],
                    dtype=np.float64,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid date or numeric value in {info.filename}:{row_number}."
                ) from error

            if not np.isfinite(current_values).all():
                raise ValueError(
                    f"Non-finite numeric value in {info.filename}:{row_number}."
                )
            if np.any(current_values[:4] <= 0.0):
                raise ValueError(
                    f"OHLC prices must be positive in {info.filename}:{row_number}."
                )
            if np.any(current_values[4:] < 0.0):
                raise ValueError(
                    f"Volume and Marketcap must be non-negative in "
                    f"{info.filename}:{row_number}."
                )
            if previous_date is not None and current_date <= previous_date:
                raise ValueError(
                    f"Dates must be unique and strictly increasing in {info.filename}."
                )

            row_name = row["Name"].strip()
            row_symbol = row["Symbol"].strip()
            if not row_name or not row_symbol:
                raise ValueError(
                    f"Empty Name or Symbol in {info.filename}:{row_number}."
                )
            if asset_name is None:
                asset_name, symbol = row_name, row_symbol
            elif asset_name != row_name or symbol != row_symbol:
                raise ValueError(
                    f"Multiple assets are mixed inside {info.filename}."
                )

            parsed_dates.append(current_date)
            parsed_values.append(current_values)
            previous_date = current_date

    if not parsed_dates:
        raise ValueError(f"{info.filename} contains no observations.")
    return _AssetSeries(
        name=asset_name,
        symbol=symbol,
        dates=tuple(parsed_dates),
        values=np.stack(parsed_values),
    )


@functools.lru_cache(maxsize=4)
def _load_assets_cached(path_string, file_size, modified_ns, expected_count):
    del file_size, modified_ns  # These values only invalidate the cache key.
    path = pathlib.Path(path_string)
    with zipfile.ZipFile(path, "r") as archive:
        members = _safe_csv_members(archive)
        assets = tuple(_parse_member(archive, member) for member in members)

    if expected_count is not None and len(assets) != expected_count:
        raise ValueError(
            f"Expected {expected_count} crypto CSV files, found {len(assets)}."
        )
    symbols = [asset.symbol for asset in assets]
    if len(symbols) != len(set(symbols)):
        raise ValueError("Every CSV must contain a unique cryptocurrency symbol.")
    return tuple(sorted(assets, key=lambda asset: asset.symbol))


def _load_assets(archive_path=None, expected_count=EXPECTED_ASSET_COUNT):
    path = _resolved_archive_path(archive_path)
    metadata = path.stat()
    return _load_assets_cached(
        str(path),
        metadata.st_size,
        metadata.st_mtime_ns,
        expected_count,
    )


def _validate_window_lengths(time_seq, y_seq):
    if isinstance(time_seq, bool) or not isinstance(time_seq, int) or time_seq < 2:
        raise ValueError("time_seq must be an integer of at least 2.")
    if isinstance(y_seq, bool) or not isinstance(y_seq, int) or y_seq < 1:
        raise ValueError("y_seq must be a positive integer.")
    if y_seq > time_seq:
        raise ValueError(
            "The current forecasting heads require y_seq <= time_seq; "
            f"got y_seq={y_seq}, time_seq={time_seq}."
        )
    shortest_period = min(
        (VAL_END - VAL_START).days + 1,
        (TEST_END - TEST_START).days + 1,
    )
    if y_seq > shortest_period:
        raise ValueError(
            f"y_seq={y_seq} does not fit the shortest validation/test period "
            f"({shortest_period} days)."
        )


def _feature_engineering(asset):
    """Return daily features, calendar-presence mask and daily dates."""
    first_date, last_date = asset.dates[0], asset.dates[-1]
    num_days = (last_date - first_date).days + 1
    calendar_dates = tuple(first_date + timedelta(days=index) for index in range(num_days))
    raw = np.full((num_days, NUM_FEATURES), np.nan, dtype=np.float64)
    present = np.zeros(num_days, dtype=bool)

    for current_date, values in zip(asset.dates, asset.values):
        index = (current_date - first_date).days
        raw[index] = values
        present[index] = True

    features = np.full_like(raw, np.nan)
    for index in range(1, num_days):
        if not (present[index - 1] and present[index]):
            continue
        previous_close = raw[index - 1, 3]
        features[index, :4] = np.log(raw[index, :4] / previous_close)

        previous_volume, current_volume = raw[index - 1, 4], raw[index, 4]
        if previous_volume > 0.0 and current_volume > 0.0:
            features[index, 4] = math.log1p(current_volume) - math.log1p(
                previous_volume
            )

        previous_cap, current_cap = raw[index - 1, 5], raw[index, 5]
        if previous_cap > 0.0 and current_cap > 0.0:
            features[index, 5] = math.log(current_cap / previous_cap)

    return calendar_dates, present, features


def _robust_normalise(features, calendar_dates):
    train_mask = np.asarray([current_date <= TRAIN_END for current_date in calendar_dates])
    normalised = np.full_like(features, np.nan)
    centres = np.zeros(NUM_FEATURES, dtype=np.float64)
    scales = np.ones(NUM_FEATURES, dtype=np.float64)
    clipped = np.zeros(NUM_FEATURES, dtype=np.int64)
    finite_counts = np.zeros(NUM_FEATURES, dtype=np.int64)

    for channel in range(NUM_FEATURES):
        train_values = features[train_mask, channel]
        train_values = train_values[np.isfinite(train_values)]
        if train_values.size < 2:
            raise ValueError(
                f"Not enough finite training observations for channel "
                f"{FEATURE_NAMES[channel]}."
            )
        centre = float(np.median(train_values))
        mad_scale = float(1.4826 * np.median(np.abs(train_values - centre)))
        if not math.isfinite(mad_scale) or mad_scale < 1e-8:
            mad_scale = float(np.std(train_values))
        if not math.isfinite(mad_scale) or mad_scale < 1e-8:
            mad_scale = 1.0

        values = (features[:, channel] - centre) / mad_scale
        finite = np.isfinite(values)
        clipped[channel] = int(np.count_nonzero(np.abs(values[finite]) > ROBUST_CLIP))
        finite_counts[channel] = int(np.count_nonzero(finite))
        values[finite] = np.clip(values[finite], -ROBUST_CLIP, ROBUST_CLIP)

        normalised[:, channel] = values
        centres[channel] = centre
        scales[channel] = mad_scale

    return normalised, centres, scales, clipped, finite_counts


def _causal_fill_for_inputs(values):
    """Forward-fill without consulting a future or target observation."""
    filled = np.empty_like(values)
    for channel in range(values.shape[1]):
        last_value = 0.0  # Robustly normalised train centre.
        for index, value in enumerate(values[:, channel]):
            if math.isfinite(float(value)):
                last_value = float(value)
            filled[index, channel] = last_value
    return filled


def _split_for_target(start_date, end_date):
    if end_date <= TRAIN_END:
        return "train"
    if start_date >= VAL_START and end_date <= VAL_END:
        return "val"
    if start_date >= TEST_START and end_date <= TEST_END:
        return "test"
    return None


def _build_window_splits(time_seq, y_seq, archive_path=None):
    _validate_window_lengths(time_seq, y_seq)
    assets = _load_assets(archive_path)
    split_inputs = {name: [] for name in ("train", "val", "test")}
    split_targets = {name: [] for name in ("train", "val", "test")}
    split_asset_ids = {name: [] for name in ("train", "val", "test")}
    split_target_dates = {name: [] for name in ("train", "val", "test")}
    asset_metadata = []

    for asset_index, asset in enumerate(assets):
        calendar_dates, calendar_present, features = _feature_engineering(asset)
        (
            target_values,
            centres,
            scales,
            clipped,
            finite_counts,
        ) = _robust_normalise(features, calendar_dates)
        input_values = _causal_fill_for_inputs(target_values)
        per_split_counts = {name: 0 for name in split_inputs}

        last_target_start = len(calendar_dates) - y_seq
        for target_start in range(time_seq, last_target_start + 1):
            input_start = target_start - time_seq
            target_stop = target_start + y_seq
            # Calendar gaps are not silently fabricated in either context or
            # target.  Source-zero Volume/Marketcap values are handled at the
            # coordinate level instead.
            if not calendar_present[input_start:target_stop].all():
                continue
            source_input = target_values[input_start:target_start]
            if not np.isfinite(source_input).any(axis=0).all():
                continue

            target_start_date = calendar_dates[target_start]
            target_end_date = calendar_dates[target_stop - 1]
            split_name = _split_for_target(target_start_date, target_end_date)
            if split_name is None:
                continue

            split_inputs[split_name].append(
                input_values[input_start:target_start].astype(np.float32, copy=True)
            )
            split_targets[split_name].append(
                target_values[target_start:target_stop].astype(np.float32, copy=True)
            )
            split_asset_ids[split_name].append(asset_index)
            split_target_dates[split_name].append(target_start_date.toordinal())
            per_split_counts[split_name] += 1

        asset_metadata.append(
            {
                "name": asset.name,
                "symbol": asset.symbol,
                "first_date": asset.dates[0].isoformat(),
                "last_date": asset.dates[-1].isoformat(),
                "source_rows": len(asset.dates),
                "window_counts": per_split_counts,
                "normalisation_centre": centres.tolist(),
                "normalisation_scale": scales.tolist(),
                "clipped_count": clipped.tolist(),
                "finite_feature_count": finite_counts.tolist(),
            }
        )

    output = {}
    for split_name in split_inputs:
        if not split_inputs[split_name]:
            raise ValueError(
                f"No {split_name} windows were produced for "
                f"time_seq={time_seq}, y_seq={y_seq}."
            )
        output[split_name] = {
            "inputs": torch.from_numpy(np.stack(split_inputs[split_name])),
            "targets": torch.from_numpy(np.stack(split_targets[split_name])),
            "asset_ids": torch.tensor(split_asset_ids[split_name], dtype=torch.long),
            "target_start_ordinals": torch.tensor(
                split_target_dates[split_name], dtype=torch.long
            ),
        }

    source_path = _resolved_archive_path(archive_path)
    output["metadata"] = {
        "dataset_name": "crypto",
        "preprocessing_version": PREPROCESSING_VERSION,
        "archive_sha256": _sha256(source_path),
        "input_features": NUM_INPUT_FEATURES,
        "output_features": NUM_OUTPUT_FEATURES,
        "feature_names": list(FEATURE_NAMES),
        "feature_transform": "stationary daily log returns/log changes",
        "normalisation": "per-asset train-only median / (1.4826 * MAD)",
        "normalisation_clip": ROBUST_CLIP,
        "input_missing_value_policy": "causal forward fill after normalisation",
        "target_missing_value_policy": "NaN retained and masked by loss",
        "calendar_gap_policy": "drop every window spanning a missing day",
        "training_sampling": (
            "uniform over available windows; assets with longer histories "
            "contribute more training windows"
        ),
        "split": {
            "train_target_end": TRAIN_END.isoformat(),
            "validation_target_start": VAL_START.isoformat(),
            "validation_target_end": VAL_END.isoformat(),
            "test_target_start": TEST_START.isoformat(),
            "test_target_end": TEST_END.isoformat(),
        },
        "time_seq": time_seq,
        "y_seq": y_seq,
        "asset_count": len(assets),
        "assets": asset_metadata,
        "window_counts": {
            name: int(output[name]["inputs"].size(0))
            for name in ("train", "val", "test")
        },
    }
    return output


def _window_cache_path(time_seq, y_seq, archive_path=None):
    source_path = _resolved_archive_path(archive_path)
    source_hash = _sha256(source_path)[:16]
    filename = (
        f"{PREPROCESSING_VERSION}_{source_hash}_t{time_seq}_y{y_seq}_windows.pt"
    )
    return _PROCESSED_ROOT / filename


def _atomic_torch_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _torch_load(path):
    """Load a trusted local cache across old and new PyTorch releases."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # ``weights_only`` was added after the PyTorch version used by some
        # installations of the original Stable-Neural-SDEs benchmark.
        return torch.load(path, map_location="cpu")


def _load_or_build_windows(time_seq, y_seq, archive_path=None):
    path = _window_cache_path(time_seq, y_seq, archive_path)
    if path.is_file():
        return _torch_load(path)
    windows = _build_window_splits(time_seq, y_seq, archive_path)
    _atomic_torch_save(windows, path)
    return windows


def _validate_corruption(missing_rate, noise_level, missing_pattern):
    if not 0.0 <= float(missing_rate) < 1.0:
        raise ValueError(
            f"missing_rate must satisfy 0 <= rate < 1; got {missing_rate}."
        )
    if float(noise_level) < 0.0:
        raise ValueError(f"noise_level must be non-negative; got {noise_level}.")
    if missing_pattern not in {"random", "block", "tail"}:
        raise ValueError(
            "missing_pattern must be one of 'random', 'block', or 'tail'; "
            f"got {missing_pattern!r}."
        )


def _apply_missingness(inputs, missing_rate, missing_pattern, corruption_seed):
    corrupted = inputs.clone()
    num_examples, num_times, _ = corrupted.shape
    num_removed = int(num_times * float(missing_rate))
    if num_removed == 0:
        return corrupted

    generator = torch.Generator().manual_seed(int(corruption_seed))
    if missing_pattern == "random":
        for example in corrupted:
            removed = torch.randperm(num_times, generator=generator)[:num_removed]
            example[removed, :] = float("nan")
    elif missing_pattern == "block":
        locations = torch.rand(num_examples, generator=generator)
        max_start = num_times - num_removed
        starts = torch.floor(locations * (max_start + 1)).to(torch.long)
        offsets = torch.arange(num_removed)
        for example_index, start in enumerate(starts):
            corrupted[example_index, start + offsets, :] = float("nan")
    else:
        corrupted[:, num_times - num_removed :, :] = float("nan")
    return corrupted


def _finite_channel_std(train_inputs):
    values = []
    for channel in range(train_inputs.size(-1)):
        finite = train_inputs[..., channel]
        finite = finite[torch.isfinite(finite)]
        if finite.numel() < 2:
            values.append(torch.tensor(1.0, dtype=train_inputs.dtype))
        else:
            values.append(finite.std(unbiased=False).clamp_min(1e-6))
    return torch.stack(values)


def _add_scaled_input_noise(inputs, channel_std, noise_level, corruption_seed):
    if float(noise_level) == 0.0:
        return inputs
    generator = torch.Generator().manual_seed(
        int(corruption_seed) + _NOISE_SEED_OFFSET
    )
    noise = torch.randn(inputs.shape, generator=generator, dtype=inputs.dtype)
    noise = noise * float(noise_level) * channel_std.view(1, 1, -1)
    observed = torch.isfinite(inputs)
    return torch.where(observed, inputs + noise, inputs)


def _append_time_channel(inputs):
    times = torch.arange(1, inputs.size(1) + 1, dtype=inputs.dtype)
    times = times.view(1, -1, 1).expand(inputs.size(0), -1, -1)
    return torch.cat((times, inputs), dim=-1)


def _spline_coefficients(times, inputs, chunk_size=2048):
    # Keep dataset construction CPU-only.  The training loop transfers each
    # batch to its requested device.
    import controldiffeq

    pieces = [[], [], [], []]
    for start in range(0, inputs.size(0), chunk_size):
        coefficients = controldiffeq.natural_cubic_spline_coeffs(
            times, inputs[start : start + chunk_size]
        )
        for destination, coefficient in zip(pieces, coefficients):
            destination.append(coefficient.cpu())
    return tuple(torch.cat(part, dim=0) for part in pieces)


def _coefficient_cache_path(
    append_time, time_seq, y_seq, missing_rate, archive_path=None
):
    source_path = _resolved_archive_path(archive_path)
    source_hash = _sha256(source_path)[:16]
    suffix = "time1" if append_time else "time0"
    missing = format(float(missing_rate), ".8g").replace(".", "p")
    return _PROCESSED_ROOT / (
        f"{PREPROCESSING_VERSION}_{source_hash}_t{time_seq}_y{y_seq}_"
        f"m{missing}_{suffix}_coefficients.pt"
    )


def _build_coefficients(
    append_time, time_seq, y_seq, missing_rate, archive_path=None
):
    _validate_corruption(missing_rate, 0.0, "random")
    windows = _load_or_build_windows(time_seq, y_seq, archive_path)
    times = torch.arange(time_seq, dtype=torch.float32)
    output = {"times": times, "metadata": windows["metadata"]}
    offset = 0
    for split_name in ("train", "val", "test"):
        split = windows[split_name]
        inputs = _apply_missingness(
            split["inputs"],
            missing_rate=missing_rate,
            missing_pattern="random",
            corruption_seed=_DEFAULT_MISSING_SEED + offset,
        )
        offset += inputs.size(0)
        if append_time:
            inputs = _append_time_channel(inputs)
        output[split_name] = {
            "coefficients": _spline_coefficients(times, inputs),
            "targets": split["targets"],
            "asset_ids": split["asset_ids"],
            "target_start_ordinals": split["target_start_ordinals"],
        }
    return output


def _load_or_build_coefficients(
    append_time, time_seq, y_seq, missing_rate, archive_path=None
):
    path = _coefficient_cache_path(
        append_time, time_seq, y_seq, missing_rate, archive_path
    )
    if path.is_file():
        return _torch_load(path)
    coefficients = _build_coefficients(
        append_time, time_seq, y_seq, missing_rate, archive_path
    )
    _atomic_torch_save(coefficients, path)
    return coefficients


def _make_loader(split, times, batch_size, *, shuffle, loader_seed=0):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    final_index = torch.full(
        (split["targets"].size(0),), times.numel() - 1, dtype=torch.long
    )
    dataset = torch.utils.data.TensorDataset(
        *split["coefficients"], split["targets"], final_index
    )
    # Sidecar metadata does not change the batch tuple expected by the legacy
    # training loop.
    dataset.asset_ids = split["asset_ids"]
    dataset.target_start_ordinals = split["target_start_ordinals"]

    kwargs = dict(
        dataset=dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )
    if shuffle:
        kwargs["generator"] = torch.Generator().manual_seed(int(loader_seed))
    return torch.utils.data.DataLoader(**kwargs)


def get_data(
    batch_size,
    missing_rate,
    append_time,
    time_seq,
    y_seq,
    noise_std=0.0,
    loader_seed=0,
):
    """Return baseline train/validation/test loaders for this dataset."""
    if float(noise_std) != 0.0:
        raise ValueError(
            "get_data() never adds evaluation noise. Use "
            "get_stress_test_dataloader() after training."
        )
    cached = _load_or_build_coefficients(
        append_time, time_seq, y_seq, missing_rate
    )
    times = cached["times"]
    return (
        times,
        _make_loader(
            cached["train"], times, batch_size, shuffle=True, loader_seed=loader_seed
        ),
        _make_loader(cached["val"], times, batch_size, shuffle=False),
        _make_loader(cached["test"], times, batch_size, shuffle=False),
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
    """Return a test-only corrupted view with unchanged targets/order."""
    _validate_corruption(missing_rate, noise_level, missing_pattern)
    windows = _load_or_build_windows(time_seq, y_seq)
    train_inputs = windows["train"]["inputs"]
    test = windows["test"]
    inputs = _apply_missingness(
        test["inputs"], missing_rate, missing_pattern, corruption_seed
    )
    inputs = _add_scaled_input_noise(
        inputs,
        channel_std=_finite_channel_std(train_inputs),
        noise_level=noise_level,
        corruption_seed=corruption_seed,
    )
    if append_time:
        inputs = _append_time_channel(inputs)
    times = torch.arange(time_seq, dtype=torch.float32)
    split = {
        "coefficients": _spline_coefficients(times, inputs),
        "targets": test["targets"],
        "asset_ids": test["asset_ids"],
        "target_start_ordinals": test["target_start_ordinals"],
    }
    return times, _make_loader(split, times, batch_size, shuffle=False)


def validate_source(time_seq=None, y_seq=None):
    """Validate the bundled archive and optionally report exact window counts."""
    path = _resolved_archive_path()
    assets = _load_assets(path)
    result = {
        "dataset_name": "crypto",
        "archive": str(path),
        "archive_sha256": _sha256(path),
        "asset_count": len(assets),
        "feature_names": list(FEATURE_NAMES),
        "input_features": NUM_INPUT_FEATURES,
        "output_features": NUM_OUTPUT_FEATURES,
        "first_date": min(asset.dates[0] for asset in assets).isoformat(),
        "last_date": max(asset.dates[-1] for asset in assets).isoformat(),
    }
    if time_seq is not None or y_seq is not None:
        if time_seq is None or y_seq is None:
            raise ValueError("time_seq and y_seq must be supplied together.")
        windows = _load_or_build_windows(time_seq, y_seq)
        result["window_counts"] = windows["metadata"]["window_counts"]
        result["split"] = windows["metadata"]["split"]
    return result


def dataset_metadata(time_seq, y_seq):
    """JSON-compatible preprocessing provenance for benchmark results."""
    return _load_or_build_windows(time_seq, y_seq)["metadata"]
