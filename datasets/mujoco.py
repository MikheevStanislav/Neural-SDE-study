"""MuJoCo forecasting dataset and test-only corruption builders.

The public ``get_data`` function preserves the original training/validation/
test preprocessing and cache layout. ``get_stress_test_dataloader`` creates
alternative views of the same held-out test windows. It applies missingness
and scaled Gaussian noise to raw inputs before recomputing spline
coefficients; targets and model weights are left unchanged.
"""

import pathlib

import numpy as np
import torch

import controldiffeq
from . import common


here = pathlib.Path(__file__).resolve().parent

NUM_FEATURES = 14

_DEFAULT_MISSING_SEED = 56789
_NOISE_SEED_OFFSET = 1_000_003


def _validate_window_lengths(time_seq, y_seq, trajectory_length):
    if time_seq < 2:
        raise ValueError("time_seq must be at least 2 for cubic interpolation.")
    if y_seq <= 0:
        raise ValueError("y_seq must be positive.")

    # Intentionally preserve the repository's original window count:
    # range(trajectory_length - time_seq - y_seq), without a trailing +1.
    if trajectory_length - time_seq - y_seq <= 0:
        raise ValueError(
            "time_seq + y_seq must be smaller than the trajectory length; "
            f"got {time_seq} + {y_seq} for length {trajectory_length}."
        )


def _validate_corruption(missing_rate, noise_level, missing_pattern):
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError(
            f"missing_rate must satisfy 0 <= rate < 1; got {missing_rate}."
        )
    if noise_level < 0.0:
        raise ValueError(
            f"noise_level must be non-negative; got {noise_level}."
        )
    if missing_pattern not in {"random", "block", "tail"}:
        raise ValueError(
            "missing_pattern must be one of 'random', 'block', or 'tail'; "
            f"got {missing_pattern!r}."
        )


def _make_forecasting_windows(time_seq, y_seq):
    """Return the raw windows used by the original forecasting benchmark."""
    data_path = here / "mujoco.npy"
    trajectories = torch.from_numpy(np.load(data_path)).clone()

    if trajectories.ndim != 3:
        raise ValueError(
            "mujoco.npy must have shape [trajectories, time, channels]; "
            f"got {tuple(trajectories.shape)}."
        )

    num_trajectories, trajectory_length, _ = trajectories.shape
    _validate_window_lengths(time_seq, y_seq, trajectory_length)

    num_starts = trajectory_length - time_seq - y_seq
    input_windows = []
    target_windows = []

    for trajectory_index in range(num_trajectories):
        trajectory = trajectories[trajectory_index]
        for start in range(num_starts):
            input_windows.append(trajectory[start:start + time_seq])
            target_windows.append(
                trajectory[start + time_seq:start + time_seq + y_seq]
            )

    return torch.stack(input_windows), torch.stack(target_windows)


def _apply_missingness(
    inputs,
    missing_rate,
    missing_pattern,
    corruption_seed,
    *,
    global_window_offset=0,
):
    """Remove complete observation times across every feature channel.

    ``global_window_offset`` makes random masks for held-out windows line up
    with the masks that would have been generated after iterating over the
    complete dataset. With the same seed, random masks are nested across
    missing rates because each window reuses one permutation and only the
    prefix length changes.
    """
    corrupted = inputs.clone()
    num_examples, num_times, _ = corrupted.shape
    num_removed = int(num_times * missing_rate)

    if num_removed == 0:
        return corrupted

    generator = torch.Generator().manual_seed(int(corruption_seed))

    if missing_pattern == "random":
        # Advance to the global position of the first requested test window.
        for _ in range(global_window_offset):
            torch.randperm(num_times, generator=generator)

        for example in corrupted:
            removed = torch.randperm(
                num_times,
                generator=generator,
            )[:num_removed]
            example[removed, :] = float("nan")

    elif missing_pattern == "block":
        # Draw one location variable per global window so that repeated calls
        # with the same seed remain paired across corruption conditions.
        locations = torch.rand(
            global_window_offset + num_examples,
            generator=generator,
        )[global_window_offset:]
        max_start = num_times - num_removed
        starts = torch.floor(locations * (max_start + 1)).to(torch.long)
        offsets = torch.arange(num_removed)

        for example_index, start in enumerate(starts):
            removed = start + offsets
            corrupted[example_index, removed, :] = float("nan")

    else:  # tail
        corrupted[:, num_times - num_removed:, :] = float("nan")

    return corrupted


def _finite_channel_std(train_inputs):
    """Compute finite per-channel scales from the uncorrupted train split."""
    channel_stds = []

    for channel_index in range(train_inputs.size(-1)):
        values = train_inputs[..., channel_index]
        values = values[torch.isfinite(values)]

        if values.numel() < 2:
            channel_std = torch.tensor(1.0, dtype=train_inputs.dtype)
        else:
            channel_std = values.std(unbiased=False).clamp_min(1e-6)
        channel_stds.append(channel_std)

    return torch.stack(channel_stds)


def _add_scaled_input_noise(
    inputs,
    channel_std,
    noise_level,
    corruption_seed,
):
    """Add Gaussian noise scaled by each training feature's standard deviation."""
    if noise_level == 0.0:
        return inputs

    generator = torch.Generator().manual_seed(
        int(corruption_seed) + _NOISE_SEED_OFFSET
    )
    standard_noise = torch.randn(
        inputs.shape,
        generator=generator,
        dtype=inputs.dtype,
        device=inputs.device,
    )
    scaled_noise = (
        float(noise_level)
        * standard_noise
        * channel_std.to(inputs).view(1, 1, -1)
    )

    observed = torch.isfinite(inputs)
    return torch.where(observed, inputs + scaled_noise, inputs)


def _append_time_channel(inputs):
    # Preserve the original preprocessing convention: the appended channel is
    # 1..T, whilst solver integration times returned below are 0..T-1.
    num_times = inputs.size(1)
    time_channel = torch.linspace(
        1,
        num_times,
        num_times,
        dtype=inputs.dtype,
        device=inputs.device,
    )
    time_channel = time_channel.view(1, num_times, 1)
    time_channel = time_channel.expand(inputs.size(0), -1, -1)
    return torch.cat([time_channel, inputs], dim=-1)


def _process_data(append_time, time_seq, missing_rate, y_seq):
    """Build the original cached train/validation/test benchmark data."""
    _validate_corruption(missing_rate, 0.0, "random")
    input_windows, target_windows = _make_forecasting_windows(time_seq, y_seq)
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
        / f"mujoco{time_seq}_{y_seq}_{missing_rate}{suffix}"
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
    """Return the original cached loaders used for model training."""
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

    The train split is used only to calculate per-channel noise scales. No
    train/validation loader is created, no cache is written, and no target is
    modified. Calling this function therefore cannot retrain either model.

    ``noise_level`` is a fraction of the raw train standard deviation for each
    channel: 0.05 means Gaussian noise with 5% of that channel's scale.
    """
    _validate_corruption(missing_rate, noise_level, missing_pattern)
    input_windows, target_windows = _make_forecasting_windows(time_seq, y_seq)

    full_len = input_windows.size(0)
    train_len = int(full_len * 0.7)
    test_start = int(full_len * 0.85)

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
