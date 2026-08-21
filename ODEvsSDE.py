"""Train MuJoCo ODE/SDE models once and run paired corruption stress tests.

This file is intended for the current server-side benchmark layout. It reuses
``mujoco.py`` and ``mujoco-sde.py`` for training, then evaluates their returned
models on test-only views produced by
``datasets.mujoco.get_stress_test_dataloader``. No model is retrained inside
the missing-rate/noise/repeat grid. Test evaluation also includes a
drift-only ablation of the trained SDE with ``g_theta=0``; the learned
diffusion remains active throughout training and model selection.
"""

import importlib.util
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

import common_sde
import datasets
import mujoco
from parse import parse_args


ODE_MODEL = None
MC_SAMPLES = None
MC_SEED = None
EXPLOSION_THRESHOLD = None
DEVICE = "cuda"
SELECTED_COORDINATE_INDICES = (0, 1, 2, 4, 5, 10, 11)
SELECTED_HORIZON_INDICES = (0, 1, 2, 4, 5, 10, 11)

# ODE/SDE prediction distances are evaluated in float64 and are never
# normalised. SDE predictive variance is reported separately.

SDE_MODELS = {
    "diffusionsde",
    "staticsde",
    "naivesde",
    "neurallsde",
    "neurallnsde",
    "neuralgsde",
}


def load_mujoco_sde_module():
    """Load ``mujoco-sde.py`` despite the hyphen in its filename."""
    path = Path(__file__).resolve().with_name("mujoco-sde.py")
    spec = importlib.util.spec_from_file_location("mujoco_sde", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@contextmanager
def preserve_random_state(device):
    """Keep a diagnostic solver call from changing later MC trajectories."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    device = torch.device(device)
    cuda_devices = []
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_devices = [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]

    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def prediction_only(output):
    # ``learnable_forecasting`` returns prediction plus two auxiliary losses.
    if isinstance(output, tuple):
        return output[0]
    return output


def check_shape(name, prediction, target):
    if prediction.shape != target.shape:
        raise ValueError(
            f"{name} shape {tuple(prediction.shape)} != target shape "
            f"{tuple(target.shape)}. Use --intensity false with the current "
            "forecasting implementation."
        )


def ode_sde_distance_values(sde_mean, ode_prediction):
    """Return E[z_sde] - z_ode and its unnormalised Euclidean distance."""
    sde_mean = sde_mean.to(dtype=torch.float64)
    ode_prediction = ode_prediction.to(dtype=torch.float64)

    difference = (sde_mean - ode_prediction).reshape(
        -1,
        sde_mean.size(-1),
    )
    distance_l2 = torch.linalg.vector_norm(
        difference,
        ord=2,
        dim=1,
    )
    return common_sde._AttrDict(
        difference_components=difference.cpu(),
        raw_distance_l2=distance_l2.cpu(),
    )


def _distribution_statistics(values):
    """Summarise one empirical metric distribution.

    Variance is the population variance over all collected test values. A
    non-finite value is reported and counted as outside the finite mean +/- 2
    standard-deviation interval.
    """
    values = values.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    value_count = values.numel()
    finite_mask = torch.isfinite(values)
    finite_values = values[finite_mask]
    finite_count = finite_values.numel()
    nonfinite_count = value_count - finite_count

    if finite_count == 0:
        mean = math.nan
        root_mean_square = math.nan
        variance = math.nan
        standard_deviation = math.nan
        finite_outside_count = 0
    else:
        mean_tensor = finite_values.mean()
        root_mean_square_tensor = finite_values.square().mean().sqrt()
        variance_tensor = finite_values.var(unbiased=False)
        standard_deviation_tensor = variance_tensor.sqrt()
        finite_outside_count = (
            (finite_values - mean_tensor).abs()
            > 2.0 * standard_deviation_tensor
        ).sum().item()
        mean = mean_tensor.item()
        root_mean_square = root_mean_square_tensor.item()
        variance = variance_tensor.item()
        standard_deviation = standard_deviation_tensor.item()

    return common_sde._AttrDict(
        mean=mean,
        root_mean_square=root_mean_square,
        variance=variance,
        standard_deviation=standard_deviation,
        outside_2std_count=finite_outside_count + nonfinite_count,
        value_count=value_count,
        finite_count=finite_count,
        nonfinite_count=nonfinite_count,
    )


def _per_dimension_statistics(values):
    if values.ndim != 2:
        raise ValueError(
            "Per-dimension values must have shape [values, dimensions]; "
            f"got {tuple(values.shape)}."
        )
    dimensions = []
    for dimension in range(values.size(1)):
        statistics = _distribution_statistics(values[:, dimension])
        statistics["dimension"] = dimension
        dimensions.append(statistics)
    return dimensions


def ode_sde_distance_statistics(values):
    """Summarise the gap between mean SDE and ODE forecasts."""
    difference = values.difference_components
    raw_l2 = values.raw_distance_l2

    if difference.ndim != 2:
        raise ValueError(
            "ODE/SDE differences must have shape [values, dimensions]; "
            f"got {tuple(difference.shape)}."
        )
    if raw_l2.ndim != 1 or raw_l2.size(0) != difference.size(0):
        raise ValueError(
            "Raw ODE/SDE distances do not match component differences: "
            f"components={tuple(difference.shape)}, l2={tuple(raw_l2.shape)}."
        )

    output_dimensions = difference.size(1)
    if output_dimensions < 1:
        raise ValueError("ODE/SDE differences must have at least one dimension.")
    rms_per_dimension = raw_l2 / math.sqrt(output_dimensions)

    return common_sde._AttrDict(
        container_name_note=(
            "The legacy JSON key relative_ode_sde_metrics is retained for "
            "compatibility; this block now contains distances, not the old "
            "division by z_ode."
        ),
        sde_expectation=(
            "empirical mean over valid, non-exploded Monte Carlo paths"
        ),
        aggregation_unit="evaluable example x forecast horizon",
        variance_estimator="population (N denominator)",
        output_dimensions=output_dimensions,
        raw_evaluable_vector_count=difference.size(0),
        raw_distance=common_sde._AttrDict(
            component_formula="|E[z_sde_j] - z_ode_j|",
            l2_formula=(
                "||E[z_sde] - z_ode||_2 / sqrt(output_dimensions)"
            ),
            l2_equivalent_formula=(
                "sqrt(mean_j((E[z_sde_j] - z_ode_j)^2))"
            ),
            normalization=(
                "Euclidean norm divided by sqrt(output_dimensions)"
            ),
            units="same units as each output channel",
            dimensions=_per_dimension_statistics(difference.abs()),
            l2_norm=_distribution_statistics(rms_per_dimension),
            unnormalized_l2_formula="||E[z_sde] - z_ode||_2",
            unnormalized_l2_norm=_distribution_statistics(raw_l2),
        ),
        signed_difference=common_sde._AttrDict(
            component_formula="E[z_sde_j] - z_ode_j",
            dimensions=_per_dimension_statistics(difference),
        ),
    )


def _l2_comparison_metrics_from_sums(
    ode_to_truth_squared_sum,
    sde_mean_to_truth_squared_sum,
    ode_to_sde_squared_sum,
    scalar_value_count,
    output_dimensions,
):
    """Build three directly comparable RMS-per-dimension distances.

    ``scalar_value_count`` is the original shared scalar for dense targets.
    For sparse (NaN) targets it is a per-comparison dict: truth distances
    are averaged over observed entries only, whilst the prediction-only
    ODE-vs-SDE distance keeps the dense count.
    """
    squared_sums = {
        "ode_to_truth": ode_to_truth_squared_sum,
        "sde_mean_to_truth": sde_mean_to_truth_squared_sum,
        "ode_to_sde": ode_to_sde_squared_sum,
    }
    if isinstance(scalar_value_count, dict):
        counts = scalar_value_count
    else:
        counts = {name: scalar_value_count for name in squared_sums}

    mean_squared = {
        name: (
            squared_sums[name] / counts[name]
            if counts[name] > 0
            else math.nan
        )
        for name in squared_sums
    }

    def root(value):
        return (
            math.sqrt(max(value, 0.0))
            if math.isfinite(value)
            else math.nan
        )

    result = common_sde._AttrDict(
        formula="sqrt(mean_{example,horizon,dimension}(difference^2))",
        equivalent_vector_formula=(
            "RMS of ||difference||_2 / sqrt(output_dimensions)"
        ),
        units="same units as each output channel",
        sde_expectation=(
            "empirical mean over valid, non-exploded Monte Carlo paths"
        ),
        population=(
            "paired examples for which the SDE has at least one valid path"
        ),
        output_dimensions=output_dimensions,
        scalar_value_count=scalar_value_count,
        ode_to_truth=root(mean_squared["ode_to_truth"]),
        sde_mean_to_truth=root(mean_squared["sde_mean_to_truth"]),
        ode_to_sde=root(mean_squared["ode_to_sde"]),
        mean_squared_components=common_sde._AttrDict(mean_squared),
    )
    if isinstance(scalar_value_count, dict):
        result.sparse_target_note = (
            "Targets contained unobserved (NaN) entries; each comparison "
            "is averaged over its own observed-entry count."
        )
        result.scalar_value_counts = common_sde._AttrDict(counts)
    return result


def _available_indices(requested_indices, size):
    available = [index for index in requested_indices if index < size]
    unavailable = [index for index in requested_indices if index >= size]
    return available, unavailable


def _sde_coordinate_metrics_from_sums(
    loss_by_dimension_squared_sum,
    loss_value_count_per_dimension,
    variance_by_horizon_and_dimension_sum,
    variance_evaluable_examples,
    forecast_horizons,
    output_dimensions,
):
    """Build loss/MC-variance breakdowns without changing their population.

    ``loss_value_count_per_dimension`` is the original shared scalar for
    dense targets. For sparse (NaN) targets it is a per-dimension tensor of
    observed-entry counts, and dimensions without observations report NaN.
    """
    if torch.is_tensor(loss_value_count_per_dimension):
        counts = loss_value_count_per_dimension.to(
            dtype=torch.float64,
            device="cpu",
        )
        loss_by_dimension = torch.where(
            counts > 0,
            loss_by_dimension_squared_sum / counts.clamp_min(1),
            torch.full_like(loss_by_dimension_squared_sum, math.nan),
        )
    elif loss_value_count_per_dimension > 0:
        loss_by_dimension = (
            loss_by_dimension_squared_sum
            / loss_value_count_per_dimension
        )
    else:
        loss_by_dimension = torch.full(
            (output_dimensions,),
            math.nan,
            dtype=torch.float64,
        )

    if variance_evaluable_examples > 0:
        variance_matrix = (
            variance_by_horizon_and_dimension_sum
            / variance_evaluable_examples
        )
    else:
        variance_matrix = torch.full(
            (forecast_horizons, output_dimensions),
            math.nan,
            dtype=torch.float64,
        )

    available_dimensions, unavailable_dimensions = _available_indices(
        SELECTED_COORDINATE_INDICES,
        output_dimensions,
    )
    available_horizons, unavailable_horizons = _available_indices(
        SELECTED_HORIZON_INDICES,
        forecast_horizons,
    )

    variance_by_dimension = variance_matrix.mean(dim=0)
    variance_by_horizon = variance_matrix.mean(dim=1)

    return common_sde._AttrDict(
        sde_point_forecast="empirical Monte Carlo mean E[z_sde]",
        loss_formula=(
            "mean_{example,horizon}((E[z_sde_j] - true_j)^2)"
        ),
        prediction_variance_formula=(
            "mean_{example,horizon}(sample_variance_MC(z_sde_j))"
        ),
        prediction_variance_by_dimension_formula=(
            "mean_{example,horizon}(sample_variance_MC(z_sde_j))"
        ),
        prediction_variance_by_horizon_formula=(
            "mean_{example,dimension}(sample_variance_MC(z_sde[h,j]))"
        ),
        prediction_variance_by_horizon_and_dimension_formula=(
            "mean_{example}(sample_variance_MC(z_sde[h,j]))"
        ),
        prediction_variance_estimator=(
            "unbiased sample variance over valid MC paths (M-1 denominator), "
            "then population mean over evaluable examples"
        ),
        forecast_horizons=forecast_horizons,
        output_dimensions=output_dimensions,
        loss_value_count_per_dimension=loss_value_count_per_dimension,
        variance_evaluable_examples=variance_evaluable_examples,
        mean_prediction_loss_by_dimension=loss_by_dimension.tolist(),
        prediction_variance_by_dimension=variance_by_dimension.tolist(),
        prediction_variance_by_horizon=variance_by_horizon.tolist(),
        prediction_variance_by_horizon_and_dimension=(
            variance_matrix.tolist()
        ),
        selected_coordinate_indices_requested=list(
            SELECTED_COORDINATE_INDICES
        ),
        selected_coordinate_indices_available=available_dimensions,
        unavailable_coordinate_indices=unavailable_dimensions,
        selected_coordinate_variance_by_horizon=[
            common_sde._AttrDict(
                dimension=dimension,
                values=variance_matrix[:, dimension].tolist(),
            )
            for dimension in available_dimensions
        ],
        selected_horizon_indices_requested=list(SELECTED_HORIZON_INDICES),
        selected_horizon_indices_available=available_horizons,
        unavailable_horizon_indices=unavailable_horizons,
        selected_horizon_variance_by_dimension=[
            common_sde._AttrDict(
                horizon=horizon,
                values=variance_matrix[horizon].tolist(),
                mean_over_dimensions=variance_by_horizon[horizon].item(),
            )
            for horizon in available_horizons
        ],
    )


def _zero_diffusion_metrics_from_sums(
    squared_error_sums,
    squared_error_by_horizon_sums,
    scalar_value_count,
    horizon_scalar_value_count,
    paired_evaluable_examples,
    zero_diffusion_valid_examples,
    zero_diffusion_failed_examples,
    zero_diffusion_explosion_count,
    zero_diffusion_attempted_examples,
    zero_diffusion_solver_failure_count,
):
    def mse(name):
        if scalar_value_count == 0:
            return math.nan
        return squared_error_sums[name] / scalar_value_count

    horizon_sums = {
        name: torch.as_tensor(
            squared_error_by_horizon_sums[name],
            dtype=torch.float64,
            device="cpu",
        ).reshape(-1)
        for name in ("sde_mean", "truth", "ode")
    }
    horizon_shapes = {tuple(values.shape) for values in horizon_sums.values()}
    if len(horizon_shapes) != 1:
        raise ValueError(
            "Zero-diffusion horizon squared-error sums have different "
            f"shapes: {sorted(str(shape) for shape in horizon_shapes)}"
        )

    forecast_horizons = next(iter(horizon_sums.values())).numel()
    expected_scalar_value_count = (
        horizon_scalar_value_count * forecast_horizons
    )
    if scalar_value_count != expected_scalar_value_count:
        raise ValueError(
            "Zero-diffusion scalar counts are inconsistent with the "
            "horizon counts: scalar_value_count={}, "
            "horizon_scalar_value_count={}, forecast_horizons={}.".format(
                scalar_value_count,
                horizon_scalar_value_count,
                forecast_horizons,
            )
        )

    def mse_by_horizon(name):
        if horizon_scalar_value_count == 0:
            return [math.nan] * forecast_horizons
        return (
            horizon_sums[name] / horizon_scalar_value_count
        ).tolist()

    return common_sde._AttrDict(
        enabled_during_training=False,
        test_only=True,
        deterministic=True,
        brownian_path_independent=True,
        description=(
            "Same trained SDE initialization, drift, readout, solver and "
            "step size, evaluated once with g_theta(z,t)=0."
        ),
        sde_reference="empirical mean over valid non-exploded MC paths",
        population=(
            "paired examples with a finite non-exploded zero-diffusion "
            "forecast and at least one valid full-SDE path"
        ),
        formula="mean_{example,horizon,dimension}((prediction_a-prediction_b)^2)",
        horizon_formula=(
            "mean_{example,dimension}((prediction_a[:,h,:]-"
            "prediction_b[:,h,:])^2)"
        ),
        horizon_axis_note=(
            "The current forecasting heads expose output-sequence indices. "
            "These are retained under the repository's historical "
            "by_horizon naming."
        ),
        scalar_value_count=scalar_value_count,
        horizon_scalar_value_count=horizon_scalar_value_count,
        forecast_horizons=forecast_horizons,
        paired_evaluable_examples=paired_evaluable_examples,
        zero_diffusion_valid_examples=zero_diffusion_valid_examples,
        zero_diffusion_failed_examples=zero_diffusion_failed_examples,
        zero_diffusion_invalid_example_count=(
            zero_diffusion_failed_examples
        ),
        zero_diffusion_explosion_count=zero_diffusion_explosion_count,
        zero_diffusion_explosion_definition=(
            "non-finite output or max absolute output above the configured "
            "explosion threshold; numerical solver failures are included"
        ),
        zero_diffusion_attempted_examples=zero_diffusion_attempted_examples,
        zero_diffusion_solver_failure_count=(
            zero_diffusion_solver_failure_count
        ),
        squared_error_sums=common_sde._AttrDict(squared_error_sums),
        squared_error_by_horizon_sums=common_sde._AttrDict(
            {
                name: values.tolist()
                for name, values in horizon_sums.items()
            }
        ),
        mse_to_sde_mean=mse("sde_mean"),
        mse_to_truth=mse("truth"),
        mse_to_ode=mse("ode"),
        mse_to_sde_mean_by_horizon=mse_by_horizon("sde_mean"),
        mse_to_truth_by_horizon=mse_by_horizon("truth"),
        mse_to_ode_by_horizon=mse_by_horizon("ode"),
    )


def _zero_diffusion_metrics_from_masked_sums(
    squared_error_sums,
    squared_error_by_horizon_sums,
    scalar_value_counts,
    horizon_scalar_value_counts,
    paired_evaluable_examples,
    zero_diffusion_valid_examples,
    zero_diffusion_failed_examples,
    zero_diffusion_explosion_count,
    zero_diffusion_attempted_examples,
    zero_diffusion_solver_failure_count,
    forecast_horizons,
):
    """Sparse-target counterpart of ``_zero_diffusion_metrics_from_sums``.

    Every comparison keeps its own observed-entry counts: the truth
    comparison is averaged over finite target entries only, whilst the two
    prediction-vs-prediction comparisons remain dense.
    """
    horizon_sums = {
        name: torch.as_tensor(
            squared_error_by_horizon_sums[name],
            dtype=torch.float64,
            device="cpu",
        ).reshape(-1)
        for name in ("sde_mean", "truth", "ode")
    }
    horizon_counts = {
        name: torch.as_tensor(
            horizon_scalar_value_counts[name],
            dtype=torch.float64,
            device="cpu",
        ).reshape(-1)
        for name in ("sde_mean", "truth", "ode")
    }

    def mse(name):
        count = scalar_value_counts[name]
        if count == 0:
            return math.nan
        return squared_error_sums[name] / count

    def mse_by_horizon(name):
        counts = horizon_counts[name]
        if not bool((counts > 0).any()):
            return [math.nan] * forecast_horizons
        values = torch.where(
            counts > 0,
            horizon_sums[name] / counts.clamp_min(1),
            torch.full_like(horizon_sums[name], math.nan),
        )
        return values.tolist()

    return common_sde._AttrDict(
        enabled_during_training=False,
        test_only=True,
        deterministic=True,
        brownian_path_independent=True,
        description=(
            "Same trained SDE initialization, drift, readout, solver and "
            "step size, evaluated once with g_theta(z,t)=0."
        ),
        sparse_target_note=(
            "Targets contained unobserved (NaN) entries; the truth "
            "comparison is averaged over observed entries only."
        ),
        sde_reference="empirical mean over valid non-exploded MC paths",
        population=(
            "paired examples with a finite non-exploded zero-diffusion "
            "forecast and at least one valid full-SDE path"
        ),
        formula="mean_{example,horizon,dimension}((prediction_a-prediction_b)^2)",
        horizon_formula=(
            "mean_{example,dimension}((prediction_a[:,h,:]-"
            "prediction_b[:,h,:])^2)"
        ),
        horizon_axis_note=(
            "The current forecasting heads expose output-sequence indices. "
            "These are retained under the repository's historical "
            "by_horizon naming."
        ),
        scalar_value_counts=common_sde._AttrDict(scalar_value_counts),
        horizon_scalar_value_counts=common_sde._AttrDict(
            {
                name: counts.tolist()
                for name, counts in horizon_counts.items()
            }
        ),
        forecast_horizons=forecast_horizons,
        paired_evaluable_examples=paired_evaluable_examples,
        zero_diffusion_valid_examples=zero_diffusion_valid_examples,
        zero_diffusion_failed_examples=zero_diffusion_failed_examples,
        zero_diffusion_invalid_example_count=(
            zero_diffusion_failed_examples
        ),
        zero_diffusion_explosion_count=zero_diffusion_explosion_count,
        zero_diffusion_explosion_definition=(
            "non-finite output or max absolute output above the configured "
            "explosion threshold; numerical solver failures are included"
        ),
        zero_diffusion_attempted_examples=zero_diffusion_attempted_examples,
        zero_diffusion_solver_failure_count=(
            zero_diffusion_solver_failure_count
        ),
        squared_error_sums=common_sde._AttrDict(squared_error_sums),
        squared_error_by_horizon_sums=common_sde._AttrDict(
            {
                name: values.tolist()
                for name, values in horizon_sums.items()
            }
        ),
        mse_to_sde_mean=mse("sde_mean"),
        mse_to_truth=mse("truth"),
        mse_to_ode=mse("ode"),
        mse_to_sde_mean_by_horizon=mse_by_horizon("sde_mean"),
        mse_to_truth_by_horizon=mse_by_horizon("truth"),
        mse_to_ode_by_horizon=mse_by_horizon("ode"),
    )


def numerical_solver_failure(error):
    message = str(error).lower()
    if "out of memory" in message:
        return False
    markers = (
        "underflow",
        "overflow",
        "non-finite",
        "nonfinite",
        "nan",
        "infinity",
        "step size",
        "failed to converge",
    )
    return any(marker in message for marker in markers)


def sde_monte_carlo_batch(
    model,
    times,
    coeffs,
    lengths,
    target,
    solver_kwargs,
):
    predictions = []
    solver_failure_count = 0

    for _ in range(MC_SAMPLES):
        try:
            prediction = prediction_only(
                model(times, coeffs, lengths, **solver_kwargs)
            )
            check_shape("SDE", prediction, target)
        except (RuntimeError, AssertionError, FloatingPointError) as error:
            if not numerical_solver_failure(error):
                raise
            # A failed batch solve invalidates every example in this draw.
            prediction = torch.full_like(target, float("nan"))
            solver_failure_count += target.size(0)
        predictions.append(prediction)

    samples = torch.stack(predictions, dim=0)
    flat_samples = samples.flatten(start_dim=2)

    finite = torch.isfinite(flat_samples).all(dim=2)
    max_abs = torch.nan_to_num(
        flat_samples.abs(),
        nan=math.inf,
        posinf=math.inf,
        neginf=math.inf,
    ).amax(dim=2)
    valid_paths = finite & (max_abs <= EXPLOSION_THRESHOLD)

    # Invalid paths are excluded from mean/variance and counted separately.
    expanded_valid = valid_paths[..., None, None]
    safe_samples = torch.where(
        expanded_valid,
        samples,
        torch.zeros_like(samples),
    )
    valid_count = valid_paths.sum(dim=0)

    mean = safe_samples.sum(dim=0) / valid_count.clamp_min(1).to(
        samples.dtype
    )[:, None, None]

    centered = torch.where(
        expanded_valid,
        samples - mean.unsqueeze(0),
        torch.zeros_like(samples),
    )
    variance = centered.square().sum(dim=0) / (
        (valid_count - 1)
        .clamp_min(1)
        .to(samples.dtype)[:, None, None]
    )
    variance = torch.where(
        (valid_count > 1)[:, None, None],
        variance,
        torch.zeros_like(variance),
    )

    return common_sde._AttrDict(
        mean=mean,
        variance=variance,
        valid_count=valid_count,
        explosion_count=(~valid_paths).sum().item(),
        attempted_path_count=valid_paths.numel(),
        solver_failure_count=solver_failure_count,
    )


def sde_zero_diffusion_batch(
    model,
    times,
    coeffs,
    lengths,
    target,
    solver_kwargs,
):
    """Evaluate the trained SDE once with diffusion disabled at test time."""
    solver_failure_count = 0
    with preserve_random_state(times.device):
        try:
            prediction = prediction_only(
                model(
                    times,
                    coeffs,
                    lengths,
                    zero_diffusion=True,
                    **solver_kwargs,
                )
            )
            check_shape("SDE(g=0)", prediction, target)
        except (RuntimeError, AssertionError, FloatingPointError) as error:
            if not numerical_solver_failure(error):
                raise
            prediction = torch.full_like(target, float("nan"))
            solver_failure_count = target.size(0)

    flattened = prediction.flatten(start_dim=1)
    finite = torch.isfinite(flattened).all(dim=1)
    max_abs = torch.nan_to_num(
        flattened.abs(),
        nan=math.inf,
        posinf=math.inf,
        neginf=math.inf,
    ).amax(dim=1)
    valid = finite & (max_abs <= EXPLOSION_THRESHOLD)

    return common_sde._AttrDict(
        prediction=prediction,
        valid=valid,
        explosion_count=(~valid).sum().item(),
        attempted_example_count=valid.numel(),
        solver_failure_count=solver_failure_count,
    )


def _masked_example_mean(squared_error, target_finite):
    """Per-example mean squared error over observed (finite) target entries.

    PhysioNet targets are naturally sparse; examples contribute their mean
    over the entries that actually carry ground truth. Examples without any
    observed entry contribute 0 here, matching the convention that per
    horizon/dimension counts (computed separately) exclude them.
    """
    masked = torch.where(
        target_finite,
        squared_error,
        torch.zeros_like(squared_error),
    )
    counts = target_finite.flatten(start_dim=1).sum(dim=1)
    return masked.flatten(start_dim=1).sum(dim=1) / counts.clamp_min(1).to(
        squared_error.dtype
    )


def _masked_horizon_mean_sum(squared_error, target_finite):
    """Per-horizon loss accumulation over observed target entries.

    Returns the sum over examples of each example's finite-channel mean at
    every horizon, together with the number of examples that have at least
    one observed channel at that horizon (so the caller averages over
    contributing examples only).
    """
    masked = torch.where(
        target_finite,
        squared_error,
        torch.zeros_like(squared_error),
    )
    channel_counts = target_finite.sum(dim=2)  # [batch, horizon]
    per_example = masked.sum(dim=2) / channel_counts.clamp_min(1).to(
        squared_error.dtype
    )
    present = channel_counts > 0
    per_example = torch.where(
        present,
        per_example,
        torch.zeros_like(per_example),
    )
    return (
        per_example.sum(dim=0).cpu(),
        present.sum(dim=0).cpu().to(torch.float32),
    )


def full_test_dataloader(test_dataloader):
    """Evaluate every item; repository loaders normally use drop_last=True."""
    return torch.utils.data.DataLoader(
        test_dataloader.dataset,
        batch_size=min(
            test_dataloader.batch_size,
            len(test_dataloader.dataset),
        ),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


def evaluate(
    ode_result,
    sde_result,
    sde_method,
    test_dataloader=None,
):
    """Evaluate one fixed test view with paired Monte Carlo randomness."""
    ode_model = ode_result.model.to(DEVICE).eval()
    sde_model = sde_result.model.to(DEVICE).eval()
    times = ode_result.times.to(DEVICE)

    if test_dataloader is None:
        test_dataloader = ode_result.test_dataloader
    test_dataloader = full_test_dataloader(test_dataloader)

    ode_solver_kwargs = {"method": "rk4"}
    sde_solver_kwargs = {"method": sde_method}

    dataset_size = 0
    ode_loss_sum = 0.0
    sde_loss_sum = 0.0
    sde_evaluable_examples = 0
    sde_loss_by_dimension_squared_sum = None
    sde_loss_value_count_per_dimension = 0
    paired_ode_squared_error_sum = 0.0
    paired_sde_squared_error_sum = 0.0
    paired_ode_sde_squared_distance_sum = 0.0
    paired_scalar_value_count = 0
    # Masked (NaN-target) accumulators. They are only used when a dataset
    # such as PhysioNet carries unobserved forecast entries; dense datasets
    # keep using the original scalar accumulators above.
    saw_nan_targets = False
    paired_truth_scalar_count = 0
    paired_dense_scalar_count = 0
    ode_loss_by_horizon_count = None
    sde_loss_by_horizon_count = None
    zero_diffusion_masked_squared_error_sums = {
        "sde_mean": 0.0,
        "truth": 0.0,
        "ode": 0.0,
    }
    zero_diffusion_masked_scalar_value_counts = {
        "sde_mean": 0,
        "truth": 0,
        "ode": 0,
    }
    zero_diffusion_masked_squared_error_by_horizon_sums = {
        "sde_mean": None,
        "truth": None,
        "ode": None,
    }
    zero_diffusion_masked_scalar_value_by_horizon_counts = {
        "sde_mean": None,
        "truth": None,
        "ode": None,
    }
    variance_sum = 0.0
    variance_evaluable_examples = 0
    variance_by_horizon_sum = None
    variance_by_horizon_and_dimension_sum = None
    ode_loss_by_horizon_sum = None
    sde_loss_by_horizon_sum = None
    explosion_count = 0
    attempted_path_count = 0
    solver_failure_count = 0
    failed_example_count = 0
    insufficient_variance_example_count = 0
    zero_diffusion_squared_error_sums = {
        "sde_mean": 0.0,
        "truth": 0.0,
        "ode": 0.0,
    }
    zero_diffusion_squared_error_by_horizon_sums = {
        "sde_mean": None,
        "truth": None,
        "ode": None,
    }
    zero_diffusion_scalar_value_count = 0
    zero_diffusion_horizon_scalar_value_count = 0
    zero_diffusion_paired_evaluable_examples = 0
    zero_diffusion_valid_examples = 0
    zero_diffusion_failed_examples = 0
    zero_diffusion_explosion_count = 0
    zero_diffusion_attempted_examples = 0
    zero_diffusion_solver_failure_count = 0
    distance_value_chunks = {
        "difference_components": [],
        "raw_distance_l2": [],
    }
    output_channels = None
    forecast_horizons = None

    # Reset once per condition. Consecutive MC calls inside this evaluation
    # receive different paths, whilst different corruption conditions receive
    # the same sequence and are therefore paired.
    set_seed(MC_SEED)

    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(value.to(DEVICE) for value in batch)
            *coeffs, target, lengths = batch
            batch_size = target.size(0)
            output_channels = target.size(-1)
            target_finite = torch.isfinite(target)
            targets_all_finite = bool(target_finite.all())
            saw_nan_targets = saw_nan_targets or not targets_all_finite
            if forecast_horizons is None:
                forecast_horizons = target.size(1)
            elif forecast_horizons != target.size(1):
                raise ValueError(
                    "All evaluation batches must have the same forecast "
                    "horizon."
                )

            ode_prediction = prediction_only(
                ode_model(times, coeffs, lengths, **ode_solver_kwargs)
            )
            check_shape("ODE", ode_prediction, target)
            if not torch.isfinite(ode_prediction).all():
                raise FloatingPointError(
                    "ODE prediction contains NaN or Inf."
                )

            ode_squared_error = (ode_prediction - target).square()
            if targets_all_finite:
                ode_example_loss = ode_squared_error.flatten(
                    start_dim=1
                ).mean(dim=1)
            else:
                ode_example_loss = _masked_example_mean(
                    ode_squared_error,
                    target_finite,
                )
            ode_loss_sum += ode_example_loss.sum().item()
            if targets_all_finite:
                ode_batch_horizon = ode_squared_error.mean(dim=2).sum(
                    dim=0
                ).cpu()
                ode_batch_horizon_count = torch.full(
                    (target.size(1),),
                    float(batch_size),
                    dtype=torch.float32,
                )
            else:
                (
                    ode_batch_horizon,
                    ode_batch_horizon_count,
                ) = _masked_horizon_mean_sum(
                    ode_squared_error,
                    target_finite,
                )
            if ode_loss_by_horizon_sum is None:
                ode_loss_by_horizon_sum = ode_batch_horizon
                ode_loss_by_horizon_count = ode_batch_horizon_count
            else:
                ode_loss_by_horizon_sum += ode_batch_horizon
                ode_loss_by_horizon_count += ode_batch_horizon_count
            dataset_size += batch_size

            mc = sde_monte_carlo_batch(
                sde_model,
                times,
                coeffs,
                lengths,
                target,
                sde_solver_kwargs,
            )
            zero_diffusion = sde_zero_diffusion_batch(
                sde_model,
                times,
                coeffs,
                lengths,
                target,
                sde_solver_kwargs,
            )
            evaluable = mc.valid_count > 0
            variance_evaluable = mc.valid_count > 1
            zero_diffusion_paired = evaluable & zero_diffusion.valid

            explosion_count += mc.explosion_count
            attempted_path_count += mc.attempted_path_count
            solver_failure_count += mc.solver_failure_count
            failed_example_count += (~evaluable).sum().item()
            insufficient_variance_example_count += (
                (~variance_evaluable).sum().item()
            )
            zero_diffusion_valid_examples += (
                zero_diffusion.valid.sum().item()
            )
            zero_diffusion_failed_examples += (
                (~zero_diffusion.valid).sum().item()
            )
            zero_diffusion_explosion_count += (
                zero_diffusion.explosion_count
            )
            zero_diffusion_attempted_examples += (
                zero_diffusion.attempted_example_count
            )
            zero_diffusion_solver_failure_count += (
                zero_diffusion.solver_failure_count
            )

            if evaluable.any():
                sde_squared_error = (mc.mean - target).square()
                if targets_all_finite:
                    sde_example_loss = sde_squared_error.flatten(
                        start_dim=1
                    ).mean(dim=1)
                    masked_sde_squared_error = None
                else:
                    sde_example_loss = _masked_example_mean(
                        sde_squared_error,
                        target_finite,
                    )
                    masked_sde_squared_error = torch.where(
                        target_finite,
                        sde_squared_error,
                        torch.zeros_like(sde_squared_error),
                    )
                sde_loss_sum += sde_example_loss[evaluable].sum().item()
                evaluable_count = evaluable.sum().item()
                sde_evaluable_examples += evaluable_count

                if targets_all_finite:
                    batch_loss_by_dimension_sum = sde_squared_error[
                        evaluable
                    ].to(dtype=torch.float64).sum(dim=(0, 1)).cpu()
                else:
                    batch_loss_by_dimension_sum = masked_sde_squared_error[
                        evaluable
                    ].to(dtype=torch.float64).sum(dim=(0, 1)).cpu()
                if sde_loss_by_dimension_squared_sum is None:
                    sde_loss_by_dimension_squared_sum = (
                        batch_loss_by_dimension_sum
                    )
                else:
                    sde_loss_by_dimension_squared_sum += (
                        batch_loss_by_dimension_sum
                    )
                if targets_all_finite:
                    sde_loss_value_count_per_dimension += (
                        evaluable_count * target.size(1)
                    )
                else:
                    sde_loss_value_count_per_dimension += (
                        target_finite[evaluable]
                        .to(dtype=torch.float64)
                        .sum(dim=(0, 1))
                        .cpu()
                    )

                if targets_all_finite:
                    sde_batch_horizon = sde_squared_error[
                        evaluable
                    ].mean(dim=2).sum(dim=0).cpu()
                    sde_batch_horizon_count = torch.full(
                        (target.size(1),),
                        float(evaluable_count),
                        dtype=torch.float32,
                    )
                else:
                    (
                        sde_batch_horizon,
                        sde_batch_horizon_count,
                    ) = _masked_horizon_mean_sum(
                        sde_squared_error[evaluable],
                        target_finite[evaluable],
                    )
                if sde_loss_by_horizon_sum is None:
                    sde_loss_by_horizon_sum = sde_batch_horizon
                    sde_loss_by_horizon_count = sde_batch_horizon_count
                else:
                    sde_loss_by_horizon_sum += sde_batch_horizon
                    sde_loss_by_horizon_count += sde_batch_horizon_count

                if targets_all_finite:
                    paired_ode_squared_error_sum += ode_squared_error[
                        evaluable
                    ].sum().item()
                    paired_sde_squared_error_sum += sde_squared_error[
                        evaluable
                    ].sum().item()
                else:
                    paired_ode_squared_error_sum += torch.where(
                        target_finite,
                        ode_squared_error,
                        torch.zeros_like(ode_squared_error),
                    )[evaluable].sum().item()
                    paired_sde_squared_error_sum += masked_sde_squared_error[
                        evaluable
                    ].sum().item()
                paired_ode_sde_squared_distance_sum += (
                    mc.mean[evaluable] - ode_prediction[evaluable]
                ).square().sum().item()
                if targets_all_finite:
                    paired_scalar_value_count += target[evaluable].numel()
                else:
                    paired_truth_scalar_count += (
                        target_finite[evaluable].sum().item()
                    )
                    paired_dense_scalar_count += target[evaluable].numel()

                raw_distance_values = ode_sde_distance_values(
                    mc.mean[evaluable],
                    ode_prediction[evaluable],
                )
                distance_value_chunks["difference_components"].append(
                    raw_distance_values.difference_components
                )
                distance_value_chunks["raw_distance_l2"].append(
                    raw_distance_values.raw_distance_l2
                )

                if zero_diffusion_paired.any():
                    paired_zero_prediction = zero_diffusion.prediction[
                        zero_diffusion_paired
                    ]
                    paired_sde_mean = mc.mean[zero_diffusion_paired]
                    paired_target = target[zero_diffusion_paired]
                    paired_ode = ode_prediction[zero_diffusion_paired]

                    paired_zero_prediction = paired_zero_prediction.to(
                        dtype=torch.float64
                    )
                    zero_diffusion_comparisons = {
                        "sde_mean": paired_sde_mean.to(dtype=torch.float64),
                        "truth": paired_target.to(dtype=torch.float64),
                        "ode": paired_ode.to(dtype=torch.float64),
                    }
                    if targets_all_finite:
                        for name, other_prediction in (
                            zero_diffusion_comparisons.items()
                        ):
                            squared_error = (
                                paired_zero_prediction - other_prediction
                            ).square()
                            zero_diffusion_squared_error_sums[name] += (
                                squared_error.sum().item()
                            )
                            horizon_sum = squared_error.sum(
                                dim=(0, 2)
                            ).cpu()
                            if (
                                zero_diffusion_squared_error_by_horizon_sums[
                                    name
                                ]
                                is None
                            ):
                                zero_diffusion_squared_error_by_horizon_sums[
                                    name
                                ] = horizon_sum
                            else:
                                zero_diffusion_squared_error_by_horizon_sums[
                                    name
                                ] += horizon_sum
                        zero_diffusion_scalar_value_count += (
                            paired_target.numel()
                        )
                        zero_diffusion_horizon_scalar_value_count += (
                            paired_target.size(0) * paired_target.size(2)
                        )
                    else:
                        # Sparse targets: the comparison against truth is
                        # accumulated over observed entries only, with its
                        # own scalar and per-horizon counts. The two
                        # prediction-vs-prediction comparisons stay dense.
                        paired_finite = target_finite[zero_diffusion_paired]
                        for name, other_prediction in (
                            zero_diffusion_comparisons.items()
                        ):
                            squared_error = (
                                paired_zero_prediction - other_prediction
                            ).square()
                            if name == "truth":
                                masked_error = torch.where(
                                    paired_finite,
                                    squared_error,
                                    torch.zeros_like(squared_error),
                                )
                                scalar_count = int(
                                    paired_finite.sum().item()
                                )
                                horizon_count = (
                                    paired_finite.to(dtype=torch.float64)
                                    .sum(dim=(0, 2))
                                    .cpu()
                                )
                            else:
                                masked_error = squared_error
                                scalar_count = paired_target.numel()
                                horizon_count = torch.full(
                                    (paired_target.size(1),),
                                    float(
                                        paired_target.size(0)
                                        * paired_target.size(2)
                                    ),
                                    dtype=torch.float64,
                                )
                            zero_diffusion_masked_squared_error_sums[
                                name
                            ] += masked_error.sum().item()
                            zero_diffusion_masked_scalar_value_counts[
                                name
                            ] += scalar_count
                            horizon_sum = masked_error.sum(dim=(0, 2)).cpu()
                            if (
                                zero_diffusion_masked_squared_error_by_horizon_sums[
                                    name
                                ]
                                is None
                            ):
                                zero_diffusion_masked_squared_error_by_horizon_sums[
                                    name
                                ] = horizon_sum
                                zero_diffusion_masked_scalar_value_by_horizon_counts[
                                    name
                                ] = horizon_count
                            else:
                                zero_diffusion_masked_squared_error_by_horizon_sums[
                                    name
                                ] += horizon_sum
                                zero_diffusion_masked_scalar_value_by_horizon_counts[
                                    name
                                ] += horizon_count
                    zero_diffusion_paired_evaluable_examples += (
                        zero_diffusion_paired.sum().item()
                    )

            if variance_evaluable.any():
                example_variance = mc.variance.flatten(
                    start_dim=1
                ).mean(dim=1)
                variance_sum += example_variance[
                    variance_evaluable
                ].sum().item()
                variance_count = variance_evaluable.sum().item()
                variance_evaluable_examples += variance_count

                batch_variance_by_horizon = mc.variance.mean(dim=2)[
                    variance_evaluable
                ].sum(dim=0).cpu()
                if variance_by_horizon_sum is None:
                    variance_by_horizon_sum = batch_variance_by_horizon
                else:
                    variance_by_horizon_sum += batch_variance_by_horizon

                batch_variance_matrix_sum = mc.variance[
                    variance_evaluable
                ].to(dtype=torch.float64).sum(dim=0).cpu()
                if variance_by_horizon_and_dimension_sum is None:
                    variance_by_horizon_and_dimension_sum = (
                        batch_variance_matrix_sum
                    )
                else:
                    variance_by_horizon_and_dimension_sum += (
                        batch_variance_matrix_sum
                    )

    if dataset_size == 0:
        raise RuntimeError("The test dataloader is empty.")

    if sde_loss_by_dimension_squared_sum is None:
        sde_loss_by_dimension_squared_sum = torch.zeros(
            output_channels,
            dtype=torch.float64,
        )
    if variance_by_horizon_and_dimension_sum is None:
        variance_by_horizon_and_dimension_sum = torch.zeros(
            (forecast_horizons, output_channels),
            dtype=torch.float64,
        )
    for name, values in (
        zero_diffusion_squared_error_by_horizon_sums.items()
    ):
        if values is None:
            zero_diffusion_squared_error_by_horizon_sums[name] = (
                torch.zeros(forecast_horizons, dtype=torch.float64)
            )
    for name, values in (
        zero_diffusion_masked_squared_error_by_horizon_sums.items()
    ):
        if values is None:
            zero_diffusion_masked_squared_error_by_horizon_sums[name] = (
                torch.zeros(forecast_horizons, dtype=torch.float64)
            )
            zero_diffusion_masked_scalar_value_by_horizon_counts[name] = (
                torch.zeros(forecast_horizons, dtype=torch.float64)
            )

    sde_coordinate_metrics = _sde_coordinate_metrics_from_sums(
        loss_by_dimension_squared_sum=(
            sde_loss_by_dimension_squared_sum
        ),
        loss_value_count_per_dimension=(
            sde_loss_value_count_per_dimension
        ),
        variance_by_horizon_and_dimension_sum=(
            variance_by_horizon_and_dimension_sum
        ),
        variance_evaluable_examples=variance_evaluable_examples,
        forecast_horizons=forecast_horizons,
        output_dimensions=output_channels,
    )
    if saw_nan_targets:
        zero_diffusion_metrics = _zero_diffusion_metrics_from_masked_sums(
            squared_error_sums=(
                zero_diffusion_masked_squared_error_sums
            ),
            squared_error_by_horizon_sums=(
                zero_diffusion_masked_squared_error_by_horizon_sums
            ),
            scalar_value_counts=(
                zero_diffusion_masked_scalar_value_counts
            ),
            horizon_scalar_value_counts=(
                zero_diffusion_masked_scalar_value_by_horizon_counts
            ),
            paired_evaluable_examples=(
                zero_diffusion_paired_evaluable_examples
            ),
            zero_diffusion_valid_examples=zero_diffusion_valid_examples,
            zero_diffusion_failed_examples=zero_diffusion_failed_examples,
            zero_diffusion_explosion_count=zero_diffusion_explosion_count,
            zero_diffusion_attempted_examples=(
                zero_diffusion_attempted_examples
            ),
            zero_diffusion_solver_failure_count=(
                zero_diffusion_solver_failure_count
            ),
            forecast_horizons=forecast_horizons,
        )
    else:
        zero_diffusion_metrics = _zero_diffusion_metrics_from_sums(
            squared_error_sums=zero_diffusion_squared_error_sums,
            squared_error_by_horizon_sums=(
                zero_diffusion_squared_error_by_horizon_sums
            ),
            scalar_value_count=zero_diffusion_scalar_value_count,
            horizon_scalar_value_count=(
                zero_diffusion_horizon_scalar_value_count
            ),
            paired_evaluable_examples=(
                zero_diffusion_paired_evaluable_examples
            ),
            zero_diffusion_valid_examples=zero_diffusion_valid_examples,
            zero_diffusion_failed_examples=zero_diffusion_failed_examples,
            zero_diffusion_explosion_count=zero_diffusion_explosion_count,
            zero_diffusion_attempted_examples=(
                zero_diffusion_attempted_examples
            ),
            zero_diffusion_solver_failure_count=(
                zero_diffusion_solver_failure_count
            ),
        )

    ode_loss = ode_loss_sum / dataset_size
    ode_loss_by_horizon = (
        (ode_loss_by_horizon_sum / ode_loss_by_horizon_count).tolist()
        if ode_loss_by_horizon_sum is not None
        else []
    )
    sde_loss = (
        sde_loss_sum / sde_evaluable_examples
        if sde_evaluable_examples
        else math.nan
    )
    sde_loss_by_horizon = (
        (sde_loss_by_horizon_sum / sde_loss_by_horizon_count).tolist()
        if sde_loss_by_horizon_sum is not None
        and sde_evaluable_examples
        else []
    )
    sde_variance = (
        variance_sum / variance_evaluable_examples
        if variance_evaluable_examples
        else math.nan
    )
    variance_by_horizon = (
        sde_coordinate_metrics.prediction_variance_by_horizon
    )
    explosion_rate = (
        explosion_count / attempted_path_count
        if attempted_path_count
        else math.nan
    )

    distance_values = common_sde._AttrDict()
    for field, chunks in distance_value_chunks.items():
        if chunks:
            distance_values[field] = torch.cat(chunks, dim=0)
        elif field == "difference_components":
            distance_values[field] = torch.empty(
                (0, output_channels),
                dtype=torch.float64,
            )
        else:
            distance_values[field] = torch.empty(
                0,
                dtype=torch.float64,
            )
    distance_metrics = ode_sde_distance_statistics(
        distance_values,
    )
    if saw_nan_targets:
        # Truth-related distances share the observed-entry count; the
        # ODE-vs-SDE distance is prediction-only and stays dense.
        paired_counts = {
            "ode_to_truth": paired_truth_scalar_count,
            "sde_mean_to_truth": paired_truth_scalar_count,
            "ode_to_sde": paired_dense_scalar_count,
        }
    else:
        paired_counts = paired_scalar_value_count
    l2_comparison_metrics = _l2_comparison_metrics_from_sums(
        ode_to_truth_squared_sum=paired_ode_squared_error_sum,
        sde_mean_to_truth_squared_sum=paired_sde_squared_error_sum,
        ode_to_sde_squared_sum=paired_ode_sde_squared_distance_sum,
        scalar_value_count=paired_counts,
        output_dimensions=output_channels,
    )

    return common_sde._AttrDict(
        dataset_size=dataset_size,
        monte_carlo_samples=MC_SAMPLES,
        ode_metrics=common_sde._AttrDict(
            loss=ode_loss,
            loss_by_horizon=ode_loss_by_horizon,
        ),
        sde_metrics=common_sde._AttrDict(
            loss=sde_loss,
            loss_by_horizon=sde_loss_by_horizon,
            mean_prediction_loss_by_dimension=(
                sde_coordinate_metrics.mean_prediction_loss_by_dimension
            ),
            prediction_variance=sde_variance,
            prediction_std=(
                math.sqrt(max(sde_variance, 0.0))
                if math.isfinite(sde_variance)
                else math.nan
            ),
            prediction_variance_by_horizon=variance_by_horizon,
            prediction_variance_by_dimension=(
                sde_coordinate_metrics.prediction_variance_by_dimension
            ),
            prediction_variance_by_horizon_and_dimension=(
                sde_coordinate_metrics
                .prediction_variance_by_horizon_and_dimension
            ),
            coordinate_metrics=sde_coordinate_metrics,
            evaluable_examples=sde_evaluable_examples,
            variance_evaluable_examples=variance_evaluable_examples,
            failed_examples=failed_example_count,
            examples_with_fewer_than_two_valid_paths=(
                insufficient_variance_example_count
            ),
            explosion_count=explosion_count,
            attempted_path_count=attempted_path_count,
            explosion_rate=explosion_rate,
            solver_failure_count=solver_failure_count,
            explosion_threshold=EXPLOSION_THRESHOLD,
        ),
        sde_zero_diffusion_metrics=zero_diffusion_metrics,
        relative_ode_sde_metrics=distance_metrics,
        l2_comparison_metrics=l2_comparison_metrics,
        # Raw CPU values are retained only long enough to pool corruption
        # repeats exactly. They are never copied into the saved result JSON.
        _ode_sde_distance_values=distance_values,
    )


def _validate_stress_arguments(args):
    if args.corruption_repeats < 1:
        raise ValueError("--corruption_repeats must be at least 1.")
    if not args.test_missing_rates:
        raise ValueError("--test_missing_rates must not be empty.")
    if not args.input_noise_levels:
        raise ValueError("--input_noise_levels must not be empty.")

    all_missing_rates = [args.missing_rate, *args.test_missing_rates]
    for missing_rate in all_missing_rates:
        if not 0.0 <= missing_rate < 1.0:
            raise ValueError(
                "All missing rates must satisfy 0 <= rate < 1; "
                f"got {missing_rate}."
            )
    for noise_level in args.input_noise_levels:
        if noise_level < 0.0:
            raise ValueError(
                "All input noise levels must be non-negative; "
                f"got {noise_level}."
            )


def _is_reference_condition(args, missing_rate, noise_level):
    return math.isclose(missing_rate, args.missing_rate) and math.isclose(
        noise_level,
        0.0,
    )


def _relative_degradation(value, reference):
    """Return the original loss degradation relative to the reference view."""
    if not math.isfinite(value) or not math.isfinite(reference):
        return math.nan
    return value / max(reference, 1e-12) - 1.0


def _loss_relative_degradation_metrics(
    args,
    ode_relative_degradation,
    sde_relative_degradation,
    reference_ode_loss,
    reference_sde_loss,
):
    """Represent the restored loss metrics inside the combined metric block."""
    return common_sde._AttrDict(
        formula="condition_loss / max(reference_loss, 1e-12) - 1",
        reference_missing_rate=float(args.missing_rate),
        reference_input_noise_level=0.0,
        reference_ode_loss=reference_ode_loss,
        reference_sde_loss=reference_sde_loss,
        ode=_distribution_statistics(
            torch.tensor(
                [ode_relative_degradation],
                dtype=torch.float64,
            )
        ),
        sde=_distribution_statistics(
            torch.tensor(
                [sde_relative_degradation],
                dtype=torch.float64,
            )
        ),
    )


def _make_stress_record(
    args,
    repeat,
    corruption_seed,
    missing_rate,
    noise_level,
    comparison,
    reference,
):
    ode_loss = comparison.ode_metrics.loss
    sde_loss = comparison.sde_metrics.loss
    reference_ode_loss = reference.ode_metrics.loss
    reference_sde_loss = reference.sde_metrics.loss
    ode_relative_degradation = _relative_degradation(
        ode_loss,
        reference_ode_loss,
    )
    sde_relative_degradation = _relative_degradation(
        sde_loss,
        reference_sde_loss,
    )

    combined_relative_metrics = common_sde._AttrDict(
        comparison.relative_ode_sde_metrics.copy()
    )
    combined_relative_metrics.loss_relative_degradation = (
        _loss_relative_degradation_metrics(
            args=args,
            ode_relative_degradation=ode_relative_degradation,
            sde_relative_degradation=sde_relative_degradation,
            reference_ode_loss=reference_ode_loss,
            reference_sde_loss=reference_sde_loss,
        )
    )

    return common_sde._AttrDict(
        repeat=repeat,
        corruption_seed=corruption_seed,
        missing_rate=float(missing_rate),
        input_noise_level=float(noise_level),
        missing_pattern=args.missing_pattern,
        dataset_size=comparison.dataset_size,
        reference_missing_rate=float(args.missing_rate),
        reference_ode_loss=reference_ode_loss,
        reference_sde_loss=reference_sde_loss,
        ode_loss=ode_loss,
        ode_loss_by_horizon=comparison.ode_metrics.loss_by_horizon,
        ode_relative_degradation=ode_relative_degradation,
        sde_loss=sde_loss,
        sde_loss_by_horizon=comparison.sde_metrics.loss_by_horizon,
        sde_relative_degradation=sde_relative_degradation,
        sde_prediction_variance=(
            comparison.sde_metrics.prediction_variance
        ),
        sde_prediction_std=comparison.sde_metrics.prediction_std,
        sde_variance_by_horizon=(
            comparison.sde_metrics.prediction_variance_by_horizon
        ),
        sde_loss_by_dimension=(
            comparison.sde_metrics.mean_prediction_loss_by_dimension
        ),
        sde_variance_by_dimension=(
            comparison.sde_metrics.prediction_variance_by_dimension
        ),
        sde_variance_by_horizon_and_dimension=(
            comparison.sde_metrics
            .prediction_variance_by_horizon_and_dimension
        ),
        sde_coordinate_metrics=(
            comparison.sde_metrics.coordinate_metrics
        ),
        sde_zero_diffusion_metrics=(
            comparison.sde_zero_diffusion_metrics
        ),
        sde_zero_diffusion_mse_to_sde_mean=(
            comparison.sde_zero_diffusion_metrics.mse_to_sde_mean
        ),
        sde_zero_diffusion_mse_to_truth=(
            comparison.sde_zero_diffusion_metrics.mse_to_truth
        ),
        sde_zero_diffusion_mse_to_ode=(
            comparison.sde_zero_diffusion_metrics.mse_to_ode
        ),
        sde_zero_diffusion_mse_to_sde_mean_by_horizon=(
            comparison.sde_zero_diffusion_metrics
            .mse_to_sde_mean_by_horizon
        ),
        sde_zero_diffusion_mse_to_truth_by_horizon=(
            comparison.sde_zero_diffusion_metrics
            .mse_to_truth_by_horizon
        ),
        sde_zero_diffusion_mse_to_ode_by_horizon=(
            comparison.sde_zero_diffusion_metrics
            .mse_to_ode_by_horizon
        ),
        sde_explosion_count=comparison.sde_metrics.explosion_count,
        sde_attempted_path_count=(
            comparison.sde_metrics.attempted_path_count
        ),
        sde_explosion_rate=comparison.sde_metrics.explosion_rate,
        sde_solver_failure_count=(
            comparison.sde_metrics.solver_failure_count
        ),
        sde_failed_examples=comparison.sde_metrics.failed_examples,
        relative_ode_sde_metrics=combined_relative_metrics,
        l2_comparison_metrics=comparison.l2_comparison_metrics,
    )


def _finite_mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return mean, std


def _aggregate_l2_comparison_metrics(metrics):
    """Pool squared distances exactly across corruption repeats."""
    output_dimensions = metrics[0].output_dimensions
    if any(
        metric.output_dimensions != output_dimensions
        for metric in metrics
    ):
        raise ValueError(
            "Cannot aggregate L2 metrics with different output dimensions."
        )

    sparse_counts = isinstance(metrics[0].scalar_value_count, dict)

    squared_sums = {}
    for name in (
        "ode_to_truth",
        "sde_mean_to_truth",
        "ode_to_sde",
    ):
        if sparse_counts:
            squared_sums[name] = sum(
                metric.mean_squared_components[name]
                * metric.scalar_value_count[name]
                for metric in metrics
                if math.isfinite(metric.mean_squared_components[name])
            )
        else:
            squared_sums[name] = sum(
                metric.mean_squared_components[name]
                * metric.scalar_value_count
                for metric in metrics
                if math.isfinite(metric.mean_squared_components[name])
            )

    if sparse_counts:
        total_scalar_count = {
            name: sum(
                metric.scalar_value_count[name] for metric in metrics
            )
            for name in (
                "ode_to_truth",
                "sde_mean_to_truth",
                "ode_to_sde",
            )
        }
    else:
        total_scalar_count = sum(
            metric.scalar_value_count for metric in metrics
        )

    pooled = _l2_comparison_metrics_from_sums(
        ode_to_truth_squared_sum=squared_sums["ode_to_truth"],
        sde_mean_to_truth_squared_sum=(
            squared_sums["sde_mean_to_truth"]
        ),
        ode_to_sde_squared_sum=squared_sums["ode_to_sde"],
        scalar_value_count=total_scalar_count,
        output_dimensions=output_dimensions,
    )
    pooled["corruption_repeats"] = len(metrics)
    pooled["repeat_statistics"] = common_sde._AttrDict()
    for name in (
        "ode_to_truth",
        "sde_mean_to_truth",
        "ode_to_sde",
    ):
        mean, std = _finite_mean_std([metric[name] for metric in metrics])
        pooled.repeat_statistics[name] = common_sde._AttrDict(
            mean=mean,
            standard_deviation=std,
        )
    return pooled


def _aggregate_zero_diffusion_metrics(metrics):
    squared_error_sums = {
        name: sum(
            metric.squared_error_sums[name]
            for metric in metrics
        )
        for name in ("sde_mean", "truth", "ode")
    }
    horizon_lengths = {
        len(metric.mse_to_truth_by_horizon) for metric in metrics
    }
    if len(horizon_lengths) != 1:
        raise ValueError(
            "Cannot aggregate zero-diffusion horizon metrics with "
            f"different lengths: {sorted(horizon_lengths)}"
        )
    squared_error_by_horizon_sums = {
        name: torch.stack(
            [
                torch.as_tensor(
                    metric.squared_error_by_horizon_sums[name],
                    dtype=torch.float64,
                    device="cpu",
                )
                for metric in metrics
            ],
            dim=0,
        ).sum(dim=0)
        for name in ("sde_mean", "truth", "ode")
    }
    if "scalar_value_counts" in metrics[0]:
        pooled = _zero_diffusion_metrics_from_masked_sums(
            squared_error_sums=squared_error_sums,
            squared_error_by_horizon_sums=(
                squared_error_by_horizon_sums
            ),
            scalar_value_counts={
                name: sum(
                    metric.scalar_value_counts[name] for metric in metrics
                )
                for name in ("sde_mean", "truth", "ode")
            },
            horizon_scalar_value_counts={
                name: torch.stack(
                    [
                        torch.as_tensor(
                            metric.horizon_scalar_value_counts[name],
                            dtype=torch.float64,
                            device="cpu",
                        )
                        for metric in metrics
                    ],
                    dim=0,
                ).sum(dim=0)
                for name in ("sde_mean", "truth", "ode")
            },
            paired_evaluable_examples=sum(
                metric.paired_evaluable_examples for metric in metrics
            ),
            zero_diffusion_valid_examples=sum(
                metric.zero_diffusion_valid_examples for metric in metrics
            ),
            zero_diffusion_failed_examples=sum(
                metric.zero_diffusion_failed_examples for metric in metrics
            ),
            zero_diffusion_explosion_count=sum(
                metric.zero_diffusion_explosion_count for metric in metrics
            ),
            zero_diffusion_attempted_examples=sum(
                metric.zero_diffusion_attempted_examples
                for metric in metrics
            ),
            zero_diffusion_solver_failure_count=sum(
                metric.zero_diffusion_solver_failure_count
                for metric in metrics
            ),
            forecast_horizons=next(iter(horizon_lengths)),
        )
    else:
        pooled = _zero_diffusion_metrics_from_sums(
            squared_error_sums=squared_error_sums,
            squared_error_by_horizon_sums=(
                squared_error_by_horizon_sums
            ),
            scalar_value_count=sum(
                metric.scalar_value_count for metric in metrics
            ),
            horizon_scalar_value_count=sum(
                metric.horizon_scalar_value_count for metric in metrics
            ),
            paired_evaluable_examples=sum(
                metric.paired_evaluable_examples for metric in metrics
            ),
            zero_diffusion_valid_examples=sum(
                metric.zero_diffusion_valid_examples for metric in metrics
            ),
            zero_diffusion_failed_examples=sum(
                metric.zero_diffusion_failed_examples for metric in metrics
            ),
            zero_diffusion_explosion_count=sum(
                metric.zero_diffusion_explosion_count for metric in metrics
            ),
            zero_diffusion_attempted_examples=sum(
                metric.zero_diffusion_attempted_examples
                for metric in metrics
            ),
            zero_diffusion_solver_failure_count=sum(
                metric.zero_diffusion_solver_failure_count
                for metric in metrics
            ),
        )
    pooled.repeats = len(metrics)
    pooled.repeat_statistics = common_sde._AttrDict()
    for name in ("mse_to_sde_mean", "mse_to_truth", "mse_to_ode"):
        mean, std = _finite_mean_std([metric[name] for metric in metrics])
        pooled.repeat_statistics[name] = common_sde._AttrDict(
            mean=mean,
            standard_deviation=std,
        )
    for name in (
        "mse_to_sde_mean_by_horizon",
        "mse_to_truth_by_horizon",
        "mse_to_ode_by_horizon",
    ):
        mean, std = _vector_mean_std([metric[name] for metric in metrics])
        pooled.repeat_statistics[name] = common_sde._AttrDict(
            mean=mean,
            standard_deviation=std,
        )
    return pooled


def _aggregate_sde_coordinate_metrics(metrics):
    first = metrics[0]
    forecast_horizons = first.forecast_horizons
    output_dimensions = first.output_dimensions
    if any(
        metric.forecast_horizons != forecast_horizons
        or metric.output_dimensions != output_dimensions
        for metric in metrics
    ):
        raise ValueError(
            "Cannot aggregate coordinate metrics with different shapes."
        )

    counts_are_tensors = any(
        torch.is_tensor(metric.loss_value_count_per_dimension)
        for metric in metrics
    )
    if counts_are_tensors:
        # Sparse targets: per-dimension observed-entry counts. Dense
        # conditions are promoted to a full count per dimension.
        def _count_tensor(metric):
            count = metric.loss_value_count_per_dimension
            if torch.is_tensor(count):
                return count.to(dtype=torch.float64, device="cpu")
            return torch.full(
                (output_dimensions,),
                float(count),
                dtype=torch.float64,
            )

        total_loss_count = None
        for metric in metrics:
            count = _count_tensor(metric)
            total_loss_count = (
                count if total_loss_count is None
                else total_loss_count + count
            )
        loss_squared_sum = torch.zeros(
            output_dimensions,
            dtype=torch.float64,
        )
        for metric in metrics:
            count = _count_tensor(metric)
            if not bool((count > 0).any()):
                continue
            values = torch.tensor(
                metric.mean_prediction_loss_by_dimension,
                dtype=torch.float64,
            )
            # Dimensions without observations report NaN loss with a zero
            # count; they contribute nothing to the pooled sum.
            loss_squared_sum += (
                torch.nan_to_num(values, nan=0.0) * count
            )
    else:
        total_loss_count = sum(
            metric.loss_value_count_per_dimension for metric in metrics
        )
        loss_squared_sum = torch.zeros(
            output_dimensions,
            dtype=torch.float64,
        )
        for metric in metrics:
            if metric.loss_value_count_per_dimension == 0:
                continue
            values = torch.tensor(
                metric.mean_prediction_loss_by_dimension,
                dtype=torch.float64,
            )
            loss_squared_sum += (
                values * metric.loss_value_count_per_dimension
            )

    total_variance_examples = sum(
        metric.variance_evaluable_examples for metric in metrics
    )
    variance_matrix_sum = torch.zeros(
        (forecast_horizons, output_dimensions),
        dtype=torch.float64,
    )
    for metric in metrics:
        if metric.variance_evaluable_examples == 0:
            continue
        values = torch.tensor(
            metric.prediction_variance_by_horizon_and_dimension,
            dtype=torch.float64,
        )
        variance_matrix_sum += values * metric.variance_evaluable_examples

    pooled = _sde_coordinate_metrics_from_sums(
        loss_by_dimension_squared_sum=loss_squared_sum,
        loss_value_count_per_dimension=total_loss_count,
        variance_by_horizon_and_dimension_sum=variance_matrix_sum,
        variance_evaluable_examples=total_variance_examples,
        forecast_horizons=forecast_horizons,
        output_dimensions=output_dimensions,
    )
    pooled.repeats = len(metrics)
    pooled.repeat_statistics = common_sde._AttrDict()
    for name in (
        "mean_prediction_loss_by_dimension",
        "prediction_variance_by_dimension",
        "prediction_variance_by_horizon",
        "prediction_variance_by_horizon_and_dimension",
    ):
        mean, std = _vector_mean_std([metric[name] for metric in metrics])
        pooled.repeat_statistics[name] = common_sde._AttrDict(
            mean=mean,
            standard_deviation=std,
        )
    return pooled


def _vector_mean_std(vectors):
    usable = [
        np.asarray(vector, dtype=np.float64)
        for vector in vectors
        if vector
    ]
    if not usable:
        return [], []
    lengths = {vector.shape for vector in usable}
    if len(lengths) != 1:
        raise ValueError(
            "Cannot aggregate horizon metrics with different shapes: "
            f"{sorted(str(shape) for shape in lengths)}"
        )
    stacked = np.stack(usable, axis=0)
    return (
        np.nanmean(stacked, axis=0).tolist(),
        (
            np.nanstd(stacked, axis=0, ddof=1).tolist()
            if stacked.shape[0] > 1
            else np.zeros_like(stacked[0], dtype=np.float64).tolist()
        ),
    )


def aggregate_stress_results(records, distance_value_groups):
    grouped = {}
    for record in records:
        key = (
            record.missing_pattern,
            record.missing_rate,
            record.input_noise_level,
        )
        grouped.setdefault(key, []).append(record)

    summaries = []
    scalar_metrics = (
        "ode_loss",
        "ode_relative_degradation",
        "sde_loss",
        "sde_relative_degradation",
        "sde_prediction_variance",
        "sde_prediction_std",
        "sde_zero_diffusion_mse_to_sde_mean",
        "sde_zero_diffusion_mse_to_truth",
        "sde_zero_diffusion_mse_to_ode",
        "sde_explosion_rate",
        "sde_solver_failure_count",
        "sde_failed_examples",
    )

    for (pattern, missing_rate, noise_level), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][1], item[0][2], item[0][0]),
    ):
        summary = common_sde._AttrDict(
            missing_pattern=pattern,
            missing_rate=missing_rate,
            input_noise_level=noise_level,
            repeats=len(group),
            total_explosions=sum(
                record.sde_explosion_count for record in group
            ),
            total_attempted_paths=sum(
                record.sde_attempted_path_count for record in group
            ),
        )
        summary.total_explosion_rate = (
            summary.total_explosions / summary.total_attempted_paths
            if summary.total_attempted_paths
            else math.nan
        )

        for metric in scalar_metrics:
            mean, std = _finite_mean_std(
                [record[metric] for record in group]
            )
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std

        distance_group = distance_value_groups[
            (pattern, missing_rate, noise_level)
        ]
        pooled_distance_values = common_sde._AttrDict(
            (
                field,
                torch.cat(chunks, dim=0),
            )
            for field, chunks in distance_group.items()
        )
        combined_relative_metrics = ode_sde_distance_statistics(
            pooled_distance_values,
        )
        combined_relative_metrics.loss_relative_degradation = (
            common_sde._AttrDict(
                formula=(
                    "condition_loss / max(reference_loss, 1e-12) - 1"
                ),
                reference_missing_rate=group[0].reference_missing_rate,
                reference_input_noise_level=0.0,
                ode=_distribution_statistics(
                    torch.tensor(
                        [
                            record.ode_relative_degradation
                            for record in group
                        ],
                        dtype=torch.float64,
                    )
                ),
                sde=_distribution_statistics(
                    torch.tensor(
                        [
                            record.sde_relative_degradation
                            for record in group
                        ],
                        dtype=torch.float64,
                    )
                ),
                reference_ode_loss=_distribution_statistics(
                    torch.tensor(
                        [record.reference_ode_loss for record in group],
                        dtype=torch.float64,
                    )
                ),
                reference_sde_loss=_distribution_statistics(
                    torch.tensor(
                        [record.reference_sde_loss for record in group],
                        dtype=torch.float64,
                    )
                ),
            )
        )
        summary.relative_ode_sde_metrics = combined_relative_metrics
        summary.l2_comparison_metrics = _aggregate_l2_comparison_metrics(
            [record.l2_comparison_metrics for record in group]
        )
        summary.sde_zero_diffusion_metrics = (
            _aggregate_zero_diffusion_metrics(
                [record.sde_zero_diffusion_metrics for record in group]
            )
        )
        for relation in ("sde_mean", "truth", "ode"):
            metric_name = f"mse_to_{relation}_by_horizon"
            summary[
                f"sde_zero_diffusion_{metric_name}_mean"
            ] = summary.sde_zero_diffusion_metrics[metric_name]
            summary[
                f"sde_zero_diffusion_{metric_name}_std"
            ] = (
                summary.sde_zero_diffusion_metrics.repeat_statistics[
                    metric_name
                ].standard_deviation
            )
        summary.sde_coordinate_metrics = (
            _aggregate_sde_coordinate_metrics(
                [record.sde_coordinate_metrics for record in group]
            )
        )
        summary.sde_loss_by_dimension = (
            summary.sde_coordinate_metrics
            .mean_prediction_loss_by_dimension
        )
        summary.sde_variance_by_dimension = (
            summary.sde_coordinate_metrics.prediction_variance_by_dimension
        )
        summary.sde_variance_by_horizon_and_dimension = (
            summary.sde_coordinate_metrics
            .prediction_variance_by_horizon_and_dimension
        )
        summary.sde_loss_pooled = float(
            np.mean(summary.sde_loss_by_dimension)
        )
        summary.sde_prediction_variance_pooled = float(
            np.mean(summary.sde_variance_by_dimension)
        )

        for metric in (
            "ode_loss_by_horizon",
            "sde_loss_by_horizon",
            "sde_variance_by_horizon",
        ):
            mean, std = _vector_mean_std(
                [record[metric] for record in group]
            )
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std

        summaries.append(summary)

    return summaries


def _check_stress_times(stress_times, trained_times):
    trained_times = trained_times.detach().cpu()
    stress_times = stress_times.detach().cpu()
    if stress_times.shape != trained_times.shape or not torch.allclose(
        stress_times,
        trained_times,
    ):
        raise ValueError(
            "Stress-test integration times do not match training times: "
            f"stress={stress_times.tolist()}, "
            f"trained={trained_times.tolist()}."
        )


def _build_stress_loader(
    args,
    batch_size,
    missing_rate,
    noise_level,
    corruption_seed,
    repeat,
):
    print(
        "Preparing stress data | Repeat: {} | Missing: {:.2f} | "
        "Noise: {:.3f} | Pattern: {}".format(
            repeat,
            missing_rate,
            noise_level,
            args.missing_pattern,
        ),
        flush=True,
    )
    started = time.perf_counter()
    dataset = datasets.get_dataset(args.dataset_name)
    if not hasattr(dataset, "get_stress_test_dataloader"):
        raise ValueError(
            f"Dataset '{args.dataset_name}' does not implement "
            "get_stress_test_dataloader()."
        )
    stress_times, stress_loader = dataset.get_stress_test_dataloader(
        batch_size=batch_size,
        append_time=args.intensity,
        time_seq=args.time_seq,
        y_seq=args.y_seq,
        missing_rate=missing_rate,
        noise_level=noise_level,
        corruption_seed=corruption_seed,
        missing_pattern=args.missing_pattern,
    )
    elapsed = time.perf_counter() - started
    print(
        "Prepared stress data | Repeat: {} | Missing: {:.2f} | "
        "Noise: {:.3f} | Examples: {} | Time: {:.2f}s".format(
            repeat,
            missing_rate,
            noise_level,
            len(stress_loader.dataset),
            elapsed,
        ),
        flush=True,
    )
    return stress_times, stress_loader


def run_stress_grid(args, ode_result, sde_result, sde_method):
    records = []
    distance_value_groups = {}
    batch_size = ode_result.test_dataloader.batch_size

    for repeat in range(args.corruption_repeats):
        corruption_seed = args.corruption_seed + repeat

        reference_times, reference_loader = _build_stress_loader(
            args=args,
            batch_size=batch_size,
            missing_rate=args.missing_rate,
            noise_level=0.0,
            corruption_seed=corruption_seed,
            repeat=repeat,
        )
        _check_stress_times(reference_times, ode_result.times)
        reference = evaluate(
            ode_result,
            sde_result,
            sde_method,
            test_dataloader=reference_loader,
        )

        for missing_rate in args.test_missing_rates:
            for noise_level in args.input_noise_levels:
                if _is_reference_condition(
                    args,
                    missing_rate,
                    noise_level,
                ):
                    comparison = reference
                else:
                    stress_times, stress_loader = _build_stress_loader(
                        args=args,
                        batch_size=batch_size,
                        missing_rate=missing_rate,
                        noise_level=noise_level,
                        corruption_seed=corruption_seed,
                        repeat=repeat,
                    )
                    _check_stress_times(stress_times, ode_result.times)
                    comparison = evaluate(
                        ode_result,
                        sde_result,
                        sde_method,
                        test_dataloader=stress_loader,
                    )

                record = _make_stress_record(
                    args=args,
                    repeat=repeat,
                    corruption_seed=corruption_seed,
                    missing_rate=missing_rate,
                    noise_level=noise_level,
                    comparison=comparison,
                    reference=reference,
                )
                records.append(record)

                group_key = (
                    args.missing_pattern,
                    float(missing_rate),
                    float(noise_level),
                )
                value_group = distance_value_groups.setdefault(
                    group_key,
                    {
                        field: []
                        for field in (
                            "difference_components",
                            "raw_distance_l2",
                        )
                    },
                )
                for field in value_group:
                    value_group[field].append(
                        comparison._ode_sde_distance_values[field]
                    )

                print(
                    "Stress | Repeat: {} | Missing: {:.2f} | "
                    "Noise: {:.3f} | ODE loss: {:.6g} ({:+.1%}) | "
                    "SDE loss: {:.6g} ({:+.1%}) | "
                    "SDE variance: {:.6g} | "
                    "SDE(g=0) MSE -> SDEmean/True/ODE: "
                    "{:.6g}/{:.6g}/{:.6g} | Explosions: {}/{}".format(
                        repeat,
                        missing_rate,
                        noise_level,
                        record.ode_loss,
                        record.ode_relative_degradation,
                        record.sde_loss,
                        record.sde_relative_degradation,
                        record.sde_prediction_variance,
                        record.sde_zero_diffusion_mse_to_sde_mean,
                        record.sde_zero_diffusion_mse_to_truth,
                        record.sde_zero_diffusion_mse_to_ode,
                        record.sde_explosion_count,
                        record.sde_attempted_path_count,
                    )
                )

    return records, aggregate_stress_results(
        records,
        distance_value_groups,
    )


def print_relative_ode_sde_metrics(title, metrics, indent=""):
    """Print component and dimension-adjusted ODE/SDE distances."""
    print(f"{indent}{title}")

    print(f"{indent}  Component distances:")
    for statistics in metrics.raw_distance.dimensions:
        print(
            "{}    Dimension {:02d} | Mean: {:.8g} | Variance: {:.8g} | "
            "Outside 2 std: {}/{}".format(
                indent,
                statistics.dimension,
                statistics.mean,
                statistics.variance,
                statistics.outside_2std_count,
                statistics.value_count,
            )
        )

    statistics = metrics.raw_distance.l2_norm
    print(
        "{}    L2/sqrt(dim)  | Mean: {:.8g} | RMS: {:.8g} | "
        "Variance: {:.8g} | "
        "Outside 2 std: {}/{}".format(
            indent,
            statistics.mean,
            statistics.root_mean_square,
            statistics.variance,
            statistics.outside_2std_count,
            statistics.value_count,
        )
    )

    statistics = metrics.raw_distance.unnormalized_l2_norm
    print(
        "{}    Raw 2-norm    | Mean: {:.8g} | Variance: {:.8g} | "
        "Outside 2 std: {}/{}".format(
            indent,
            statistics.mean,
            statistics.variance,
            statistics.outside_2std_count,
            statistics.value_count,
        )
    )

    if "loss_relative_degradation" in metrics:
        degradation = metrics.loss_relative_degradation
        print(
            "{}  Loss degradation relative to Missing: {:.2f}, "
            "Noise: {:.3f}".format(
                indent,
                degradation.reference_missing_rate,
                degradation.reference_input_noise_level,
            )
        )
        for model_name in ("ODE", "SDE"):
            statistics = degradation[model_name.lower()]
            print(
                "{}    {} | Mean: {:.8g} | Variance: {:.8g} | "
                "Outside 2 std: {}/{}".format(
                    indent,
                    model_name,
                    statistics.mean,
                    statistics.variance,
                    statistics.outside_2std_count,
                    statistics.value_count,
                )
            )


def print_l2_comparison_metrics(title, metrics, indent=""):
    """Print the three directly comparable RMS-per-dimension distances."""
    print(f"{indent}{title}")
    print(
        "{}  ODE -> truth: {:.8g} | SDE mean -> truth: {:.8g} | "
        "ODE -> SDE mean: {:.8g}".format(
            indent,
            metrics.ode_to_truth,
            metrics.sde_mean_to_truth,
            metrics.ode_to_sde,
        )
    )


def print_zero_diffusion_metrics(title, metrics, indent=""):
    """Print test-only drift ablation MSEs on their paired population."""
    print(f"{indent}{title}")
    print(
        "{}  MSE SDE(g=0) -> SDE mean: {:.8g} | -> truth: {:.8g} | "
        "-> ODE: {:.8g} | paired examples: {}".format(
            indent,
            metrics.mse_to_sde_mean,
            metrics.mse_to_truth,
            metrics.mse_to_ode,
            metrics.paired_evaluable_examples,
        )
    )
    for label, values in (
        ("SDE mean", metrics.mse_to_sde_mean_by_horizon),
        ("truth", metrics.mse_to_truth_by_horizon),
        ("ODE", metrics.mse_to_ode_by_horizon),
    ):
        formatted_values = ", ".join(
            f"{value:.8g}" for value in values
        )
        print(
            f"{indent}    MSE SDE(g=0) -> {label} by horizon/output "
            f"index: [{formatted_values}]"
        )


def print_sde_coordinate_metrics(title, metrics, indent=""):
    """Print per-coordinate loss/variance and requested horizon slices."""
    print(f"{indent}{title}")
    losses = metrics.mean_prediction_loss_by_dimension
    variances = metrics.prediction_variance_by_dimension
    for dimension, (loss, variance) in enumerate(zip(losses, variances)):
        print(
            "{}  Dimension {:02d} | SDE-mean loss: {:.8g} | "
            "MC variance: {:.8g}".format(
                indent,
                dimension,
                loss,
                variance,
            )
        )

    selected_dimensions = metrics.selected_coordinate_indices_available
    print(f"{indent}  Selected horizon variance by dimension:")
    for horizon_metrics in metrics.selected_horizon_variance_by_dimension:
        selected = " | ".join(
            "d{:02d}={:.6g}".format(
                dimension,
                horizon_metrics["values"][dimension],
            )
            for dimension in selected_dimensions
        )
        print(
            "{}    h{:02d} | mean(all dims)={:.8g} | {}".format(
                indent,
                horizon_metrics.horizon,
                horizon_metrics.mean_over_dimensions,
                selected,
            )
        )
    if metrics.unavailable_horizon_indices:
        print(
            f"{indent}    unavailable horizons: "
            f"{metrics.unavailable_horizon_indices}"
        )


def main():
    global ODE_MODEL, MC_SAMPLES, MC_SEED, EXPLOSION_THRESHOLD

    args = parse_args()
    ODE_MODEL = args.ode_model
    MC_SAMPLES = args.mc_samples
    MC_SEED = args.mc_seed
    EXPLOSION_THRESHOLD = args.explosion_threshold

    if args.model not in SDE_MODELS:
        raise ValueError(
            f"For the combined run --model must be one of "
            f"{sorted(SDE_MODELS)}; got {args.model!r}."
        )
    sde_config = common_sde.resolve_sde_config(
        args.model,
        sde_input_option=args.sde_input_option,
        sde_noise_option=args.sde_noise_option,
        sde_mixture_options=args.sde_mixture_options,
    )
    print(
        "SDE config | Model: {} | Input option: {} | Noise option: {} "
        "({}) | Raw g: {} | Effective g: {}".format(
            sde_config.model_name,
            sde_config.input_option,
            sde_config.noise_option,
            sde_config.diffusion_name,
            sde_config.raw_diffusion_formula,
            sde_config.effective_diffusion_formula,
        ),
        flush=True,
    )

    dataset = datasets.get_dataset(args.dataset_name)
    num_input_features, num_output_features = datasets.feature_dimensions(dataset)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by the current forecasting training "
            "pipeline, but torch.cuda.is_available() is False."
        )
    if MC_SAMPLES < 2:
        raise ValueError(
            "--mc_samples must be at least 2 to estimate variance."
        )
    if EXPLOSION_THRESHOLD <= 0:
        raise ValueError("--explosion_threshold must be positive.")
    _validate_stress_arguments(args)
    _, unavailable_horizons = _available_indices(
        SELECTED_HORIZON_INDICES,
        args.y_seq,
    )
    if unavailable_horizons:
        print(
            "Requested variance horizons {} are outside --y_seq {} and "
            "will be recorded as unavailable. Use --y_seq 20 for the "
            "standard analysis.".format(
                unavailable_horizons,
                args.y_seq,
            ),
            flush=True,
        )

    mujoco_sde = load_mujoco_sde_module()
    sde_method = mujoco_sde._resolve_sde_method(args.method)

    # Train the ODE once on the unperturbed benchmark loaders.
    ode_result = mujoco.main(
        manual_seed=args.seed,
        intensity=args.intensity,
        device=DEVICE,
        max_epochs=args.epoch,
        missing_rate=args.missing_rate,
        model_name=ODE_MODEL,
        hidden_channels=args.h_channels,
        hidden_hidden_channels=args.hh_channels,
        num_hidden_layers=args.layers,
        ode_hidden_hidden_channels=args.ode_hidden_hidden_channels,
        lr=args.lr,
        c1=args.c1,
        c2=args.c2,
        weight_decay=args.weight_decay,
        dry_run=True,
        method="rk4",
        step_mode=args.step_mode,
        time_seq=args.time_seq,
        y_seq=args.y_seq,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
    )

    torch.cuda.empty_cache()

    # Train the selected SDE once on exactly the same benchmark split.
    sde_result = mujoco_sde.main(
        manual_seed=args.seed,
        intensity=args.intensity,
        device=DEVICE,
        max_epochs=args.epoch,
        missing_rate=args.missing_rate,
        model_name=args.model,
        hidden_channels=args.h_channels,
        hidden_hidden_channels=args.hh_channels,
        num_hidden_layers=args.layers,
        ode_hidden_hidden_channels=args.ode_hidden_hidden_channels,
        dry_run=True,
        method=sde_method,
        step_mode=args.step_mode,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss=args.loss,
        reg=args.reg,
        scale=args.scale,
        time_seq=args.time_seq,
        y_seq=args.y_seq,
        sde_input_option=args.sde_input_option,
        sde_noise_option=args.sde_noise_option,
        sde_mixture_options=args.sde_mixture_options,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
    )
    if "sde_config" in sde_result:
        sde_config = sde_result.sde_config

    # Preserve the original cached-test comparison as a separate baseline.
    comparison = evaluate(ode_result, sde_result, sde_method)

    # Reuse the trained weights for the complete test-only corruption grid.
    stress_records, stress_summary = run_stress_grid(
        args,
        ode_result,
        sde_result,
        sde_method,
    )

    combined_result_relative_metrics = common_sde._AttrDict(
        comparison.relative_ode_sde_metrics.copy()
    )
    combined_result_relative_metrics[
        "loss_relative_degradation_by_stress_condition"
    ] = [
        common_sde._AttrDict(
            missing_pattern=summary.missing_pattern,
            missing_rate=summary.missing_rate,
            input_noise_level=summary.input_noise_level,
            repeats=summary.repeats,
            metrics=summary.relative_ode_sde_metrics[
                "loss_relative_degradation"
            ],
        )
        for summary in stress_summary
    ]

    final_l2_metrics = common_sde._AttrDict(
        metric_scale="RMS per output dimension",
        baseline=comparison.l2_comparison_metrics,
        stress_summary=[
            common_sde._AttrDict(
                missing_pattern=summary.missing_pattern,
                missing_rate=summary.missing_rate,
                input_noise_level=summary.input_noise_level,
                repeats=summary.repeats,
                ode_to_truth=(
                    summary.l2_comparison_metrics.ode_to_truth
                ),
                sde_mean_to_truth=(
                    summary.l2_comparison_metrics.sde_mean_to_truth
                ),
                ode_to_sde=summary.l2_comparison_metrics.ode_to_sde,
                metrics=summary.l2_comparison_metrics,
            )
            for summary in stress_summary
        ],
    )

    final_zero_diffusion_metrics = common_sde._AttrDict(
        metric_scale="MSE (no square root)",
        training_used_learned_diffusion=True,
        ablation_scope="test only",
        baseline=comparison.sde_zero_diffusion_metrics,
        stress_summary=[
            common_sde._AttrDict(
                missing_pattern=summary.missing_pattern,
                missing_rate=summary.missing_rate,
                input_noise_level=summary.input_noise_level,
                repeats=summary.repeats,
                metrics=summary.sde_zero_diffusion_metrics,
            )
            for summary in stress_summary
        ],
    )
    final_sde_coordinate_metrics = common_sde._AttrDict(
        selected_coordinate_indices=list(SELECTED_COORDINATE_INDICES),
        selected_horizon_indices=list(SELECTED_HORIZON_INDICES),
        baseline=comparison.sde_metrics.coordinate_metrics,
        stress_summary=[
            common_sde._AttrDict(
                missing_pattern=summary.missing_pattern,
                missing_rate=summary.missing_rate,
                input_noise_level=summary.input_noise_level,
                repeats=summary.repeats,
                metrics=summary.sde_coordinate_metrics,
            )
            for summary in stress_summary
        ],
    )

    if hasattr(dataset, "dataset_metadata"):
        dataset_metadata = dataset.dataset_metadata(args.time_seq, args.y_seq)
    else:
        dataset_metadata = {
            "dataset_name": args.dataset_name,
            "input_features": num_input_features,
            "output_features": num_output_features,
        }

    result = common_sde._AttrDict(
        times=ode_result.times,
        dataset_metadata=dataset_metadata,
        batch_size=args.batch_size,
        memory_usage=common_sde._AttrDict(
            ode=ode_result.memory_usage,
            sde=sde_result.memory_usage,
        ),
        baseline_memory=None,
        train_dataloader=ode_result.train_dataloader,
        val_dataloader=ode_result.val_dataloader,
        test_dataloader=ode_result.test_dataloader,
        model=(
            f"ODE={ODE_MODEL}; SDE={args.model}"
            f"[input={sde_config.input_option}, "
            f"noise={sde_config.noise_option}:"
            f"{sde_config.diffusion_name}]"
        ),
        sde_config=sde_config,
        parameters=common_sde._AttrDict(
            ode=ode_result.parameters,
            sde=sde_result.parameters,
        ),
        history=common_sde._AttrDict(
            ode=ode_result.history,
            sde=sde_result.history,
        ),
        ode_metrics=comparison.ode_metrics,
        sde_metrics=comparison.sde_metrics,
        sde_zero_diffusion_metrics=(
            comparison.sde_zero_diffusion_metrics
        ),
        relative_ode_sde_metrics=combined_result_relative_metrics,
        stress_config=common_sde._AttrDict(
            dataset_name=args.dataset_name,
            training_missing_rate=args.missing_rate,
            reference_missing_rate=args.missing_rate,
            reference_input_noise_level=0.0,
            test_missing_rates=args.test_missing_rates,
            input_noise_levels=args.input_noise_levels,
            corruption_repeats=args.corruption_repeats,
            corruption_seed=args.corruption_seed,
            missing_pattern=args.missing_pattern,
            selected_coordinate_indices=list(
                SELECTED_COORDINATE_INDICES
            ),
            selected_horizon_indices=list(SELECTED_HORIZON_INDICES),
        ),
        stress_metrics=stress_records,
        stress_summary=stress_summary,
        monte_carlo_samples=MC_SAMPLES,
        training_seed=args.seed,
        monte_carlo_seed=MC_SEED,
        final_l2_metrics=final_l2_metrics,
        final_zero_diffusion_metrics=final_zero_diffusion_metrics,
        final_sde_coordinate_metrics=final_sde_coordinate_metrics,
    )

    (Path(__file__).resolve().parent / "results").mkdir(exist_ok=True)
    common_sde._save_results(f"{args.dataset_name}/ODEvsSDE", result)

    print(
        "Test | Dataset size: {} | ODE loss: {:.6g} | "
        "SDE mean loss: {:.6g} | SDE variance: {:.6g} | "
        "SDE explosions: {}/{} ({:.3%}) | Failed examples: {}".format(
            comparison.dataset_size,
            comparison.ode_metrics.loss,
            comparison.sde_metrics.loss,
            comparison.sde_metrics.prediction_variance,
            comparison.sde_metrics.explosion_count,
            comparison.sde_metrics.attempted_path_count,
            comparison.sde_metrics.explosion_rate,
            comparison.sde_metrics.failed_examples,
        )
    )

    print_relative_ode_sde_metrics(
        "ODE/SDE prediction distance metrics (baseline):",
        comparison.relative_ode_sde_metrics,
    )

    print("Stress summary:")
    for summary in stress_summary:
        print(
            "  Missing: {:.2f} | Noise: {:.3f} | "
            "ODE: {:.6g} +/- {:.3g} | "
            "SDE: {:.6g} +/- {:.3g} | "
            "SDE variance: {:.6g} | "
            "SDE(g=0) MSE -> SDEmean/True/ODE: "
            "{:.6g}/{:.6g}/{:.6g} | Explosions: {}/{}".format(
                summary.missing_rate,
                summary.input_noise_level,
                summary.ode_loss_mean,
                summary.ode_loss_std,
                summary.sde_loss_mean,
                summary.sde_loss_std,
                summary.sde_prediction_variance_pooled,
                summary.sde_zero_diffusion_metrics.mse_to_sde_mean,
                summary.sde_zero_diffusion_metrics.mse_to_truth,
                summary.sde_zero_diffusion_metrics.mse_to_ode,
                summary.total_explosions,
                summary.total_attempted_paths,
            )
        )
        print_relative_ode_sde_metrics(
            "ODE/SDE prediction distance metrics:",
            summary.relative_ode_sde_metrics,
            indent="    ",
        )

    print("Final comparable L2 metrics (RMS per output dimension):")
    print_l2_comparison_metrics(
        "Baseline:",
        comparison.l2_comparison_metrics,
        indent="  ",
    )
    for summary in stress_summary:
        print_l2_comparison_metrics(
            "Missing: {:.2f} | Noise: {:.3f} | Pattern: {}".format(
                summary.missing_rate,
                summary.input_noise_level,
                summary.missing_pattern,
            ),
            summary.l2_comparison_metrics,
            indent="  ",
        )

    print("Final zero-diffusion ablation metrics (test only, MSE):")
    print_zero_diffusion_metrics(
        "Baseline:",
        comparison.sde_zero_diffusion_metrics,
        indent="  ",
    )
    for summary in stress_summary:
        print_zero_diffusion_metrics(
            "Missing: {:.2f} | Noise: {:.3f} | Pattern: {}".format(
                summary.missing_rate,
                summary.input_noise_level,
                summary.missing_pattern,
            ),
            summary.sde_zero_diffusion_metrics,
            indent="  ",
        )

    print("Final SDE coordinate and horizon metrics:")
    print_sde_coordinate_metrics(
        "Baseline:",
        comparison.sde_metrics.coordinate_metrics,
        indent="  ",
    )
    for summary in stress_summary:
        print_sde_coordinate_metrics(
            "Missing: {:.2f} | Noise: {:.3f} | Pattern: {}".format(
                summary.missing_rate,
                summary.input_noise_level,
                summary.missing_pattern,
            ),
            summary.sde_coordinate_metrics,
            indent="  ",
        )


if __name__ == "__main__":
    main()
