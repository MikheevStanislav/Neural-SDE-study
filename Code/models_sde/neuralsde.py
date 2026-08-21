"""Benchmark NSDE runtime with explicit finite-horizon stabilization heuristics.

LSDE, LNSDE, and GSDE runtime paths intentionally keep bounded `sin/cos(t)`
time features and outer `tanh` clipping on drift/diffusion outputs. These are
engineering stabilizers for discrete solver robustness on normalized finite
horizons, not pure theorem-faithful parameterizations. In this runtime
layer LSDE mainly uses them as practical anti-blow-up controls, LNSDE gives up
the paper's infinite-horizon asymptotic interpretation, and GSDE trades pure
geometric positivity structure for a stabilized discrete-time approximation.
"""

import math
import numbers
import pathlib
import sys
import torch
import torchcde
import torchsde

here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(here / '..' / '..'))

import controldiffeq

_PROPOSAL_METHOD_CONTRACT = {
    "lsde": (2, 16),
    "lnsde": (4, 17),
    "gsde": (6, 17),
}


_LOG_SCALE_MIN = -20.0
_LOG_SCALE_MAX = 10.0
_EXP_STATE_MAX = 10.0
_DEFAULT_DIFFUSION_EPS = 1e-6
_DEFAULT_MIXTURE_EPS = 1e-12
_MIXTURE_CLAMP_DIVISOR = 4.0
_MIXTURE_DEFAULT_OPTIONS = (16, 23, 6)


def _positive_scale(log_scale, *, min=_LOG_SCALE_MIN, max=_LOG_SCALE_MAX):
    """Return a positive scale from an unconstrained log-scale parameter."""
    return log_scale.clamp(min=min, max=max).exp()


def _safe_cube(y):
    """Cube a tensor with a dtype-aware clamp to avoid overflow."""
    if not y.is_floating_point():
        raise TypeError(
            "The SDE hidden state must use a floating-point dtype."
        )
    # Leave a factor-four margin so rounding at the clamp boundary cannot
    # overflow during the cube operation, including for float16.
    cube_limit = (torch.finfo(y.dtype).max / _MIXTURE_CLAMP_DIVISOR) ** (1.0 / 3.0)
    return y.clamp(min=-cube_limit, max=cube_limit).pow(3)


def _diffusion_spec(noise_option, label, raw_formula):
    return {
        "noise_option": noise_option,
        "label": label,
        "raw_formula": raw_formula,
        "effective_formula": (
            f"tanh(sigmoid(theta) * ({raw_formula}))"
        ),
        "noise_type": "diagonal",
    }


# Options 1..23 are the 23 non-zero diffusion families used by the benchmark.
# Option 0 is retained as the deterministic no-diffusion control. Labels and
# formula strings are persisted in experiment metadata, so keep them stable.
DIFFUSION_SPECS = {
    0: _diffusion_spec(0, "zero", "0"),
    1: _diffusion_spec(
        1,
        "scalar_constant",
        "exp(clamp(log_sigma, -20, 10))",
    ),
    2: _diffusion_spec(
        2,
        "scalar_time",
        "exp(clamp(log_sigma, -20, 10)) * t",
    ),
    3: _diffusion_spec(
        3,
        "scalar_state",
        "exp(clamp(log_sigma, -20, 10)) * y",
    ),
    4: _diffusion_spec(
        4,
        "diagonal_constant",
        "exp(clamp(log_sigma_diag, -20, 10))",
    ),
    5: _diffusion_spec(
        5,
        "diagonal_time",
        "exp(clamp(log_sigma_diag, -20, 10)) * t",
    ),
    6: _diffusion_spec(
        6,
        "diagonal_state",
        "exp(clamp(log_sigma_diag, -20, 10)) * y",
    ),
    7: _diffusion_spec(
        7,
        "holder_sqrt_abs_state",
        "sqrt(abs(y) + eps) - sqrt(eps)",
    ),
    8: _diffusion_spec(
        8,
        "cubic_state",
        "clamp(y, -cube_limit(dtype), cube_limit(dtype)) ** 3",
    ),
    9: _diffusion_spec(9, "sigmoid_state", "sigmoid(y)"),
    10: _diffusion_spec(10, "relu_state", "relu(y)"),
    11: _diffusion_spec(11, "time_state", "t * y"),
    12: _diffusion_spec(
        12,
        "linear_time",
        "linear_t([sin(t), cos(t)])",
    ),
    13: _diffusion_spec(
        13,
        "linear_time_times_state",
        "linear_t([sin(t), cos(t)]) * y",
    ),
    14: _diffusion_spec(
        14,
        "linear_joint_time_state",
        "linear_ty([sin(t), cos(t), y])",
    ),
    15: _diffusion_spec(
        15,
        "linear_joint_time_state_times_state",
        "linear_ty([sin(t), cos(t), y]) * y",
    ),
    16: _diffusion_spec(
        16,
        "mlp_time",
        "relu(mlp_t([sin(t), cos(t)]))",
    ),
    17: _diffusion_spec(
        17,
        "mlp_time_times_state",
        "relu(mlp_t([sin(t), cos(t)])) * y",
    ),
    18: _diffusion_spec(
        18,
        "mlp_joint_time_state",
        "relu(mlp_ty([sin(t), cos(t), y]))",
    ),
    19: _diffusion_spec(
        19,
        "mlp_joint_time_state_times_state",
        "relu(mlp_ty([sin(t), cos(t), y])) * y",
    ),
    20: _diffusion_spec(
        20,
        "log1p_abs_state",
        "log1p(abs(y))",
    ),
    21: _diffusion_spec(
        21,
        "exp_state",
        "exp(clamp(y, max=10))",
    ),
    22: _diffusion_spec(
        22,
        "linear_time_times_linear_state",
        (
            "relu(linear_t([sin(t), cos(t)])) * "
            "relu(linear_state(y))"
        ),
    ),
    23: _diffusion_spec(
        23,
        "linear_time_plus_linear_state",
        (
            "relu(linear_t([sin(t), cos(t)])) + "
            "relu(linear_state(y))"
        ),
    ),
    24: _diffusion_spec(
        24,
        "mixture3_rms",
        "s * sqrt(pi1*g1(t,y)^2 + pi2*g2(t,y)^2 + pi3*g3(t,y)^2), "
        "pi=softmax(alpha)",
    ),
}


def _validated_integer_option(value, *, name, valid_values):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(
            f"{name} must be an integer in {sorted(valid_values)}; "
            f"got {value!r}."
        )
    value = int(value)
    if value not in valid_values:
        raise ValueError(
            f"Unknown {name} {value}. Valid values are "
            f"{sorted(valid_values)}."
        )
    return value


def get_diffusion_spec(noise_option):
    """Return JSON-safe metadata for one diffusion option.

    A fresh dictionary is returned so callers can add run-specific metadata
    without mutating the stable module registry.
    """
    noise_option = _validated_integer_option(
        noise_option,
        name="noise_option",
        valid_values=DIFFUSION_SPECS,
    )
    return DIFFUSION_SPECS[noise_option].copy()


def _prepare_sde_solver_kwargs(times, kwargs, *, default_method, respect_euler_grid):
    kwargs = dict(kwargs)
    time_diffs = times[1:] - times[:-1]
    dt = max(time_diffs.min().item(), 1e-3)

    if 'method' not in kwargs:
        kwargs['method'] = default_method

    if kwargs['method'] == 'srk':
        options = kwargs.setdefault('options', {})
        if 'dt' not in options:
            options['dt'] = dt
    elif kwargs['method'] == 'euler':
        options = kwargs.setdefault('options', {})
        if 'dt' not in options:
            if not respect_euler_grid or ('step_size' not in options and 'grid_constructor' not in options):
                options['dt'] = dt

    return kwargs, dt


class NeuralSDE(torch.nn.Module):
    def __init__(self, func, input_channels, hidden_channels, output_channels, initial=True):
        super().__init__()
        self.func = func
        self.initial = initial
        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        
        # self.linear = torch.nn.Linear(hidden_channels, output_channels)
        self.linear = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels),
                                          torch.nn.BatchNorm1d(hidden_channels), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                          torch.nn.Linear(hidden_channels, output_channels))    

    def _prepare_initial_state(self, times, z0):
        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            z0 = self.initial_network(self.func.X.evaluate(times[0]))
        else:
            assert not self.initial, "Was expecting to be given a value of z0."
        return z0

    def _solve_sde_path(self, times, ts, z0, kwargs):
        kwargs, dt = _prepare_sde_solver_kwargs(
            times,
            kwargs,
            default_method='euler',
            respect_euler_grid=False,
        )
        return torchsde.sdeint(sde=self.func,
                              y0=z0,
                              ts=ts,
                              dt=dt,
                              **kwargs)
        
    def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
        # control module
        self.func.set_X(*coeffs, times)
        
        z0 = self._prepare_initial_state(times, z0)
        
        # Figure out what times we need to solve for
        if stream:
            t = times
        else:
            # faff around to make sure that we're outputting at all the times we need for final_index.
            sorted_final_index, inverse_final_index = final_index.unique(sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]
            t = torch.cat([times[0].unsqueeze(0), times[sorted_final_index], times[-1].unsqueeze(0)])
                
        z_t = self._solve_sde_path(times, t, z0, kwargs)
                       
        # Organise the output
        if stream:
            # z_t is a tensor of shape (times, ..., channels), so change this to (..., times, channels)
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            # final_index is a tensor of shape (...)
            # z_t is a tensor of shape (times, ..., channels)
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)
            
        # Linear map and return
        pred_y = self.linear(z_t)
        return pred_y
    

class _ZeroDiffusionSDE(torch.nn.Module):
    """Test-time view of an SDE that keeps ``f`` and replaces ``g`` by zero.

    This proxy deliberately does not mutate the trained vector field.  It is
    constructed only for one solver call, so an exception cannot leave the
    model in a zero-diffusion state.
    """

    def __init__(self, base_sde):
        super().__init__()
        self.base_sde = base_sde
        self.sde_type = base_sde.sde_type
        self.noise_type = base_sde.noise_type
        if self.noise_type != "diagonal":
            raise NotImplementedError(
                "The zero-diffusion forecasting proxy currently supports "
                f"diagonal noise; got {self.noise_type!r}."
            )

    def f(self, t, y):
        return self.base_sde.f(t, y)

    def g(self, t, y):
        return torch.zeros_like(y)


class NeuralSDE_forecasting(torch.nn.Module):
    def __init__(self, func, input_channels, output_time, hidden_channels, output_channels, initial=True):
        super().__init__()
        self.func = func
        self.initial = initial
        self.output_time = output_time
        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        
        # self.linear = torch.nn.Linear(hidden_channels, output_channels)
        self.linear = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels),
                                          # torch.nn.BatchNorm1d(hidden_channels), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_channels, output_channels))    

    def _prepare_initial_state(self, times, z0):
        if z0 is None:
            assert self.initial, "Was not expecting to be given no value of z0."
            z0 = self.initial_network(self.func.X.evaluate(times[0]))
        else:
            assert not self.initial, "Was expecting to be given a value of z0."
        return z0

    def _solve_sde_path(
        self,
        times,
        ts,
        z0,
        kwargs,
        *,
        zero_diffusion=False,
    ):
        kwargs, dt = _prepare_sde_solver_kwargs(
            times,
            kwargs,
            default_method='euler',
            respect_euler_grid=False,
        )
        solver_sde = (
            _ZeroDiffusionSDE(self.func)
            if zero_diffusion
            else self.func
        )
        return torchsde.sdeint(sde=solver_sde,
                              y0=z0,
                              ts=ts,
                              dt=dt,
                              **kwargs)
        
    def forward(
        self,
        times,
        coeffs,
        final_index,
        z0=None,
        stream=False,
        zero_diffusion=False,
        **kwargs,
    ):
        if zero_diffusion and self.training:
            raise RuntimeError(
                "zero_diffusion=True is a test-only ablation. Call model.eval() "
                "before requesting it; training must use the learned g_theta."
            )

        # control module
        # self.func.set_X(*coeffs, times)
        self.func.set_X(torch.cat(coeffs, dim=-1), times)
        
        z0 = self._prepare_initial_state(times, z0)
        
#         if stream:
#             t = times
#         else:
#             sorted_final_index, inverse_final_index = final_index.unique(sorted=True, return_inverse=True)
#             if 0 in sorted_final_index:
#                 sorted_final_index = sorted_final_index[1:]
#                 final_index = inverse_final_index
#             else:
#                 final_index = inverse_final_index + 1
#             if len(times) - 1 in sorted_final_index:
#                 sorted_final_index = sorted_final_index[:-1]

#             t = torch.cat([times[0].unsqueeze(0), times[sorted_final_index], times[-1].unsqueeze(0)])
        t = times
                         
        z_t = self._solve_sde_path(
            times,
            t,
            z0,
            kwargs,
            zero_diffusion=zero_diffusion,
        )
                                
        for i in range(len(z_t.shape) - 2, 0, -1):
            z_t = z_t.transpose(0, i)
        input_time = z_t.shape[1]
        pred_y = self.linear(z_t[:,input_time-self.output_time:,:])
        return pred_y


def _make_diffusion_parameters(module, noise_option, hidden_channels, sigma=1.0):
    """Create the noise-specific parameters for a single catalogue option."""
    if noise_option in (1, 2, 3):
        module.sigma = torch.nn.Parameter(
            torch.tensor([sigma]), requires_grad=True
        )
    if noise_option in (4, 5, 6):
        module.sigma_diag = torch.nn.Parameter(
            torch.tensor([sigma] * hidden_channels), requires_grad=True
        )
    if noise_option in (12, 13):
        module.noise_t = torch.nn.Linear(2, hidden_channels)
    if noise_option in (14, 15):
        module.noise_y = torch.nn.Linear(hidden_channels + 2, hidden_channels)
    if noise_option in (16, 17):
        module.noise_t = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
    if noise_option in (18, 19):
        module.noise_y = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels + 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
    if noise_option in (22, 23):
        module.noise_t = torch.nn.Linear(2, hidden_channels)
        module.noise_state = torch.nn.Linear(hidden_channels, hidden_channels)


def _raw_diffusion_for_option(noise_option, params, t, y, time_features, diffusion_eps):
    """Compute the raw diffusion for one catalogue option.

    ``params`` is any object/module carrying the option-specific parameters
    created by ``_make_diffusion_parameters``. This helper is shared between
    the standalone ``Diffusion_model`` and per-component ``_DiffusionComponent``
    instances used by the mixture diffusion.
    """
    if noise_option == 0:  # constant 0
        return torch.zeros_like(y)

    if noise_option in (1, 2, 3):
        scale = _positive_scale(params.sigma).expand_as(y)
        if noise_option == 1:  # constant sigma
            return scale
        if noise_option == 2:  # constant sigma * t
            return scale * t
        # noise_option == 3: constant sigma * y
        return scale * y

    if noise_option in (4, 5, 6):
        scale = _positive_scale(params.sigma_diag).expand_as(y)
        if noise_option == 4:  # constant diagonal sigma
            return scale
        if noise_option == 5:  # constant diagonal sigma * t
            return scale * t
        # noise_option == 6: constant diagonal sigma * y
        return scale * y

    # special cases
    if noise_option == 7:  # holder continuity
        eps = diffusion_eps
        return torch.sqrt(y.abs() + eps) - math.sqrt(eps)
    if noise_option == 8:  # nonlipschitz continuity
        return _safe_cube(y)
    if noise_option == 9:  # nonlinear (sigmoid)
        return y.sigmoid()
    if noise_option == 10:  # nonlinear (relu)
        return y.relu()
    if noise_option == 11:  # complex
        return t * y

    # Neural Network (linear / nonlinear)
    if noise_option == 12:  # NN(t)
        return params.noise_t(time_features)
    if noise_option == 13:  # NN(t) * y
        return params.noise_t(time_features) * y
    if noise_option == 14:  # NN(t,y)
        return params.noise_y(torch.cat([time_features, y], dim=-1))
    if noise_option == 15:  # NN(t&y) * y
        return params.noise_y(torch.cat([time_features, y], dim=-1)) * y
    if noise_option == 16:  # 2NN(t)
        return params.noise_t(time_features).relu()
    if noise_option == 17:  # 2NN(t) * y
        return params.noise_t(time_features).relu() * y
    if noise_option == 18:  # 2NN(t,y)
        return params.noise_y(torch.cat([time_features, y], dim=-1)).relu()
    if noise_option == 19:  # 2NN(t,y) * y
        return params.noise_y(torch.cat([time_features, y], dim=-1)).relu() * y

    if noise_option == 20:  # log : domain-safe log1p(|y|)
        return torch.log1p(y.abs())
    if noise_option == 21:  # e^y : clamp exponent to avoid inf before the outer tanh
        return torch.exp(y.clamp(max=_EXP_STATE_MAX))
    if noise_option == 22:  # NN(t) * NN(y)
        return (
            params.noise_t(time_features).relu()
            * params.noise_state(y).relu()
        )
    if noise_option == 23:  # NN(t) + NN(y)
        return (
            params.noise_t(time_features).relu()
            + params.noise_state(y).relu()
        )
    raise ValueError(f"Unknown noise_option {noise_option}.")


def _validate_mixture_options(mixture_options, *, name="mixture_options"):
    """Return a tuple of three catalogue options for the mixture diffusion."""
    if mixture_options is None:
        mixture_options = _MIXTURE_DEFAULT_OPTIONS
    try:
        options = tuple(int(option) for option in mixture_options)
    except Exception as error:
        raise ValueError(
            f"{name} must be an iterable of 3 integers; got {mixture_options!r}."
        ) from error
    if len(options) != 3:
        raise ValueError(
            f"{name} must contain exactly 3 options; got {len(options)}."
        )
    for option in options:
        if option in (0, 24):
            raise ValueError(
                f"Mixture components cannot be option {option}; "
                "use options 1-23."
            )
        if option not in range(1, 24):
            raise ValueError(
                f"Invalid mixture component option {option}; must be in 1-23."
            )
    return options


def _mixture_square_clamp_limit(y):
    """Clamp each component before squaring so the sum cannot overflow."""
    return math.sqrt(torch.finfo(y.dtype).max / _MIXTURE_CLAMP_DIVISOR)


class _DiffusionComponent(torch.nn.Module):
    """One catalogue option with its own parameters, usable as a mixture slot."""

    def __init__(self, noise_option, hidden_channels, sigma=1.0):
        super().__init__()
        self.noise_option = _validated_integer_option(
            noise_option,
            name="mixture component noise_option",
            valid_values=range(1, 24),
        )
        self.hidden_channels = hidden_channels
        _make_diffusion_parameters(self, self.noise_option, hidden_channels, sigma)

    def forward(self, t, y, time_features, diffusion_eps):
        return _raw_diffusion_for_option(
            self.noise_option, self, t, y, time_features, diffusion_eps
        )


class Diffusion_model(torch.nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        hidden_hidden_channels,
        num_hidden_layers,
        theta=1.0,
        sigma=1.0,
        input_option=0,
        noise_option=0,
        diffusion_eps=_DEFAULT_DIFFUSION_EPS,
        mixture_options=None,
        mixture_eps=_DEFAULT_MIXTURE_EPS,
    ):
        """
        Runtime vector field shared by benchmark LSDE/LNSDE/GSDE variants.

        The benchmark layer deliberately keeps finite-horizon `sin/cos(t)`
        features and outer `tanh` clipping for solver stability. Those choices
        are acceptable engineering workarounds here, but they are not the same
        thing as the paper's pure LSDE/LNSDE/GSDE parameterizations.

        Proposal-method contract preserved across benchmark and `torch_ists`:
        LSDE=(2, 16), LNSDE=(4, 17), GSDE=(6, 17).

        With ``noise_option == 24`` a mixture diffusion is used:
        ``g = s * sqrt(pi1*g1^2 + pi2*g2^2 + pi3*g3^2)`` where the component
        functions ``g_i`` are selected from the catalogue by
        ``mixture_options`` (default 16/23/6) and the weights ``pi`` come from
        a learnable softmax over three gates.
        """
        super().__init__()
        input_option = _validated_integer_option(
            input_option,
            name="input_option",
            valid_values=range(7),
        )
        noise_option = _validated_integer_option(
            noise_option,
            name="noise_option",
            valid_values=DIFFUSION_SPECS,
        )
        if (
            input_option in {0, 2, 4, 6}
            and hidden_channels != hidden_hidden_channels
        ):
            raise ValueError(
                f"input_option={input_option} requires hidden_channels == "
                "hidden_hidden_channels in the current drift architecture; "
                f"got {hidden_channels} and {hidden_hidden_channels}."
            )
        if not math.isfinite(diffusion_eps) or diffusion_eps <= 0:
            raise ValueError(
                "diffusion_eps must be a positive finite number; "
                f"got {diffusion_eps!r}."
            )
        if noise_option == 24:
            mixture_options = _validate_mixture_options(mixture_options)
        elif mixture_options is not None:
            raise ValueError(
                "mixture_options is only valid with noise_option=24."
            )
        if not math.isfinite(mixture_eps) or mixture_eps <= 0:
            raise ValueError(
                "mixture_eps must be a positive finite number; "
                f"got {mixture_eps!r}."
            )

        self.sde_type = "ito"
        self.noise_type = "diagonal" # or "scalar"
        self.input_option = input_option
        self.noise_option = noise_option
        self.diffusion_eps = float(diffusion_eps)

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # network
        self.initial_network = torch.nn.Linear(input_channels, hidden_channels)

        if self.input_option in [3,4,5,6]: # for time embedding
            self.linear_in = torch.nn.Linear(hidden_channels+2, hidden_hidden_channels)
        else:
            self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)

        if self.input_option in [2,4,6]: # for control embedding
            self.emb = torch.nn.Linear(hidden_channels*2, hidden_channels)
        else:
            pass
            
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels)

        # parameter
        self.theta = torch.nn.Parameter(torch.tensor([[theta]]), requires_grad=True) # scaling factor

        if self.noise_option == 24:
            self.mixture_options = mixture_options
            self.mixture_components = torch.nn.ModuleList(
                _DiffusionComponent(option, hidden_channels, sigma=sigma)
                for option in mixture_options
            )
            self.mixture_logits = torch.nn.Parameter(
                torch.zeros(3), requires_grad=True
            )
            self.mixture_log_scale = torch.nn.Parameter(
                torch.tensor(0.0), requires_grad=True
            )
            self.mixture_eps = float(mixture_eps)
        else:
            _make_diffusion_parameters(self, self.noise_option, hidden_channels, sigma)

    def set_X(self, coeffs, times):
        self.coeffs = coeffs
        self.times = times
        self.X = torchcde.CubicSpline(self.coeffs, self.times)

    def _ensure_time_tensor(self, t, y):
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, dtype=y.dtype, device=y.device)
        else:
            t = t.to(dtype=y.dtype, device=y.device)

        batch_size = y.size(0)
        if t.dim() == 0:
            return t.reshape(1, 1).expand(batch_size, 1)
        if t.dim() == 1:
            if t.numel() == 1:
                return t.reshape(1, 1).expand(batch_size, 1)
            if t.size(0) == batch_size:
                return t.unsqueeze(-1)
        if t.dim() == 2:
            if tuple(t.shape) == (1, 1):
                return t.expand(batch_size, 1)
            if tuple(t.shape) == (batch_size, 1):
                return t

        raise ValueError(
            "t must be scalar or have shape [1], [batch], or [batch, 1]; "
            f"got {tuple(t.shape)} for batch size {batch_size}."
        )

    def _bounded_time_features(self, t, y):
        t = self._ensure_time_tensor(t, y)
        return t, torch.cat((torch.sin(t), torch.cos(t)), dim=-1)

    def _build_drift_inputs(self, t, y, Xt):
        # Runtime-only finite-horizon time conditioning. The LNSDE/GSDE-style
        # variants use bounded sin/cos(t) features here for stability on
        # normalized tasks, which is acceptable in benchmarks but not a
        # theorem-faithful pure asymptotic construction.
        if self.input_option in [3,4,5,6]:
            _, time_features = self._bounded_time_features(t, y)
            yy = self.linear_in(torch.cat((time_features, y), dim=-1))
        else:
            yy = self.linear_in(y)

        if self.input_option == 0: # use control only
            return Xt
        if self.input_option in [1,3,5]: # use latent
            return yy
        return self.emb(torch.cat([yy,Xt], dim=-1))

    def _run_shared_mlp(self, z):
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        return self.linear_out(z)

    def _apply_geometric_interaction(self, z, y):
        if self.input_option in [5,6]: # geometric
            # Runtime GSDE heuristic: keep the geometric interaction, but accept
            # that the later clipping turns it into a stabilized discrete-time
            # approximation rather than a clean positivity-preserving proof path.
            return z * y.tanh() # z = z * (1 - torch.nan_to_num(y).sigmoid())
        return z

    def _clip_drift(self, z):
        # Runtime drift clipping. LSDE mainly uses this as a practical
        # anti-blow-up device, while LNSDE/GSDE accept a gap to the pure
        # theorems in exchange for bounded finite-horizon behavior.
        return z.tanh()

    @staticmethod
    def _positive_scale(log_scale):
        return _positive_scale(log_scale)

    @staticmethod
    def _safe_cube(y):
        return _safe_cube(y)

    def _raw_diffusion(self, t, y):
        # Runtime-only finite-horizon time features in diffusion. The benchmark
        # LSDE/LNSDE/GSDE choices all rely on bounded sin/cos(t) conditioning
        # somewhere in diffusion; for LNSDE this is the main mismatch with the
        # paper's infinite-horizon asymptotic story.
        t, time_features = self._bounded_time_features(t, y)

        if self.noise_option == 24:
            gates = torch.softmax(self.mixture_logits, dim=0)
            clamp_limit = _mixture_square_clamp_limit(y)
            weighted_squares = torch.zeros_like(y)
            for gate, component in zip(gates, self.mixture_components):
                raw_component = component(t, y, time_features, self.diffusion_eps)
                raw_component = raw_component.clamp(
                    min=-clamp_limit, max=clamp_limit
                )
                weighted_squares = weighted_squares + gate * raw_component.square()
            scale = _positive_scale(self.mixture_log_scale)
            return scale * torch.sqrt(weighted_squares + self.mixture_eps)

        return _raw_diffusion_for_option(
            self.noise_option, self, t, y, time_features, self.diffusion_eps
        )

    def _clip_diffusion(self, noise):
        # Runtime diffusion clipping with the same tradeoff: stable bounded
        # training dynamics over theorem-faithful pure structure.
        return noise.tanh()
            
    def f(self, t, y):
        Xt = self.X.evaluate(t)
        Xt = self.initial_network(Xt)

        z = self._build_drift_inputs(t, y, Xt)
        z = self._run_shared_mlp(z)
        z = self._apply_geometric_interaction(z, y)
        return self._clip_drift(z)

    def g(self, t, y):
        noise = self._raw_diffusion(t, y)
        noise = self.theta.sigmoid() * noise
        return self._clip_diffusion(noise) # diagonal noise
        # return noise.unsqueeze(-1) # scalar noise
