"""Focused smoke tests for every benchmark diffusion option.

Run from ``benchmark_forecasting`` with::

    python -m unittest -v test_diffusions.py

The algebraic tests do not require torchcde/torchsde to be installed. The
short solver test is enabled automatically when torchsde is available.
"""

import importlib
import importlib.util
import json
import pathlib
import sys
import types
import unittest

import torch


HERE = pathlib.Path(__file__).resolve().parent
MODULE_PATH = HERE / "models_sde" / "neuralsde.py"


def _import_or_stub(name):
    try:
        return importlib.import_module(name), True
    except ModuleNotFoundError:
        module = types.ModuleType(name)
        sys.modules[name] = module
        return module, False


_, _TORCHCDE_AVAILABLE = _import_or_stub("torchcde")
_TORCHSDE, _TORCHSDE_AVAILABLE = _import_or_stub("torchsde")

# The diffusion model imports this legacy module but does not use it in the
# methods under test. Stubbing it keeps the algebraic suite runnable in a
# minimal PyTorch environment without torchdiffeq.
sys.modules.setdefault("controldiffeq", types.ModuleType("controldiffeq"))

_SPEC = importlib.util.spec_from_file_location(
    "forecasting_neuralsde_diffusion_tests",
    MODULE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load {MODULE_PATH}.")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

DIFFUSION_SPECS = _MODULE.DIFFUSION_SPECS
Diffusion_model = _MODULE.Diffusion_model
NeuralSDE_forecasting = _MODULE.NeuralSDE_forecasting
ZeroDiffusionSDE = _MODULE._ZeroDiffusionSDE
get_diffusion_spec = _MODULE.get_diffusion_spec


class DiffusionOptionsTest(unittest.TestCase):
    batch_size = 3
    hidden_channels = 5

    def make_model(self, noise_option, **kwargs):
        torch.manual_seed(314159)
        defaults = dict(
            input_channels=4,
            hidden_channels=self.hidden_channels,
            hidden_hidden_channels=self.hidden_channels,
            num_hidden_layers=2,
            input_option=1,
            noise_option=noise_option,
        )
        defaults.update(kwargs)
        return Diffusion_model(**defaults).double()

    def test_registry_has_zero_control_and_24_nonzero_options(self):
        self.assertEqual(set(DIFFUSION_SPECS), set(range(25)))
        self.assertEqual(
            len(
                {
                    specification["label"]
                    for specification in DIFFUSION_SPECS.values()
                }
            ),
            25,
        )

        for noise_option in range(25):
            specification = get_diffusion_spec(noise_option)
            self.assertEqual(specification["noise_option"], noise_option)
            self.assertEqual(specification["noise_type"], "diagonal")
            self.assertTrue(specification["label"])
            self.assertTrue(specification["raw_formula"])
            self.assertTrue(specification["effective_formula"])
            json.dumps(specification)

        copied = get_diffusion_spec(1)
        copied["label"] = "changed-by-caller"
        self.assertEqual(
            get_diffusion_spec(1)["label"],
            "scalar_constant",
        )

    def test_zero_diffusion_proxy_keeps_drift_and_zeros_diffusion(self):
        class ToySDE(torch.nn.Module):
            sde_type = "ito"
            noise_type = "diagonal"

            def f(self, time, state):
                return state + time

            def g(self, time, state):
                return torch.full_like(state, 3.0)

        base = ToySDE()
        proxy = ZeroDiffusionSDE(base)
        state = torch.linspace(-1.0, 1.0, steps=10).reshape(2, 5)
        time = torch.tensor(0.25)

        self.assertTrue(torch.equal(proxy.f(time, state), base.f(time, state)))
        self.assertTrue(torch.equal(proxy.g(time, state), torch.zeros_like(state)))
        self.assertIs(proxy.base_sde, base)

    def test_zero_diffusion_forward_is_rejected_during_training(self):
        class ToySDE(torch.nn.Module):
            sde_type = "ito"
            noise_type = "diagonal"

        model = NeuralSDE_forecasting(
            func=ToySDE(),
            input_channels=2,
            output_time=2,
            hidden_channels=3,
            output_channels=2,
        ).train()
        with self.assertRaisesRegex(RuntimeError, "test-only ablation"):
            model(
                torch.tensor([0.0, 1.0]),
                (),
                torch.tensor([1]),
                zero_diffusion=True,
            )

    def test_every_option_has_finite_raw_forward_and_backward(self):
        base_state = torch.tensor(
            [
                [-2.0, -1.0, 0.0, 1.0, 2.0],
                [-0.5, -0.1, 0.0, 0.1, 0.5],
                [2.5, -2.5, 0.25, -0.25, 1.5],
            ],
            dtype=torch.float64,
        )

        for noise_option in range(25):
            with self.subTest(noise_option=noise_option):
                model = self.make_model(noise_option)
                state = base_state.clone().requires_grad_(True)
                time = torch.tensor(0.75, dtype=state.dtype)

                raw = model._raw_diffusion(time, state)
                effective = model.g(time, state)
                self.assertEqual(raw.shape, state.shape)
                self.assertEqual(effective.shape, state.shape)
                self.assertEqual(raw.dtype, state.dtype)
                self.assertEqual(effective.dtype, state.dtype)
                self.assertTrue(torch.isfinite(raw).all().item())
                self.assertTrue(torch.isfinite(effective).all().item())

                gradients = torch.autograd.grad(
                    effective.square().sum(),
                    (state, *tuple(model.parameters())),
                    allow_unused=True,
                )
                self.assertTrue(
                    all(
                        gradient is None
                        or torch.isfinite(gradient).all().item()
                        for gradient in gradients
                    )
                )

    def test_input_and_noise_options_are_validated(self):
        for invalid in (-1, 25, 1.5, True, "18"):
            with self.subTest(noise_option=invalid):
                with self.assertRaises(ValueError):
                    self.make_model(invalid)
                with self.assertRaises(ValueError):
                    get_diffusion_spec(invalid)

        for invalid in (-1, 7, 1.5, True, "1"):
            with self.subTest(input_option=invalid):
                with self.assertRaises(ValueError):
                    self.make_model(1, input_option=invalid)

    def test_width_constraint_is_explicit_for_affected_drift_modes(self):
        for input_option in (0, 2, 4, 6):
            with self.subTest(input_option=input_option):
                with self.assertRaisesRegex(
                    ValueError,
                    "hidden_channels == hidden_hidden_channels",
                ):
                    self.make_model(
                        1,
                        input_option=input_option,
                        hidden_hidden_channels=self.hidden_channels + 2,
                    )

        for input_option in (1, 3, 5):
            with self.subTest(input_option=input_option):
                self.make_model(
                    1,
                    input_option=input_option,
                    hidden_hidden_channels=self.hidden_channels + 2,
                )

    def test_supported_time_shapes_are_equivalent(self):
        scalar_time = 0.375
        time_forms = (
            torch.tensor(scalar_time, dtype=torch.float64),
            torch.tensor([scalar_time], dtype=torch.float64),
            torch.full(
                (self.batch_size,),
                scalar_time,
                dtype=torch.float64,
            ),
            torch.full(
                (self.batch_size, 1),
                scalar_time,
                dtype=torch.float64,
            ),
        )
        state = torch.linspace(
            -1.0,
            1.0,
            steps=self.batch_size * self.hidden_channels,
            dtype=torch.float64,
        ).reshape(self.batch_size, self.hidden_channels)

        for noise_option in (2, 11, 14, 22, 23):
            with self.subTest(noise_option=noise_option):
                model = self.make_model(noise_option)
                expected = model._raw_diffusion(time_forms[0], state)
                for time in time_forms[1:]:
                    actual = model._raw_diffusion(time, state)
                    self.assertTrue(torch.allclose(actual, expected))

        model = self.make_model(14)
        with self.assertRaisesRegex(ValueError, "t must be scalar"):
            model._raw_diffusion(
                torch.zeros(
                    self.batch_size,
                    2,
                    dtype=torch.float64,
                ),
                state,
            )

    def test_options_22_and_23_have_independent_one_layer_networks(self):
        state = torch.zeros(
            self.batch_size,
            self.hidden_channels,
            dtype=torch.float64,
        )
        time = torch.tensor(0.25, dtype=torch.float64)

        for noise_option, expected_value in ((22, 2.0), (23, 3.0)):
            with self.subTest(noise_option=noise_option):
                model = self.make_model(noise_option)
                self.assertIsInstance(model.noise_t, torch.nn.Linear)
                self.assertIsInstance(model.noise_state, torch.nn.Linear)
                self.assertEqual(model.noise_t.in_features, 2)
                self.assertEqual(
                    model.noise_state.in_features,
                    self.hidden_channels,
                )

                with torch.no_grad():
                    model.noise_t.weight.zero_()
                    model.noise_t.bias.fill_(1.0)
                    model.noise_state.weight.zero_()
                    model.noise_state.bias.fill_(2.0)

                raw = model._raw_diffusion(time, state)
                self.assertTrue(
                    torch.allclose(
                        raw,
                        torch.full_like(raw, expected_value),
                    )
                )

    def test_extreme_inputs_remain_finite(self):
        cases = {
            7: torch.tensor(
                [[-1e30, -1e10, 0.0, 1e10, 1e30]] * self.batch_size,
                dtype=torch.float64,
            ),
            8: torch.tensor(
                [[-1e300, -1e100, 0.0, 1e100, 1e300]]
                * self.batch_size,
                dtype=torch.float64,
            ),
            21: torch.tensor(
                [[-1e300, -1e10, 0.0, 1e10, 1e300]]
                * self.batch_size,
                dtype=torch.float64,
            ),
        }

        for noise_option, base_state in cases.items():
            with self.subTest(noise_option=noise_option):
                model = self.make_model(noise_option)
                state = base_state.clone().requires_grad_(True)
                raw = model._raw_diffusion(torch.tensor(49.0), state)
                effective = model.g(torch.tensor(49.0), state)
                self.assertTrue(torch.isfinite(raw).all().item())
                self.assertTrue(torch.isfinite(effective).all().item())
                gradient = torch.autograd.grad(
                    effective.sum(),
                    state,
                    allow_unused=True,
                )[0]
                self.assertTrue(
                    gradient is None
                    or torch.isfinite(gradient).all().item()
                )

        for noise_option, attribute in ((1, "sigma"), (4, "sigma_diag")):
            with self.subTest(noise_option=noise_option):
                model = self.make_model(noise_option)
                with torch.no_grad():
                    getattr(model, attribute).fill_(1e300)
                state = torch.ones(
                    self.batch_size,
                    self.hidden_channels,
                    dtype=torch.float64,
                )
                self.assertTrue(
                    torch.isfinite(
                        model._raw_diffusion(torch.tensor(1.0), state)
                    ).all().item()
                )
                self.assertTrue(
                    torch.isfinite(
                        model.g(torch.tensor(1.0), state)
                    ).all().item()
                )

    @unittest.skipUnless(
        _TORCHSDE_AVAILABLE,
        "torchsde is not installed in this environment",
    )
    def test_short_torchsde_integration_for_every_option(self):
        class DiffusionOnlySDE(torch.nn.Module):
            sde_type = "ito"
            noise_type = "diagonal"

            def __init__(self, diffusion):
                super().__init__()
                self.diffusion = diffusion

            def f(self, time, state):
                return torch.zeros_like(state)

            def g(self, time, state):
                return self.diffusion.g(time, state)

        initial_state = torch.linspace(
            -0.5,
            0.5,
            steps=self.batch_size * self.hidden_channels,
        ).reshape(self.batch_size, self.hidden_channels)
        times = torch.tensor([0.0, 0.05, 0.1])

        for noise_option in range(25):
            with self.subTest(noise_option=noise_option):
                model = self.make_model(noise_option).float()
                path = _TORCHSDE.sdeint(
                    DiffusionOnlySDE(model),
                    initial_state,
                    times,
                    method="euler",
                    dt=0.01,
                )
                self.assertEqual(
                    path.shape,
                    (
                        times.numel(),
                        self.batch_size,
                        self.hidden_channels,
                    ),
                )
                self.assertTrue(torch.isfinite(path).all().item())

    @unittest.skipUnless(
        _TORCHSDE_AVAILABLE,
        "torchsde is not installed in this environment",
    )
    def test_zero_diffusion_solver_path_is_seed_invariant(self):
        class ToySDE(torch.nn.Module):
            sde_type = "ito"
            noise_type = "diagonal"

            def f(self, time, state):
                return -0.25 * state

            def g(self, time, state):
                return torch.ones_like(state)

        proxy = ZeroDiffusionSDE(ToySDE())
        initial_state = torch.linspace(
            -0.5,
            0.5,
            steps=self.batch_size * self.hidden_channels,
        ).reshape(self.batch_size, self.hidden_channels)
        times = torch.tensor([0.0, 0.05, 0.1])

        torch.manual_seed(1)
        first = _TORCHSDE.sdeint(
            proxy,
            initial_state,
            times,
            method="euler",
            dt=0.01,
        )
        torch.manual_seed(987654)
        second = _TORCHSDE.sdeint(
            proxy,
            initial_state,
            times,
            method="euler",
            dt=0.01,
        )
        self.assertTrue(torch.equal(first, second))


    def test_mixture_default_components_match_defaults(self):
        model = self.make_model(24)
        self.assertEqual(model.mixture_options, (16, 23, 6))
        self.assertEqual(len(model.mixture_components), 3)

        component0 = model.mixture_components[0]
        self.assertEqual(component0.noise_option, 16)
        self.assertIsInstance(component0.noise_t, torch.nn.Sequential)

        component1 = model.mixture_components[1]
        self.assertEqual(component1.noise_option, 23)
        self.assertIsInstance(component1.noise_t, torch.nn.Linear)
        self.assertIsInstance(component1.noise_state, torch.nn.Linear)

        component2 = model.mixture_components[2]
        self.assertEqual(component2.noise_option, 6)
        self.assertIsInstance(component2.sigma_diag, torch.nn.Parameter)

    def test_mixture_raw_forward_independent_of_gates_for_repeated_component(self):
        model = self.make_model(24, mixture_options=(9, 9, 9))
        state = torch.tensor(
            [[-2.0, -1.0, 0.0, 1.0, 2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        time = torch.tensor(0.5, dtype=torch.float64)

        with torch.no_grad():
            torch.nn.init.normal_(model.mixture_logits, mean=0.0, std=2.0)
        raw = model._raw_diffusion(time, state)
        expected = state.sigmoid()
        self.assertTrue(torch.allclose(raw, expected))

    def test_mixture_forward_backward_is_finite(self):
        model = self.make_model(24)
        state = torch.tensor(
            [
                [-2.0, -1.0, 0.0, 1.0, 2.0],
                [-0.5, -0.1, 0.0, 0.1, 0.5],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        time = torch.tensor(0.75, dtype=torch.float64)

        raw = model._raw_diffusion(time, state)
        effective = model.g(time, state)
        self.assertEqual(raw.shape, state.shape)
        self.assertEqual(effective.shape, state.shape)
        self.assertTrue(torch.isfinite(raw).all().item())
        self.assertTrue(torch.isfinite(effective).all().item())

        gradients = torch.autograd.grad(
            effective.square().sum(),
            (state, *tuple(model.parameters())),
            allow_unused=True,
        )
        self.assertTrue(
            all(
                gradient is None
                or torch.isfinite(gradient).all().item()
                for gradient in gradients
            )
        )

    def test_mixture_learnable_gates_are_uniform_at_init(self):
        model = self.make_model(24)
        gates = torch.softmax(model.mixture_logits, dim=0)
        expected = torch.full((3,), 1.0 / 3.0, dtype=torch.float64)
        self.assertTrue(torch.allclose(gates, expected))

    def test_mixture_scale_is_positive_at_init(self):
        model = self.make_model(24)
        scale = model._positive_scale(model.mixture_log_scale)
        self.assertAlmostEqual(scale.item(), 1.0)
        self.assertGreater(scale.item(), 0.0)

    def test_mixture_options_are_validated(self):
        invalid_cases = (
            (0, 1, 2),
            (1, 24, 2),
            (1, 2, 25),
            (1, 2),
            (1, 2, 3, 4),
            True,
        )
        for invalid in invalid_cases:
            with self.subTest(mixture_options=invalid):
                with self.assertRaises(ValueError):
                    self.make_model(24, mixture_options=invalid)

        with self.assertRaises(ValueError):
            self.make_model(1, mixture_options=(7, 8, 9))

    def test_mixture_extreme_inputs_remain_finite(self):
        model = self.make_model(24, mixture_options=(8, 8, 8))
        state = torch.tensor(
            [[-1e300, -1e100, 0.0, 1e100, 1e300]] * self.batch_size,
            dtype=torch.float64,
            requires_grad=True,
        )
        raw = model._raw_diffusion(torch.tensor(49.0), state)
        effective = model.g(torch.tensor(49.0), state)
        self.assertTrue(torch.isfinite(raw).all().item())
        self.assertTrue(torch.isfinite(effective).all().item())
        gradient = torch.autograd.grad(
            effective.sum(),
            state,
            allow_unused=True,
        )[0]
        self.assertTrue(
            gradient is None
            or torch.isfinite(gradient).all().item()
        )


if __name__ == "__main__":
    unittest.main()
