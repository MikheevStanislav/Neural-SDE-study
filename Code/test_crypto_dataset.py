"""Regression tests for the bundled cryptocurrency forecasting adapter."""

import importlib.util
import math
import pathlib
import sys
import tempfile
import unittest
import zipfile
from datetime import date, timedelta

import numpy as np
import torch


HERE = pathlib.Path(__file__).resolve().parent
MODULE_PATH = HERE / "datasets" / "crypto.py"


def _load_module():
    name = "forecasting_crypto_dataset_tests"
    specification = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


crypto = _load_module()


class CryptoArchiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.windows20 = crypto._build_window_splits(50, 20)

    def test_bundled_archive_schema_hash_and_asset_count(self):
        details = crypto.validate_source()
        self.assertEqual(details["asset_count"], 23)
        self.assertEqual(details["input_features"], 6)
        self.assertEqual(details["output_features"], 6)
        self.assertEqual(
            details["archive_sha256"],
            "44b63683c0ef967adcf1bbc1f029e407cb372f1c526bff9226a0877d7562a933",
        )
        self.assertEqual(details["last_date"], "2021-07-06")

    def test_stationary_feature_formulas(self):
        days = tuple(date(2021, 1, 1) + timedelta(days=index) for index in range(3))
        values = np.asarray(
            [
                [10.0, 12.0, 9.0, 11.0, 100.0, 1000.0],
                [12.0, 14.0, 10.0, 13.0, 150.0, 1200.0],
                [13.0, 15.0, 12.0, 14.0, 180.0, 1400.0],
            ]
        )
        asset = crypto._AssetSeries("Example", "EX", days, values)
        _, present, actual = crypto._feature_engineering(asset)
        self.assertTrue(present.all())
        expected = np.asarray(
            [
                math.log(12.0 / 11.0),
                math.log(14.0 / 11.0),
                math.log(10.0 / 11.0),
                math.log(13.0 / 11.0),
                math.log1p(150.0) - math.log1p(100.0),
                math.log(1200.0 / 1000.0),
            ]
        )
        np.testing.assert_allclose(actual[1], expected)

    def test_zero_volume_and_marketcap_become_nan_not_infinity(self):
        days = (date(2021, 1, 1), date(2021, 1, 2))
        values = np.asarray(
            [
                [10.0, 11.0, 9.0, 10.0, 0.0, 1000.0],
                [10.0, 11.0, 9.0, 10.0, 100.0, 0.0],
            ]
        )
        asset = crypto._AssetSeries("Example", "EX", days, values)
        _, _, features = crypto._feature_engineering(asset)
        self.assertTrue(math.isnan(float(features[1, 4])))
        self.assertTrue(math.isnan(float(features[1, 5])))
        self.assertFalse(np.isinf(features).any())

    def test_default_window_counts_and_shapes(self):
        expected = {"train": 31254, "val": 1610, "test": 1104}
        self.assertEqual(self.windows20["metadata"]["window_counts"], expected)
        for split_name, count in expected.items():
            split = self.windows20[split_name]
            self.assertEqual(tuple(split["inputs"].shape), (count, 50, 6))
            self.assertEqual(tuple(split["targets"].shape), (count, 20, 6))
            self.assertEqual(split["asset_ids"].unique().numel(), 23)
        self.assertTrue(torch.isfinite(self.windows20["train"]["inputs"]).all())

    def test_horizon_40_is_supported(self):
        windows = crypto._build_window_splits(50, 40)
        self.assertEqual(
            windows["metadata"]["window_counts"],
            {"train": 30794, "val": 1150, "test": 644},
        )
        self.assertEqual(tuple(windows["test"]["targets"].shape), (644, 40, 6))

    def test_target_dates_do_not_cross_or_overlap_splits(self):
        horizon = 20
        occupied = {}
        for split_name in ("train", "val", "test"):
            starts = self.windows20[split_name]["target_start_ordinals"].tolist()
            occupied[split_name] = {
                ordinal + offset
                for ordinal in starts
                for offset in range(horizon)
            }
        self.assertTrue(occupied["train"].isdisjoint(occupied["val"]))
        self.assertTrue(occupied["train"].isdisjoint(occupied["test"]))
        self.assertTrue(occupied["val"].isdisjoint(occupied["test"]))
        self.assertLessEqual(max(occupied["train"]), crypto.TRAIN_END.toordinal())
        self.assertGreaterEqual(min(occupied["val"]), crypto.VAL_START.toordinal())
        self.assertGreaterEqual(min(occupied["test"]), crypto.TEST_START.toordinal())

    def test_corruption_is_seeded_and_never_fills_missing_values(self):
        inputs = self.windows20["test"]["inputs"][:8]
        first = crypto._apply_missingness(inputs, 0.4, "random", 123)
        second = crypto._apply_missingness(inputs, 0.4, "random", 123)
        different = crypto._apply_missingness(inputs, 0.4, "random", 124)
        self.assertTrue(torch.allclose(first, second, equal_nan=True))
        self.assertFalse(torch.allclose(first, different, equal_nan=True))

        noisy = crypto._add_scaled_input_noise(
            first,
            crypto._finite_channel_std(self.windows20["train"]["inputs"]),
            0.1,
            123,
        )
        self.assertTrue(torch.equal(torch.isnan(first), torch.isnan(noisy)))
        # Missingness removes complete timestamps across all six coordinates.
        self.assertTrue(
            torch.equal(torch.isnan(first).all(dim=-1), torch.isnan(first).any(dim=-1))
        )

    def test_loader_batch_contract_and_sequential_evaluation(self):
        count, steps, channels, horizon = 5, 4, 6, 2
        split = {
            "coefficients": tuple(
                torch.full((count, steps - 1, channels), float(index))
                for index in range(4)
            ),
            "targets": torch.arange(count * horizon * channels, dtype=torch.float32)
            .reshape(count, horizon, channels),
            "asset_ids": torch.arange(count),
            "target_start_ordinals": torch.arange(count) + 730000,
        }
        times = torch.arange(steps, dtype=torch.float32)
        loader = crypto._make_loader(split, times, batch_size=3, shuffle=False)
        batches = list(loader)
        self.assertEqual(len(batches), 2)
        self.assertEqual(tuple(batches[0][-2].shape), (3, horizon, channels))
        reconstructed = torch.cat([batch[-2] for batch in batches])
        self.assertTrue(torch.equal(reconstructed, split["targets"]))
        self.assertFalse(loader.drop_last)
        self.assertEqual(loader.dataset.asset_ids.tolist(), list(range(count)))

    def test_append_time_keeps_six_output_coordinates(self):
        inputs = torch.zeros(3, 5, 6)
        augmented = crypto._append_time_channel(inputs)
        self.assertEqual(tuple(augmented.shape), (3, 5, 7))
        self.assertEqual(crypto.NUM_OUTPUT_FEATURES, 6)

    def test_unsafe_zip_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = pathlib.Path(directory) / "unsafe.zip"
            with zipfile.ZipFile(temporary, "w") as archive:
                archive.writestr("../escape.csv", "SNo\n1\n")
            with zipfile.ZipFile(temporary) as archive:
                with self.assertRaisesRegex(ValueError, "Unsafe path"):
                    crypto._safe_csv_members(archive)


class CryptoCliTest(unittest.TestCase):
    def test_parser_accepts_crypto_and_batch_size(self):
        path = HERE / "parse.py"
        specification = importlib.util.spec_from_file_location(
            "forecasting_crypto_parse_tests", path
        )
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        previous = sys.argv
        try:
            sys.argv = [
                "test",
                "--dataset_name",
                "crypto",
                "--batch_size",
                "256",
            ]
            arguments = module.parse_args()
        finally:
            sys.argv = previous
        self.assertEqual(arguments.dataset_name, "crypto")
        self.assertEqual(arguments.batch_size, 256)


if __name__ == "__main__":
    unittest.main()
