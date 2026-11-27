"""Unit and smoke tests for the GBRBM continuous-feature anomaly detection case.

The example modules are imported under a temporarily extended ``sys.path`` and
the licensed simulated-annealing execution is replaced by a Metropolis sampler
that still minimizes the real Kaiwu Ising energy, so the suite never blocks on a
license prompt and never submits a cloud task.
"""

import contextlib
import importlib
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch

import kaiwu as kw

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine

SKLEARN_DATASETS = pytest.importorskip("sklearn.datasets")
EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "gbrbm_anomaly_detection"
CASE_MODULES = (
    "ad_config",
    "ad_data",
    "ad_sampler",
    "ad_scorer",
    "ad_trainer",
    "ad_eval",
    "run_anomaly_detection",
)
LICENSE_ENV_PAIRS = (
    ("USER_ID", "SDK_CODE"),
    ("LICENSE_USER_ID", "LICENSE_SDK_CODE"),
)
SA_RESTARTS = 6
SA_SWEEPS = 24
SA_START_TEMPERATURE = 1.0
SA_END_TEMPERATURE = 0.02
SA_COOLING_RATIO = (SA_END_TEMPERATURE / SA_START_TEMPERATURE) ** (1.0 / (SA_SWEEPS - 1))
OFFLINE_SA_SEED = 20241124

_CASE_CACHE: dict[str, SimpleNamespace] = {}


def _import_case_modules() -> SimpleNamespace:
    """Import the case modules once, in dependency order."""
    modules: dict[str, Any] = {}
    entry = str(EXAMPLE_DIR)
    sys.path.insert(0, entry)
    try:
        for name in CASE_MODULES:
            modules[name] = importlib.import_module(name)
        try:
            modules["ad_plot"] = importlib.import_module("ad_plot")
        except ImportError:
            modules["ad_plot"] = None
    finally:
        sys.path.remove(entry)
    return SimpleNamespace(**modules)


def _case() -> SimpleNamespace:
    """Return the imported case modules, importing them on first use."""
    if "modules" not in _CASE_CACHE:
        _CASE_CACHE["modules"] = _import_case_modules()
    return _CASE_CACHE["modules"]


def _bunch_target(bunch: Any) -> np.ndarray:
    """Read the raw target vector from a scikit-learn bunch."""
    return np.asarray(bunch.target)


def _manual_best_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    """Brute-force the best achievable F1 over every candidate threshold."""
    best = 0.0
    positives = labels == 1
    for threshold in np.unique(scores):
        predicted = scores >= threshold
        denominator = float(np.sum(predicted) + np.sum(positives))
        if denominator > 0.0:
            best = max(best, 2.0 * float(np.sum(predicted & positives)) / denominator)
    return best


def _kaiwu_credentials_present() -> bool:
    """Return whether a Kaiwu credential pair is configured in the environment."""
    return any(
        os.environ.get(user_key) and os.environ.get(code_key)
        for user_key, code_key in LICENSE_ENV_PAIRS
    )


def _anneal_once(matrix: np.ndarray, flip_delta, rng: np.random.Generator) -> np.ndarray:
    """Run one geometric-cooling Metropolis chain and return its best spins."""
    spins = rng.choice([-1.0, 1.0], size=matrix.shape[0])
    energy = float(kw.common.hamiltonian(matrix, spins[None])[0])
    best = spins.copy()
    best_energy = energy
    for step in range(SA_SWEEPS):
        temperature = SA_START_TEMPERATURE * SA_COOLING_RATIO**step
        for index in rng.permutation(matrix.shape[0]):
            delta = float(flip_delta(spins, int(index)))
            if delta <= 0.0 or rng.random() < math.exp(-delta / temperature):
                spins[index] *= -1.0
                energy += delta
        if energy < best_energy:
            best_energy = energy
            best = spins.copy()
    return best


def _metropolis_solve(instance, ising_matrix=None, init_solution=None, rand_seed=None):
    """Replace only the licensed annealing call with an SDK-energy sampler."""
    del init_solution, rand_seed
    instance.set_matrix(ising_matrix)
    matrix = np.asarray(instance.matrix, dtype=np.float64)
    flip_delta = getattr(instance, "_dlt_h_single_flip")
    rng = np.random.default_rng(OFFLINE_SA_SEED)
    candidates = np.asarray([_anneal_once(matrix, flip_delta, rng) for _ in range(SA_RESTARTS)])
    energies = kw.common.hamiltonian(matrix, candidates)
    return candidates[[int(np.argmin(energies))]]


@contextlib.contextmanager
def _sa_offline_context():
    """Keep the real SA object, solve dispatcher and energy API without license prompts."""
    if _kaiwu_credentials_present():
        yield
        return
    optimizer_class = kw.classical.SimulatedAnnealingOptimizer
    with mock.patch.object(optimizer_class, "single_process_solve", _metropolis_solve):
        yield


class _RecordingOptimizer:
    """Capture CIM constructor arguments and refuse any cloud submission."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.matrices: list[np.ndarray] = []

    def set_matrix(self, matrix) -> None:
        """Record the Ising matrix without submitting anything."""
        self.matrices.append(np.asarray(matrix))

    def solve(self, matrix=None) -> None:
        """Fail loudly instead of reaching the cloud during tests."""
        self.matrices.append(np.asarray(matrix))
        raise AssertionError("CIM submission must not happen in the offline test suite")


def _explicit_model() -> tuple[GaussianBernoulliRestrictedBoltzmannMachine, torch.Tensor]:
    """Build a float64 GBRBM with fully explicit parameters and probe inputs."""
    model = GaussianBernoulliRestrictedBoltzmannMachine(
        num_visible=3,
        num_hidden=4,
        is_visible_gaussian=True,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    weights = torch.tensor(
        [
            [0.30, -0.20, 0.10, 0.05],
            [-0.40, 0.25, 0.15, -0.10],
            [0.20, 0.35, -0.30, 0.45],
        ],
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.mu.copy_(torch.tensor([0.5, -0.2, 0.1], dtype=torch.float64))
        model.log_var.copy_(torch.log(torch.tensor([0.5, 2.0, 1.5], dtype=torch.float64)))
        model.quadratic_coef.copy_(weights)
        model.linear_bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float64))
    probe = torch.tensor(
        [
            [0.4, -0.1, 0.2],
            [-1.5, 2.0, -0.7],
            [3.0, -2.5, 1.1],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    return model, probe


class DatasetSplitTests(unittest.TestCase):
    """Leakage-free splitting and scaling behaviour."""

    def test_splits_are_disjoint_stratified_and_train_is_normal_only(self):
        """Splits cover every raw row exactly once and the train split is benign only."""
        case = _case()
        config = case.ad_config.DataConfig()
        split = case.ad_data.prepare_anomaly_dataset(config)
        merged = np.concatenate([split.train_indices, split.val_indices, split.test_indices])

        self.assertEqual(merged.size, 569)
        np.testing.assert_array_equal(np.sort(merged), np.arange(569))
        self.assertEqual(np.intersect1d(split.train_indices, split.val_indices).size, 0)
        self.assertEqual(np.intersect1d(split.train_indices, split.test_indices).size, 0)
        self.assertEqual(np.intersect1d(split.val_indices, split.test_indices).size, 0)

        raw_target = _bunch_target(SKLEARN_DATASETS.load_breast_cancer())
        self.assertTrue(np.all(raw_target[split.train_indices] == 1))
        self.assertTrue(np.all(split.y_val == np.where(raw_target[split.val_indices] == 1, 0, 1)))
        self.assertTrue(np.all(split.y_test == np.where(raw_target[split.test_indices] == 1, 0, 1)))
        self.assertEqual(sorted(np.unique(split.y_val).tolist()), [0, 1])
        self.assertEqual(sorted(np.unique(split.y_test).tolist()), [0, 1])
        self.assertLessEqual(abs(float(split.y_val.mean()) - float(split.y_test.mean())), 0.02)

        repeated = case.ad_data.prepare_anomaly_dataset(config)
        np.testing.assert_array_equal(repeated.train_indices, split.train_indices)
        np.testing.assert_array_equal(repeated.val_indices, split.val_indices)
        np.testing.assert_array_equal(repeated.test_indices, split.test_indices)
        np.testing.assert_array_equal(repeated.x_train, split.x_train)
        np.testing.assert_array_equal(repeated.x_test, split.x_test)

    def test_scaler_fits_exclusively_on_normal_training_split(self):
        """Extreme values in validation and test rows never reach the fitted scaler."""
        case = _case()
        rng = np.random.default_rng(7)
        features = rng.normal(size=(100, 3))
        labels = np.concatenate([np.zeros(60, dtype=np.int64), np.ones(40, dtype=np.int64)])
        names = ("first", "second", "third")
        config = case.ad_config.DataConfig()

        base = case.ad_data.split_and_scale_dataset(features, labels, names, config)
        injected = features.copy()
        injected[base.val_indices] = 1e5
        injected[base.test_indices] = 1e5
        touched = case.ad_data.split_and_scale_dataset(injected, labels, names, config)

        train_rows = features[base.train_indices]
        np.testing.assert_allclose(touched.scaler.mean_, train_rows.mean(axis=0), atol=1e-6, rtol=0)
        np.testing.assert_allclose(touched.scaler.scale_, train_rows.std(axis=0), atol=1e-6, rtol=0)
        np.testing.assert_array_equal(touched.x_train, base.x_train)
        np.testing.assert_array_equal(touched.val_indices, base.val_indices)
        np.testing.assert_array_equal(touched.test_indices, base.test_indices)
        self.assertLess(float(np.abs(base.x_test).max()), 10.0)
        self.assertGreater(float(np.abs(touched.x_test).max()), 100.0)
        self.assertEqual(touched.feature_names, names)


class AnomalyScorerTests(unittest.TestCase):
    """Score direction, exact formulas and device/dtype alignment."""

    def test_free_energy_and_reconstruction_match_manual_formulas(self):
        """Both scores reproduce the closed-form reference computation exactly."""
        case = _case()
        model, probe = _explicit_model()
        scorer = case.ad_scorer.GBRBMAnomalyScorer(model)

        free_energy = scorer.score_free_energy(probe)
        reconstruction = scorer.score_reconstruction_error(probe)
        variance = model.log_var.exp().clip(min=model.eps)
        field = (probe / variance) @ model.quadratic_coef + model.linear_bias
        expected_free_energy = (
            0.5 * ((probe - model.mu).square() / variance).sum(dim=1)
            - torch.nn.functional.softplus(field).sum(dim=1)
        )
        probability = torch.sigmoid(field)
        decoded = probability @ model.quadratic_coef.t() + model.mu
        expected_reconstruction = (probe - decoded).square().mean(dim=1)

        torch.testing.assert_close(free_energy, expected_free_energy, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(reconstruction, expected_reconstruction, rtol=1e-12, atol=1e-12)
        for scores in (free_energy, reconstruction):
            self.assertEqual(tuple(scores.shape), (probe.shape[0],))
            self.assertFalse(scores.requires_grad)
            self.assertTrue(bool(torch.isfinite(scores).all()))

        for index in range(probe.shape[0]):
            single = scorer.score_free_energy(probe[index : index + 1])
            self.assertEqual(single.shape, (1,))
            torch.testing.assert_close(single[0], free_energy[index], rtol=1e-12, atol=1e-12)

    def test_anomaly_score_direction_higher_for_outliers(self):
        """Outliers score strictly higher than inliers and free energy grows with distance."""
        case = _case()
        case.ad_trainer.set_global_seed(11)
        model = GaussianBernoulliRestrictedBoltzmannMachine(
            num_visible=6,
            num_hidden=5,
            is_visible_gaussian=True,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        scorer = case.ad_scorer.GBRBMAnomalyScorer(model)
        generator = torch.Generator().manual_seed(11)
        center = model.mu.detach()
        normal = center + 0.05 * torch.randn((16, 6), generator=generator)
        direction = torch.linspace(-1.0, 1.0, 6).unsqueeze(0)
        anomaly = center + 6.0 * direction

        for method in ("free_energy", "reconstruction"):
            normal_scores = scorer.score(normal, method)
            anomaly_scores = scorer.score(anomaly, method)
            self.assertGreater(float(anomaly_scores.min()), float(normal_scores.max()))

        radius_mean = []
        for scale in (0.5, 1.0, 2.0, 4.0, 8.0):
            probe = center + scale * direction
            radius_mean.append(float(scorer.score_free_energy(probe).mean()))
        self.assertTrue(np.all(np.diff(radius_mean) > 0.0))

    def test_scorer_device_and_dtype_consistency(self):
        """Scores follow the model device and dtype for NumPy and tensor inputs."""
        case = _case()
        model32 = GaussianBernoulliRestrictedBoltzmannMachine(
            num_visible=4,
            num_hidden=3,
            is_visible_gaussian=True,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        scorer32 = case.ad_scorer.GBRBMAnomalyScorer(model32)
        numpy_scores = scorer32.score_numpy(np.full((5, 4), 0.25, dtype=np.float64))
        self.assertEqual(numpy_scores.dtype, np.float64)
        self.assertEqual(numpy_scores.shape, (5,))
        self.assertEqual(scorer32.score(torch.ones(5, 4, dtype=torch.float64)).dtype, torch.float32)

        model64 = GaussianBernoulliRestrictedBoltzmannMachine(
            num_visible=4,
            num_hidden=3,
            is_visible_gaussian=True,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        scorer64 = case.ad_scorer.GBRBMAnomalyScorer(model64)
        scores64 = scorer64.score(torch.ones(2, 4, dtype=torch.float32))
        self.assertEqual(scores64.dtype, torch.float64)
        self.assertEqual(scores64.device.type, "cpu")

        if torch.cuda.is_available():
            model64.to("cuda")
            self.assertEqual(model64.device.type, "cuda")
            cuda_scores = scorer64.score(np.zeros((3, 4), dtype=np.float32))
            self.assertEqual(cuda_scores.device.type, "cuda")
            self.assertEqual(cuda_scores.dtype, torch.float64)

    def test_scorer_rejects_invalid_inputs_and_methods(self):
        """Malformed inputs, unknown methods and Bernoulli-visible models are rejected."""
        case = _case()
        model, _ = _explicit_model()
        scorer = case.ad_scorer.GBRBMAnomalyScorer(model)
        invalid = (
            np.zeros(3),
            np.zeros((2, 2, 3)),
            np.zeros((0, 3)),
            np.zeros((4, 4)),
            np.array([[0.0, np.nan, 0.0]]),
            np.array([[0.0, np.inf, 0.0]]),
        )
        for inputs in invalid:
            with self.subTest(shape=np.shape(inputs)):
                with self.assertRaises(ValueError):
                    scorer.score(inputs)
                with self.assertRaises(ValueError):
                    scorer.score_reconstruction_error(inputs)
        with self.assertRaises(ValueError):
            scorer.score(np.zeros((4, 3)), "unknown")
        with self.assertRaises(ValueError):
            case.ad_scorer.GBRBMAnomalyScorer(
                GaussianBernoulliRestrictedBoltzmannMachine(
                    num_visible=4,
                    num_hidden=3,
                    is_visible_gaussian=False,
                    dtype=torch.float64,
                    device=torch.device("cpu"),
                )
            )


class ArgumentGuardTests(unittest.TestCase):
    """Configuration and evaluation argument validation."""

    def test_config_and_eval_reject_invalid_arguments(self):
        """Invalid ratios, sampler names, devices and single-class labels raise."""
        case = _case()
        features = np.zeros((20, 2))
        labels = np.concatenate([np.zeros(12, dtype=np.int64), np.ones(8, dtype=np.int64)])
        names = ("a", "b")
        for ratio in (0.0, 1.0, -0.5, 1.5):
            with self.subTest(train_normal_ratio=ratio):
                with self.assertRaises(ValueError):
                    case.ad_data.split_and_scale_dataset(
                        features,
                        labels,
                        names,
                        case.ad_config.DataConfig(train_normal_ratio=ratio),
                    )
        for ratio in (0.0, 1.0):
            with self.subTest(val_ratio_of_eval=ratio):
                with self.assertRaises(ValueError):
                    case.ad_data.split_and_scale_dataset(
                        features,
                        labels,
                        names,
                        case.ad_config.DataConfig(val_ratio_of_eval=ratio),
                    )
        with self.assertRaises(ValueError):
            case.ad_data.split_and_scale_dataset(
                np.full((20, 2), np.nan),
                labels,
                names,
                case.ad_config.DataConfig(),
            )
        with self.assertRaises(ValueError):
            case.ad_data.split_and_scale_dataset(
                np.zeros(20),
                labels,
                names,
                case.ad_config.DataConfig(),
            )
        with self.assertRaises(ValueError):
            case.ad_data.split_and_scale_dataset(
                features,
                labels[:-1],
                names,
                case.ad_config.DataConfig(),
            )
        with self.assertRaises(ValueError):
            case.ad_data.split_and_scale_dataset(
                features,
                np.zeros(20, dtype=np.int64),
                names,
                case.ad_config.DataConfig(),
            )
        with self.assertRaises(ValueError):
            case.ad_sampler.build_sampler(case.ad_config.SamplerConfig(sampler_type="bogus"))
        if not torch.cuda.is_available():
            with self.assertRaises(ValueError):
                case.ad_trainer.resolve_device("cuda")
        with self.assertRaises(ValueError):
            case.ad_trainer.resolve_device("tpu")
        with self.assertRaises(ValueError):
            case.ad_eval.select_best_f1_threshold(
                np.array([0.1, 0.2, 0.3]),
                np.zeros(3, dtype=np.int64),
            )


class EvaluationTests(unittest.TestCase):
    """Threshold selection, metric computation and the IsolationForest baseline."""

    def test_threshold_selection_and_metrics_correctness(self):
        """The threshold maximizes validation F1 and is reused unchanged on test."""
        case = _case()
        perfect = np.array([0.05, 0.15, 0.25, 0.85, 0.95, 1.05])
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        threshold, best_f1 = case.ad_eval.select_best_f1_threshold(perfect, labels)
        self.assertAlmostEqual(best_f1, 1.0, places=12)
        self.assertGreater(threshold, 0.25)
        self.assertLessEqual(threshold, 0.85)

        overlapping = np.array([0.05, 0.90, 0.20, 0.80, 0.30, 0.70, 0.40, 0.60])
        mixed = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        _, mixed_f1 = case.ad_eval.select_best_f1_threshold(overlapping, mixed)
        self.assertAlmostEqual(mixed_f1, _manual_best_f1(overlapping, mixed), places=12)

        metrics = case.ad_eval.evaluate_anomaly_scores(
            case.ad_config.ScoreBundle(
                val_scores=perfect,
                val_labels=labels,
                test_scores=perfect,
                test_labels=labels,
            ),
            "gbrbm",
            "free_energy",
        )
        self.assertEqual(metrics.roc_auc, 1.0)
        self.assertEqual(metrics.pr_auc, 1.0)
        self.assertEqual(metrics.f1, 1.0)
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.threshold, threshold)

        shifted = case.ad_eval.evaluate_anomaly_scores(
            case.ad_config.ScoreBundle(
                val_scores=perfect,
                val_labels=labels,
                test_scores=perfect * 100.0 + 5.0,
                test_labels=labels,
            ),
            "gbrbm",
            "free_energy",
        )
        self.assertEqual(shifted.threshold, threshold)
        self.assertEqual(shifted.roc_auc, 1.0)

    def test_isolation_forest_baseline_shares_split_and_beats_random(self):
        """The baseline uses the shared splits and clearly beats a random scorer."""
        case = _case()
        split = case.ad_data.prepare_anomaly_dataset(case.ad_config.DataConfig())
        metrics, val_scores, test_scores = case.ad_eval.evaluate_isolation_forest_baseline(
            split, 42
        )
        self.assertEqual(val_scores.shape, (split.x_val.shape[0],))
        self.assertEqual(test_scores.shape, (split.x_test.shape[0],))
        self.assertEqual(metrics.model_name, "isolation_forest")
        self.assertGreater(metrics.roc_auc, 0.85)
        self.assertGreater(metrics.pr_auc, 0.85)
        self.assertGreater(metrics.f1, 0.70)
        self.assertGreater(metrics.roc_auc, 0.5)
        self.assertTrue(np.all(np.isfinite(val_scores)))
        self.assertTrue(np.all(np.isfinite(test_scores)))


class SamplerAndPipelineTests(unittest.TestCase):
    """Sampler construction, Kaiwu SDK compatibility and the CLI smoke run."""

    def test_sampler_builder_sa_default_and_cim_sdk_compat(self):
        """The SA default is real and every CIM alias maps to the SDK task mode."""
        case = _case()
        record: dict[str, Any] = {}
        real_class = kw.classical.SimulatedAnnealingOptimizer

        def factory(**kwargs):
            record.update(kwargs)
            return real_class(**kwargs)

        with mock.patch.object(kw.classical, "SimulatedAnnealingOptimizer", factory):
            sampler = case.ad_sampler.build_sampler(case.ad_config.SamplerConfig())
        self.assertIsInstance(sampler, real_class)
        self.assertEqual(record["size_limit"], 32)
        self.assertAlmostEqual(record["alpha"], 0.95, places=12)
        self.assertEqual(record["rand_seed"], 42)
        self.assertAlmostEqual(record["cutoff_temperature"], 1.0, places=12)

        for alias, expected in (
            ("sampling", "SAMPLING"),
            ("sample", "SAMPLING"),
            ("optimization", "OPTIMIZATION"),
            ("quota", "OPTIMIZATION"),
        ):
            for use_reducer in (True, False):
                with self.subTest(alias=alias, precision_reducer=use_reducer):
                    self._assert_cim_options(case, alias, expected, use_reducer)

    def _assert_cim_options(self, case, alias: str, expected: str, use_reducer: bool) -> None:
        """Assert one CIM alias and reducer combination without any submission."""
        created: list[_RecordingOptimizer] = []
        real_reducer = case.ad_sampler.PrecisionReducer

        def factory(**kwargs):
            optimizer = _RecordingOptimizer(**kwargs)
            created.append(optimizer)
            return optimizer

        config = case.ad_config.SamplerConfig(
            sampler_type="cim",
            cim_task_mode=alias,
            cim_use_precision_reducer=use_reducer,
        )
        with mock.patch.object(kw.cim, "CIMOptimizer", factory):
            sampler = case.ad_sampler.build_sampler(config)
        task_modes = getattr(kw.cim, "TaskMode", None)
        expected_mode = (
            getattr(task_modes, expected)
            if task_modes is not None
            else {"SAMPLING": "sample", "OPTIMIZATION": "quota"}[expected]
        )
        self.assertEqual(created[0].kwargs["task_mode"], expected_mode)
        self.assertEqual(created[0].kwargs["task_name"], "gbrbm_anomaly_detection")
        self.assertEqual(created[0].kwargs["sample_number"], 32)
        self.assertIsNone(created[0].kwargs["project_no"])

        if not use_reducer:
            self.assertIsInstance(sampler, _RecordingOptimizer)
            return
        self.assertIsInstance(sampler, real_reducer)

        with mock.patch.object(case.ad_sampler, "PrecisionReducer") as reducer_mock:
            with mock.patch.object(kw.cim, "CIMOptimizer", factory):
                case.ad_sampler.build_sampler(config)
        positional, keywords = reducer_mock.call_args
        self.assertEqual(len(positional), 1)
        self.assertIsInstance(positional[0], _RecordingOptimizer)
        self.assertEqual(keywords["precision"], 8)
        self.assertEqual(keywords["truncated_precision"], 10)
        self.assertEqual(keywords["target_bits"], 550)

    def test_cli_quick_end_to_end_smoke_with_sa(self):
        """A two-epoch quick run trains, scores, reports and updates parameters."""
        case = _case()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "quick"
            with _sa_offline_context():
                result = case.run_anomaly_detection.main(
                    [
                        "--quick",
                        "--epochs",
                        "2",
                        "--seed",
                        "42",
                        "--device",
                        "cpu",
                        "--output-dir",
                        str(output_dir),
                        "--no-plot",
                    ]
                )
            report = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(len(result["history"]), 2)
        for entry in result["history"]:
            self.assertEqual(sorted(entry), ["epoch", "mean_objective", "train_free_energy"])
            for value in entry.values():
                self.assertIsInstance(value, float)
                self.assertTrue(math.isfinite(value))

        self.assertEqual(report["primary_score_method"], "free_energy")
        self.assertEqual(report["score_convention"], "higher_is_more_anomalous")
        expected_labels = {"gbrbm_free_energy", "gbrbm_reconstruction", "isolation_forest"}
        self.assertEqual(set(report["metrics"]), expected_labels)
        for label in expected_labels:
            item = report["metrics"][label]
            self.assertGreater(item["roc_auc"], 0.80)
            self.assertGreater(item["pr_auc"], 0.80)
            self.assertGreaterEqual(item["f1"], 0.0)
            self.assertLessEqual(item["f1"], 1.0)
            self.assertTrue(0.0 <= item["precision"] <= 1.0)
            self.assertTrue(0.0 <= item["recall"] <= 1.0)
            self.assertTrue(math.isfinite(item["threshold"]))
        self.assertEqual(report["split"]["train_normal"], result["split"].x_train.shape[0])

        split = result["split"]
        case.ad_trainer.set_global_seed(42)
        reference = GaussianBernoulliRestrictedBoltzmannMachine(
            num_visible=split.x_train.shape[1],
            num_hidden=8,
            is_visible_gaussian=True,
            eps=1e-8,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        trained = result["model"]
        self.assertFalse(torch.allclose(trained.quadratic_coef, reference.quadratic_coef))
        self.assertFalse(torch.allclose(trained.mu, reference.mu))
        self.assertFalse(trained.log_var.requires_grad)
        self.assertTrue(torch.allclose(trained.log_var, torch.zeros_like(trained.log_var)))

    def test_non_blocking_agg_plot_export(self):
        """Both figures are written as PNG through the Agg canvas without blocking."""
        pytest.importorskip("matplotlib")
        case = _case()
        plot = case.ad_plot
        self.assertIsNotNone(plot)
        with tempfile.TemporaryDirectory() as temporary:
            payload = plot.PlotPayload(
                normal_scores=np.array([0.10, 0.20, 0.30, 0.40]),
                anomaly_scores=np.array([5.0, 6.5, 8.0]),
                threshold=1.0,
                roc_curves=(
                    plot.RocCurve(
                        name="GBRBM (Free Energy)",
                        false_positive_rate=np.array([0.0, 0.1, 0.4, 1.0]),
                        true_positive_rate=np.array([0.0, 0.8, 0.95, 1.0]),
                        roc_auc=0.9667,
                    ),
                ),
                output_dir=Path(temporary),
            )
            histogram, roc_path = plot.save_anomaly_plots(payload)
            for path in (histogram, roc_path):
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(histogram.name, "score_distribution.png")
            self.assertEqual(roc_path.name, "roc_curve_comparison.png")


if __name__ == "__main__":
    unittest.main()
