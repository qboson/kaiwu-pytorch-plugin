# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CLI for unsupervised continuous-feature anomaly detection with GBRBM.

The pipeline only touches the public API of
``kaiwu.torch_plugin.gbrbm.GaussianBernoulliRestrictedBoltzmannMachine`` and
never modifies the plugin sources. Plot export is optional: when ``matplotlib``
is missing the run continues and only the JSON report is written.
"""

import argparse
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_curve

from ad_config import (
    AnomalyDatasetSplit,
    DataConfig,
    EvaluationMetrics,
    ModelConfig,
    SamplerConfig,
    ScoreBundle,
    TrainConfig,
)
from ad_data import prepare_anomaly_dataset
from ad_eval import (
    evaluate_anomaly_scores,
    evaluate_isolation_forest_baseline,
    metrics_to_dict,
    save_metrics_report,
)
from ad_sampler import build_sampler, init_kaiwu_license_from_env
from ad_scorer import GBRBMAnomalyScorer
from ad_trainer import train_gbrbm

try:
    from ad_plot import PlotPayload, RocCurve, save_anomaly_plots

    PLOTTING_AVAILABLE = True
except ImportError:
    PlotPayload = None
    RocCurve = None
    save_anomaly_plots = None
    PLOTTING_AVAILABLE = False

CASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = CASE_DIR / "outputs"
GBRBM_SCORE_METHODS = ("free_energy", "reconstruction")
PLOT_SERIES = (
    ("gbrbm_free_energy", "GBRBM (Free Energy)"),
    ("gbrbm_reconstruction", "GBRBM (Reconstruction)"),
    ("isolation_forest", "IsolationForest"),
)
METRIC_ORDER = ("gbrbm_free_energy", "gbrbm_reconstruction", "isolation_forest")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command line parser of the case.

    Returns:
        argparse.ArgumentParser: Parser with all documented options.
    """
    parser = argparse.ArgumentParser(
        description="Unsupervised GBRBM anomaly detection on Breast Cancer Wisconsin."
    )
    parser.add_argument("--epochs", type=int, default=30, help="GBRBM training epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics.json and the two PNG figures.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Compute device; 'cuda' fails fast when no GPU is available.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke mode: 2 epochs, 8 hidden units and a lighter SA schedule.",
    )
    parser.add_argument(
        "--sampler",
        choices=("sa", "cim"),
        default="sa",
        help="Ising sampler: local simulated annealing (default) or CIM hardware.",
    )
    parser.add_argument(
        "--score-method",
        choices=GBRBM_SCORE_METHODS,
        default="free_energy",
        help="Primary score method used for the distribution figure.",
    )
    parser.add_argument("--num-hidden", type=int, default=16, help="Bernoulli hidden units.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=5e-3, help="SGD learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="L2 penalty factor.")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip figure export; metrics.json is still written.",
    )
    parser.add_argument(
        "--cim-task-name",
        default="gbrbm_anomaly_detection",
        help="CIM task name, used only with '--sampler cim'.",
    )
    parser.add_argument(
        "--cim-task-mode",
        default="sampling",
        help="CIM mode alias: sampling/sample/optimization/quota.",
    )
    parser.add_argument(
        "--cim-no-precision-reducer",
        action="store_true",
        help="Disable the CIM PrecisionReducer wrapper.",
    )
    return parser


def _build_configs(
    args: argparse.Namespace,
) -> tuple[DataConfig, ModelConfig, SamplerConfig, TrainConfig]:
    """Translate parsed arguments into the frozen configuration objects."""
    data_config = DataConfig(seed=args.seed)
    model_config = ModelConfig(num_hidden=args.num_hidden)
    sampler_config = SamplerConfig(
        sampler_type=args.sampler,
        seed=args.seed,
        cim_task_name=args.cim_task_name,
        cim_task_mode=args.cim_task_mode,
        cim_use_precision_reducer=not args.cim_no_precision_reducer,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )
    if args.quick:
        model_config = replace(model_config, num_hidden=8)
        train_config = replace(train_config, epochs=2)
        sampler_config = replace(
            sampler_config,
            sa_size_limit=16,
            sa_iterations_per_t=3,
            sa_alpha=0.9,
        )
    return data_config, model_config, sampler_config, train_config


def _bundle(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
) -> ScoreBundle:
    """Assemble a score bundle for the shared evaluation pipeline."""
    return ScoreBundle(
        val_scores=val_scores,
        val_labels=val_labels,
        test_scores=test_scores,
        test_labels=test_labels,
    )


def _evaluate_models(
    scorer: GBRBMAnomalyScorer,
    split: AnomalyDatasetSplit,
    seed: int,
) -> tuple[dict[str, EvaluationMetrics], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Evaluate both GBRBM scoring methods and the IsolationForest baseline."""
    metrics: dict[str, EvaluationMetrics] = {}
    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in GBRBM_SCORE_METHODS:
        label = f"gbrbm_{method}"
        val_scores = scorer.score_numpy(split.x_val, method)
        test_scores = scorer.score_numpy(split.x_test, method)
        scores[label] = (val_scores, test_scores)
        bundle = _bundle(val_scores, split.y_val, test_scores, split.y_test)
        metrics[label] = evaluate_anomaly_scores(bundle, "gbrbm", method)
    baseline, baseline_val, baseline_test = evaluate_isolation_forest_baseline(split, seed)
    metrics["isolation_forest"] = baseline
    scores["isolation_forest"] = (baseline_val, baseline_test)
    return metrics, scores


def _roc_curve(
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    roc_auc: float,
) -> "RocCurve":
    """Build one ROC curve container for the comparison figure."""
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    return RocCurve(
        name=name,
        false_positive_rate=false_positive_rate,
        true_positive_rate=true_positive_rate,
        roc_auc=roc_auc,
    )


def _render_plots(
    split: AnomalyDatasetSplit,
    metrics: dict[str, EvaluationMetrics],
    scores: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    primary_method: str,
) -> dict[str, str]:
    """Export the score histogram and the ROC comparison when possible."""
    if not PLOTTING_AVAILABLE:
        print("[info] matplotlib is not installed; skipping figure export.")
        return {}
    label = f"gbrbm_{primary_method}"
    test_scores = scores[label][1]
    payload = PlotPayload(
        normal_scores=test_scores[split.y_test == 0],
        anomaly_scores=test_scores[split.y_test == 1],
        threshold=metrics[label].threshold,
        roc_curves=tuple(
            _roc_curve(display, scores[key][1], split.y_test, metrics[key].roc_auc)
            for key, display in PLOT_SERIES
        ),
        output_dir=Path(output_dir),
    )
    histogram_path, roc_path = save_anomaly_plots(payload)
    return {
        "score_distribution": str(histogram_path),
        "roc_curve_comparison": str(roc_path),
    }


def _print_report(
    split: AnomalyDatasetSplit,
    metrics: dict[str, EvaluationMetrics],
    history: list[dict[str, float]],
    artifacts: dict[str, str],
    report_path: Path,
) -> None:
    """Print the split summary, the training summary and the metric table."""
    print("=== Unsupervised GBRBM anomaly detection (Breast Cancer Wisconsin) ===")
    print(f"features            : {split.x_train.shape[1]}")
    print(f"train (normal only) : {split.x_train.shape[0]}")
    print(f"val                 : {split.x_val.shape[0]} ({int(split.y_val.sum())} anomalies)")
    print(f"test                : {split.x_test.shape[0]} ({int(split.y_test.sum())} anomalies)")
    if history:
        first = history[0]
        last = history[-1]
        print(
            f"epochs: {len(history)}  mean objective "
            f"{first['mean_objective']:.4f} -> {last['mean_objective']:.4f}"
        )
        print(
            f"train free energy: {first['train_free_energy']:.4f} -> "
            f"{last['train_free_energy']:.4f}"
        )
    print("-" * 80)
    print(f"{'model':<26}{'ROC-AUC':>9}{'PR-AUC':>9}{'F1':>8}{'Prec':>8}{'Rec':>8}{'Thr':>12}")
    for label in METRIC_ORDER:
        item = metrics[label]
        row = (
            f"{label:<26}{item.roc_auc:>9.4f}{item.pr_auc:>9.4f}"
            f"{item.f1:>8.4f}{item.precision:>8.4f}{item.recall:>8.4f}"
            f"{item.threshold:>12.4f}"
        )
        print(row)
    print("-" * 80)
    print(f"metrics.json: {report_path}")
    for name, path in artifacts.items():
        print(f"figure ({name}): {path}")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full pipeline for already parsed arguments.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        dict: In-memory results containing the metrics, the raw scores, the
        training history, the trained model, the splits, the figure paths and
        the ``metrics.json`` path.
    """
    data_config, model_config, sampler_config, train_config = _build_configs(args)
    if not init_kaiwu_license_from_env():
        print("[info] USER_ID/SDK_CODE are not set; keeping the current Kaiwu license state.")
    split = prepare_anomaly_dataset(data_config)
    sampler = build_sampler(sampler_config)
    model, history = train_gbrbm(split.x_train, model_config, train_config, sampler)
    scorer = GBRBMAnomalyScorer(model)
    metrics, scores = _evaluate_models(scorer, split, args.seed)

    output_dir = Path(args.output_dir)
    artifacts: dict[str, str] = {}
    if not args.no_plot:
        artifacts = _render_plots(split, metrics, scores, output_dir, args.score_method)

    payload = {
        "case": "gbrbm_anomaly_detection",
        "dataset": "sklearn.datasets.load_breast_cancer",
        "score_convention": "higher_is_more_anomalous",
        "primary_score_method": args.score_method,
        "config": {
            "data": asdict(data_config),
            "model": asdict(model_config),
            "sampler": asdict(sampler_config),
            "train": asdict(train_config),
        },
        "split": {
            "train_normal": int(split.x_train.shape[0]),
            "val": int(split.x_val.shape[0]),
            "test": int(split.x_test.shape[0]),
            "num_features": int(split.x_train.shape[1]),
            "val_anomalies": int(split.y_val.sum()),
            "test_anomalies": int(split.y_test.sum()),
        },
        "training_history": history,
        "metrics": {label: metrics_to_dict(item) for label, item in metrics.items()},
        "artifacts": artifacts,
    }
    report_path = save_metrics_report(output_dir, payload)
    _print_report(split, metrics, history, artifacts, report_path)

    return {
        "metrics": metrics,
        "scores": scores,
        "history": history,
        "model": model,
        "split": split,
        "artifacts": artifacts,
        "report_path": report_path,
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Parse ``argv`` and run the pipeline.

    Args:
        argv (Sequence[str] | None, optional): Command line arguments without the
            program name. ``None`` uses ``sys.argv[1:]``.

    Returns:
        dict: The in-memory results returned by :func:`run_pipeline`.
    """
    args = build_arg_parser().parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    main()
