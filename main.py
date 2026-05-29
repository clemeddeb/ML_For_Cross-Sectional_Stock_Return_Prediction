#!/usr/bin/env python3
"""Regenerate final report outputs from saved artifacts.

This entry point does not train models. It assumes the submitted repository
contains the saved prediction files and selected model checkpoints needed for
final evaluation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PREDICTION_ARTIFACTS = [
    "outputs/predictions/baseline_predictions_with_boosting.parquet",
    "outputs/predictions/dl_xgb_predictions.parquet",
    "outputs/predictions/ft_base_seed362559_predictions.parquet",
    "outputs/predictions/ft_lr2e5_seed362559_predictions.parquet",
    "outputs/predictions/ft_lr5e5_seed362559_predictions.parquet",
    "outputs/predictions/ft_reg_seed362559_predictions.parquet",
    "outputs/predictions/ft_small_lr5e5_seed362559_predictions.parquet",
    "outputs/predictions/ft_small_seed362559_predictions.parquet",
    "outputs/predictions/ft_transformer_predictions.parquet",
    "outputs/predictions/temporal_tabular_backtest_predictions.parquet",
    "outputs/predictions/temporal_tabular_small_ft_warmstart_predictions.parquet",
    "outputs/predictions/temporal_tabular_small_mlp_branch_predictions.parquet",
    "outputs/predictions/temporal_tabular_small_mlp_warmstart_predictions.parquet",
    "outputs/predictions/temporal_tabular_small_random_predictions.parquet",
    "outputs/predictions/temporal_tabular_tiny_random_predictions.parquet",
    "outputs/predictions/temporal_tabular_transformer_seed362559_predictions.parquet",
]

MODEL_ARTIFACTS = [
    "outputs/models/baselines/elastic_net_model.joblib",
    "outputs/models/baselines/gradient_boosting_classifier.joblib",
    "outputs/models/baselines/gradient_boosting_regressor.joblib",
    "outputs/models/baselines/logistic_classifier_model.joblib",
    "outputs/models/baselines/ridge_model.joblib",
    "outputs/models/deep_learning/mlp_model.pt",
    "outputs/models/deep_learning/regression/mlp_regressor.pt",
    "outputs/models/deep_learning/ft_small_seed362559.pt",
    "outputs/models/deep_learning/ft_transformer.pt",
    "outputs/models/deep_learning/temporal_tabular_small_ft_warmstart.pt",
    "outputs/models/deep_learning/temporal_tabular_small_random.pt",
]

GENERATED_FIGURE_PATTERNS = [
    "outputs/backtests/figures/drawdown_overlay_test_*.png",
    "outputs/backtests/figures/drawdown_overlay_test_metric_bars_25bps.png",
]


def ensure_repo_root() -> Path:
    root = Path(__file__).resolve().parent
    if not (root / "scripts").exists():
        raise RuntimeError(f"Could not find scripts/ from {root}")
    return root


def check_artifacts(paths: list[str]) -> None:
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required final artifacts:\n{formatted}")


def remove_stale_generated_figures(patterns: list[str]) -> None:
    for pattern in patterns:
        for path in Path().glob(pattern):
            path.unlink()


def run_step(label: str, command: list[str]) -> None:
    print(f"\n==> {label}")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    root = ensure_repo_root()
    if Path.cwd().resolve() != root:
        raise RuntimeError(f"Run this script from the repository root: {root}")

    check_artifacts(PREDICTION_ARTIFACTS)
    check_artifacts(MODEL_ARTIFACTS)
    remove_stale_generated_figures(GENERATED_FIGURE_PATTERNS)

    python = sys.executable
    run_step("Forecasting summary tables", [python, "scripts/forecasting_analysis.py"])
    run_step(
        "Raw portfolio analysis at 25 bps",
        [python, "scripts/backtesting/run_portfolio_analysis.py", "--cost-bps", "25"],
    )
    run_step(
        "Drawdown overlay analysis at 25 bps",
        [
            python,
            "scripts/backtesting/run_drawdown_overlay_analysis.py",
            "--cost-bps",
            "25",
            "--top-n",
            "300",
        ],
    )

    print("\nFinal outputs regenerated under:")
    print("  - outputs/tables/")
    print("  - outputs/backtests/")
    print("  - outputs/backtests/figures/")


if __name__ == "__main__":
    main()
