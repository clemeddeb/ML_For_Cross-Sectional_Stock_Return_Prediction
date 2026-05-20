#!/usr/bin/env python3
"""Run FT-Transformer robustness experiments and consolidate outputs."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


TRAIN_SCRIPT = Path("scripts/deep_learning/train_ft_transformer.py")
TABLE_DIR = Path("outputs/tables")
FIGURE_DIR = Path("outputs/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small FT-Transformer experiment grid.")
    parser.add_argument("--debug", action="store_true", help="Pass --debug to each training run.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--only", type=str, default=None, help="Run only one experiment by run_name.")
    parser.add_argument("--max-experiments", type=int, default=None, help="Optionally limit the run count.")
    return parser.parse_args()


def experiment_grid() -> list[dict[str, Any]]:
    baseline = {
        "seed": 362559,
        "selection_metric": "classifier_rank_ic",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "attention_dropout": 0.1,
        "ffn_dropout": 0.1,
        "residual_dropout": 0.0,
        "d_token": 64,
        "n_layers": 3,
        "n_heads": 4,
        "batch_size": 4096,
        "max_epochs": 50,
        "patience": 6,
    }
    specs = [
        {"run_name": "ft_base_seed362559"},
        {"run_name": "ft_lr5e5_seed362559", "learning_rate": 5e-5},
        {"run_name": "ft_lr2e5_seed362559", "learning_rate": 2e-5},
        {
            "run_name": "ft_reg_seed362559",
            "weight_decay": 1e-4,
            "attention_dropout": 0.2,
            "ffn_dropout": 0.2,
        },
        {"run_name": "ft_small_seed362559", "d_token": 32, "n_layers": 2, "n_heads": 4},
        {
            "run_name": "ft_small_lr5e5_seed362559",
            "d_token": 32,
            "n_layers": 2,
            "n_heads": 4,
            "learning_rate": 5e-5,
        },
        {"run_name": "ft_base_seed42", "seed": 42},
        {"run_name": "ft_base_seed123", "seed": 123},
    ]
    out = []
    for spec in specs:
        merged = baseline.copy()
        merged.update(spec)
        out.append(merged)
    return out


def filter_experiments(experiments: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = experiments
    if args.only:
        selected = [experiment for experiment in selected if experiment["run_name"] == args.only]
        if not selected:
            raise ValueError(f"Unknown run_name for --only: {args.only}")
    if args.max_experiments is not None:
        selected = selected[: args.max_experiments]
    return selected


def build_command(experiment: dict[str, Any], debug: bool) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--run-name",
        str(experiment["run_name"]),
        "--seed",
        str(experiment["seed"]),
        "--selection-metric",
        str(experiment["selection_metric"]),
        "--learning-rate",
        str(experiment["learning_rate"]),
        "--weight-decay",
        str(experiment["weight_decay"]),
        "--attention-dropout",
        str(experiment["attention_dropout"]),
        "--ffn-dropout",
        str(experiment["ffn_dropout"]),
        "--residual-dropout",
        str(experiment["residual_dropout"]),
        "--d-token",
        str(experiment["d_token"]),
        "--n-layers",
        str(experiment["n_layers"]),
        "--n-heads",
        str(experiment["n_heads"]),
        "--batch-size",
        str(experiment["batch_size"]),
        "--max-epochs",
        str(experiment["max_epochs"]),
        "--patience",
        str(experiment["patience"]),
    ]
    if debug:
        command.append("--debug")
    return command


def metric_summary(monthly_ic: pd.DataFrame, split: str, score_name: str) -> tuple[float, float]:
    values = monthly_ic.loc[
        monthly_ic["split"].eq(split) & monthly_ic["score_name"].eq(score_name),
        "rank_ic",
    ].dropna()
    if values.empty:
        return (math.nan, math.nan)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else math.nan
    tstat = mean / (std / math.sqrt(len(values))) if len(values) > 1 and std and math.isfinite(std) else math.nan
    return (mean, tstat)


def summarize_run(run_name: str) -> dict[str, Any]:
    selected_path = TABLE_DIR / f"{run_name}_selected_hyperparameters.csv"
    metrics_path = TABLE_DIR / f"{run_name}_model_metrics.csv"
    monthly_ic_path = TABLE_DIR / f"{run_name}_monthly_rank_ic.csv"
    epoch_history_path = TABLE_DIR / f"{run_name}_epoch_history.csv"

    selected = pd.read_csv(selected_path).iloc[0]
    pd.read_csv(metrics_path)
    monthly_ic = pd.read_csv(monthly_ic_path)
    epoch_history = pd.read_csv(epoch_history_path)

    best_epoch_row = epoch_history.loc[epoch_history["epoch"].eq(int(selected["best_epoch"]))].iloc[0]
    val_classifier_mean, val_classifier_tstat = metric_summary(
        monthly_ic, "validation", "prediction_ft_classifier_score"
    )
    val_er_mean, val_er_tstat = metric_summary(monthly_ic, "validation", "prediction_ft_er_train_score")
    test_classifier_mean, test_classifier_tstat = metric_summary(monthly_ic, "test", "prediction_ft_classifier_score")
    test_er_mean, test_er_tstat = metric_summary(monthly_ic, "test", "prediction_ft_er_train_score")
    early_stopped = str(selected["early_stopped"]).strip().lower() == "true"

    return {
        "run_name": selected["run_name"],
        "seed": int(selected["seed"]),
        "selection_metric": selected["selection_metric"],
        "best_epoch": int(selected["best_epoch"]),
        "stopped_epoch": int(selected["stopped_epoch"]),
        "early_stopped": early_stopped,
        "learning_rate": float(selected["learning_rate"]),
        "weight_decay": float(selected["weight_decay"]),
        "attention_dropout": float(selected["attention_dropout"]),
        "ffn_dropout": float(selected["ffn_dropout"]),
        "residual_dropout": float(selected["residual_dropout"]),
        "d_token": int(selected["d_token"]),
        "n_layers": int(selected["n_layers"]),
        "n_heads": int(selected["n_heads"]),
        "batch_size": int(selected["batch_size"]),
        "selected_feature_count": int(selected["selected_feature_count"]),
        "validation_classifier_rank_ic_mean": val_classifier_mean,
        "validation_classifier_rank_ic_tstat": val_classifier_tstat,
        "validation_er_train_rank_ic_mean": val_er_mean,
        "validation_er_train_rank_ic_tstat": val_er_tstat,
        "test_classifier_rank_ic_mean": test_classifier_mean,
        "test_classifier_rank_ic_tstat": test_classifier_tstat,
        "test_er_train_rank_ic_mean": test_er_mean,
        "test_er_train_rank_ic_tstat": test_er_tstat,
        "validation_loss_at_best_epoch": float(best_epoch_row["validation_loss"]),
        "train_loss_at_best_epoch": float(best_epoch_row["train_loss"]),
    }


def has_completed_outputs(run_name: str) -> bool:
    required = [
        TABLE_DIR / f"{run_name}_selected_hyperparameters.csv",
        TABLE_DIR / f"{run_name}_model_metrics.csv",
        TABLE_DIR / f"{run_name}_monthly_rank_ic.csv",
        TABLE_DIR / f"{run_name}_epoch_history.csv",
    ]
    return all(path.exists() for path in required)


def build_report(summary: pd.DataFrame) -> str:
    def run_value(run_name: str) -> float | None:
        row = summary.loc[summary["run_name"].eq(run_name), "validation_classifier_rank_ic_mean"]
        return None if row.empty else float(row.iloc[0])

    lines = ["# FT-Transformer Experiment Report", ""]
    if summary.empty:
        lines.append("No completed runs were available.")
        return "\n".join(lines) + "\n"

    best_row = summary.sort_values("validation_classifier_rank_ic_mean", ascending=False).iloc[0]
    lines.append(
        f"Best run by validation classifier Rank IC: `{best_row['run_name']}` "
        f"({best_row['validation_classifier_rank_ic_mean']:.6f})."
    )
    lines.append("")
    lines.append("## Epoch 1 Check")
    for _, row in summary.sort_values("run_name").iterrows():
        lines.append(
            f"- `{row['run_name']}`: {'yes' if int(row['best_epoch']) == 1 else 'no'} "
            f"(best_epoch={int(row['best_epoch'])})."
        )
    lines.append("")
    lines.append("## Findings")
    baseline_value = run_value("ft_base_seed362559")
    lr5_value = run_value("ft_lr5e5_seed362559")
    lr2_value = run_value("ft_lr2e5_seed362559")
    reg_value = run_value("ft_reg_seed362559")
    small_value = run_value("ft_small_seed362559")
    small_lr_value = run_value("ft_small_lr5e5_seed362559")

    if baseline_value is None:
        lines.append("- Lower-learning-rate, regularization, and architecture comparisons are unavailable because the baseline run was not present.")
    else:
        if lr5_value is None and lr2_value is None:
            lines.append("- Lower learning rate comparison is unavailable with the current run set.")
        else:
            comparisons = []
            if lr5_value is not None:
                comparisons.append(f"`ft_lr5e5_seed362559`: delta={lr5_value - baseline_value:.6f}")
            if lr2_value is not None:
                comparisons.append(f"`ft_lr2e5_seed362559`: delta={lr2_value - baseline_value:.6f}")
            improved = any(value is not None and value > baseline_value for value in [lr5_value, lr2_value])
            lines.append(
                f"- Lower learning rates improved validation classifier Rank IC: {'yes' if improved else 'no'}"
                f" ({'; '.join(comparisons)})."
            )
        if reg_value is None:
            lines.append("- Stronger regularization comparison is unavailable with the current run set.")
        else:
            lines.append(
                f"- Stronger regularization improved validation classifier Rank IC: "
                f"{'yes' if reg_value > baseline_value else 'no'} "
                f"(delta={reg_value - baseline_value:.6f})."
            )
        if small_value is None and small_lr_value is None:
            lines.append("- Smaller-architecture comparison is unavailable with the current run set.")
        else:
            candidate_values = [value for value in [small_value, small_lr_value] if value is not None]
            best_small = max(candidate_values)
            lines.append(
                f"- Smaller architecture was competitive: "
                f"{'yes' if best_small >= baseline_value - 0.005 else 'no'} "
                f"(best small-run delta={best_small - baseline_value:.6f})."
            )
    seed_rows = summary.loc[summary["run_name"].isin(["ft_base_seed362559", "ft_base_seed42", "ft_base_seed123"])]
    if len(seed_rows) >= 2:
        spread = float(
            seed_rows["validation_classifier_rank_ic_mean"].max() - seed_rows["validation_classifier_rank_ic_mean"].min()
        )
        lines.append(
            f"- Seed variation materially changes results: {'yes' if spread > 0.01 else 'no'} "
            f"(spread={spread:.6f})."
        )
    else:
        lines.append("- Seed variation comparison is unavailable with the current run set.")
    lines.append("")
    lines.append("Test metrics are reported for final comparison only and are not used for checkpoint selection.")
    return "\n".join(lines) + "\n"


def write_plots(summary: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = summary.sort_values("validation_classifier_rank_ic_mean", ascending=False).reset_index(drop=True)
    x = range(len(plot_df))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - 0.18 for i in x], plot_df["validation_classifier_rank_ic_mean"], width=0.36, label="classifier")
    ax.bar([i + 0.18 for i in x], plot_df["validation_er_train_rank_ic_mean"], width=0.36, label="er_train")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["run_name"], rotation=45, ha="right")
    ax.set_ylabel("Mean monthly Rank IC")
    ax.set_title("Validation Rank IC by Run")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ft_transformer_validation_rank_ic_by_run.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - 0.18 for i in x], plot_df["test_classifier_rank_ic_mean"], width=0.36, label="classifier")
    ax.bar([i + 0.18 for i in x], plot_df["test_er_train_rank_ic_mean"], width=0.36, label="er_train")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["run_name"], rotation=45, ha="right")
    ax.set_ylabel("Mean monthly Rank IC")
    ax.set_title("Test Rank IC by Run")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ft_transformer_test_rank_ic_by_run.png", dpi=160)
    plt.close(fig)

    top_runs = plot_df.head(min(len(plot_df), 3))["run_name"].tolist()
    fig, ax = plt.subplots(figsize=(12, 6))
    for run_name in top_runs:
        epoch_history = pd.read_csv(TABLE_DIR / f"{run_name}_epoch_history.csv")
        ax.plot(
            epoch_history["epoch"],
            epoch_history["validation_classifier_rank_ic"],
            marker="o",
            linewidth=1.2,
            label=run_name,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation classifier Rank IC")
    ax.set_title("Epoch Curves for Best Runs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ft_transformer_epoch_curves_best_runs.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    grid = experiment_grid()
    experiments = filter_experiments(grid, args)

    commands = [build_command(experiment, args.debug) for experiment in experiments]
    for command in commands:
        print(" ".join(command), flush=True)
    if args.dry_run:
        return

    for command in commands:
        subprocess.run(command, check=True)

    summary_runs = [experiment["run_name"] for experiment in grid if has_completed_outputs(experiment["run_name"])]
    summary_rows = [summarize_run(run_name) for run_name in summary_runs]
    summary = pd.DataFrame(summary_rows).sort_values("validation_classifier_rank_ic_mean", ascending=False)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "ft_transformer_experiment_summary.csv", index=False)
    (TABLE_DIR / "ft_transformer_experiment_report.md").write_text(build_report(summary), encoding="utf-8")
    write_plots(summary)


if __name__ == "__main__":
    main()
