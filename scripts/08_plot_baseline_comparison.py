#!/usr/bin/env python3
"""Refresh baseline comparison figures, including XGBoost baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_SEED = 362559

DEFAULT_BASELINE_MONTHLY_IC = Path("outputs/tables/baseline_monthly_rank_ic.csv")
DEFAULT_BOOSTING_MONTHLY_IC = Path("outputs/tables/boosting_monthly_rank_ic.csv")
DEFAULT_BASELINE_CONFUSION = Path("outputs/tables/baseline_classifier_confusion_matrices.csv")
DEFAULT_BOOSTING_CONFUSION = Path("outputs/tables/boosting_classifier_confusion_matrices.csv")
DEFAULT_PREDICTIONS = Path("outputs/predictions/baseline_predictions_with_boosting.parquet")
DEFAULT_TABLE_DIR = Path("outputs/tables")
DEFAULT_FIGURE_DIR = Path("outputs/figures")

MODEL_LABELS = {
    "naive_momentum": "Naive momentum",
    "naive_reversal": "Naive reversal",
    "ridge": "Ridge",
    "elastic_net": "Elastic Net",
    "logistic_classifier": "Logistic classifier",
    "gradient_boosting_reg": "XGBoost regressor",
    "gb_classifier": "XGBoost classifier",
}

MODEL_ORDER = [
    "naive_momentum",
    "naive_reversal",
    "ridge",
    "elastic_net",
    "logistic_classifier",
    "gradient_boosting_reg",
    "gb_classifier",
]

PREDICTION_COLUMNS = {
    "naive_momentum": "prediction_naive_momentum",
    "naive_reversal": "prediction_naive_reversal",
    "ridge": "prediction_ridge",
    "elastic_net": "prediction_elastic_net",
    "logistic_classifier": "prediction_logistic_classifier_score",
    "gradient_boosting_reg": "prediction_gradient_boosting_reg",
    "gb_classifier": "prediction_gb_classifier_score",
}

CLASS_LABELS = {
    0: "Bottom",
    1: "Middle",
    2: "Top",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh baseline comparison figures.")
    parser.add_argument("--baseline-monthly-ic", type=Path, default=DEFAULT_BASELINE_MONTHLY_IC)
    parser.add_argument("--boosting-monthly-ic", type=Path, default=DEFAULT_BOOSTING_MONTHLY_IC)
    parser.add_argument("--baseline-confusion", type=Path, default=DEFAULT_BASELINE_CONFUSION)
    parser.add_argument("--boosting-confusion", type=Path, default=DEFAULT_BOOSTING_CONFUSION)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--distribution-sample-rows", type=int, default=200_000)
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def read_monthly_rank_ic(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {"model", "split", "month", "rank_ic"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out = out.loc[out["model"].isin(MODEL_ORDER)].copy()
    out["month"] = pd.to_datetime(out["month"], errors="raise")
    out["rank_ic"] = pd.to_numeric(out["rank_ic"], errors="coerce")
    out = out.dropna(subset=["rank_ic"])
    out = out.drop_duplicates(["model", "split", "month"], keep="last")
    out["model_label"] = out["model"].map(MODEL_LABELS).fillna(out["model"])
    return out


def rank_ic_summary(monthly_ic: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        monthly_ic.groupby(["model", "split"], observed=True)["rank_ic"]
        .agg(mean_monthly_rank_ic="mean", std_monthly_rank_ic="std", months="count")
        .reset_index()
    )
    grouped["t_stat_monthly_rank_ic"] = (
        grouped["mean_monthly_rank_ic"]
        / grouped["std_monthly_rank_ic"]
        * np.sqrt(grouped["months"])
    )
    grouped["model_label"] = grouped["model"].map(MODEL_LABELS).fillna(grouped["model"])
    grouped["model_order"] = grouped["model"].map({model: i for i, model in enumerate(MODEL_ORDER)})
    return grouped.sort_values(["split", "model_order"]).drop(columns="model_order")


def plot_rank_ic_bar(summary: pd.DataFrame, split: str, path: Path) -> None:
    data = summary.loc[summary["split"].eq(split)].copy()
    data = data.sort_values("mean_monthly_rank_ic")
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = ["#247ba0" if model.startswith("XGBoost") else "#6f7f8f" for model in data["model_label"]]
    ax.barh(data["model_label"], data["mean_monthly_rank_ic"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{split.title()} Monthly Rank IC by Model")
    ax.set_xlabel("Mean monthly rank IC")
    ax.set_ylabel("")
    for index, value in enumerate(data["mean_monthly_rank_ic"]):
        offset = 0.001 if value >= 0 else -0.001
        ha = "left" if value >= 0 else "right"
        ax.text(value + offset, index, f"{value:.4f}", va="center", ha=ha, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_monthly_rank_ic(monthly_ic: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for model in MODEL_ORDER:
        part = monthly_ic.loc[monthly_ic["model"].eq(model)].sort_values("month")
        if part.empty:
            continue
        smoothed = part.set_index("month")["rank_ic"].rolling(12, min_periods=3).mean()
        linewidth = 1.8 if model in {"gb_classifier", "gradient_boosting_reg"} else 1.2
        ax.plot(smoothed.index, smoothed.values, linewidth=linewidth, label=MODEL_LABELS[model])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Monthly Rank IC Over Time (12-month rolling mean)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Rank IC")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def read_prediction_sample(path: Path, max_rows: int) -> pd.DataFrame:
    columns = [column for column in PREDICTION_COLUMNS.values()]
    if not path.exists():
        raise FileNotFoundError(path)
    predictions = pd.read_parquet(path, columns=columns)
    if len(predictions) > max_rows:
        predictions = predictions.sample(max_rows, random_state=RANDOM_SEED)
    return predictions


def plot_prediction_distribution(predictions: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for model in MODEL_ORDER:
        column = PREDICTION_COLUMNS[model]
        if column not in predictions.columns:
            continue
        values = pd.to_numeric(predictions[column], errors="coerce").dropna()
        if values.empty:
            continue
        lo, hi = values.quantile([0.01, 0.99])
        ax.hist(
            values.clip(lo, hi),
            bins=80,
            histtype="step",
            density=True,
            linewidth=1.35,
            label=MODEL_LABELS[model],
        )
    ax.set_title("Prediction Distribution by Model")
    ax.set_xlabel("Prediction or ranking score, clipped at model 1st/99th percentiles")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def read_confusion(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {"model", "split", "actual_label", "predicted_label", "count"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out = out.loc[out["model"].isin(["logistic_classifier", "gb_classifier"])].copy()
    out = out.drop_duplicates(["model", "split", "actual_label", "predicted_label"], keep="last")
    out["model_label"] = out["model"].map(MODEL_LABELS)
    return out


def plot_confusion(confusion: pd.DataFrame, split: str, path: Path) -> None:
    models = ["logistic_classifier", "gb_classifier"]
    fig, axes = plt.subplots(1, len(models), figsize=(10.5, 4.5), squeeze=False)
    max_count = 0
    matrices: dict[str, pd.DataFrame] = {}
    for model in models:
        part = confusion.loc[confusion["split"].eq(split) & confusion["model"].eq(model)]
        matrix = (
            part.pivot(index="actual_label", columns="predicted_label", values="count")
            .reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
            .fillna(0)
        )
        matrices[model] = matrix
        max_count = max(max_count, int(matrix.to_numpy().max()))

    for ax, model in zip(axes[0], models):
        matrix = matrices[model]
        image = ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=max_count)
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_xticks([0, 1, 2], [CLASS_LABELS[i] for i in [0, 1, 2]])
        ax.set_yticks([0, 1, 2], [CLASS_LABELS[i] for i in [0, 1, 2]])
        row_totals = matrix.sum(axis=1).replace(0, np.nan)
        for i in range(3):
            for j in range(3):
                count = int(matrix.iloc[i, j])
                share = count / row_totals.iloc[i]
                text_color = "white" if count > max_count * 0.45 else "black"
                ax.text(
                    j,
                    i,
                    f"{count:,}\n{share:.1%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{split.title()} Classifier Confusion Matrices")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    log("Reading monthly rank IC tables...")
    monthly_ic = read_monthly_rank_ic([args.baseline_monthly_ic, args.boosting_monthly_ic])
    summary = rank_ic_summary(monthly_ic)
    summary_path = args.table_dir / "baseline_all_model_rank_ic_summary.csv"
    summary.to_csv(summary_path, index=False)
    log(f"Saved {summary_path}")

    log("Writing rank IC figures...")
    plot_rank_ic_bar(summary, "validation", args.figure_dir / "validation_rank_ic_by_model.png")
    plot_rank_ic_bar(summary, "test", args.figure_dir / "test_rank_ic_by_model.png")
    plot_monthly_rank_ic(monthly_ic, args.figure_dir / "monthly_rank_ic_over_time.png")

    log("Reading merged prediction sample for distribution plot...")
    prediction_sample = read_prediction_sample(args.predictions, args.distribution_sample_rows)
    plot_prediction_distribution(prediction_sample, args.figure_dir / "prediction_distribution_by_model.png")

    log("Writing classifier confusion matrix figures...")
    confusion = read_confusion([args.baseline_confusion, args.boosting_confusion])
    plot_confusion(confusion, "validation", args.figure_dir / "classifier_confusion_matrix_validation.png")
    plot_confusion(confusion, "test", args.figure_dir / "classifier_confusion_matrix_test.png")

    log("Done.")


if __name__ == "__main__":
    main()
