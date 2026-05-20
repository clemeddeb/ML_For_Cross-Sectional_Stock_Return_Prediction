#!/usr/bin/env python3
"""Test whether MLP regressor predictions add signal beyond the MLP classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from mlp_common import DEFAULT_OUTPUT_DIR, log  # noqa: E402


DEFAULT_CLASSIFIER_PREDICTIONS = DEFAULT_OUTPUT_DIR / "mlp_validation_predictions.parquet"
DEFAULT_REGRESSOR_PREDICTIONS = DEFAULT_OUTPUT_DIR / "regression" / "mlp_reg_validation_predictions.parquet"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "regression_signal_analysis.csv"

MERGE_KEYS = ["PERMNO", "MthCalDt"]
TARGET_COLUMN = "target_ret_1m"
CLASSIFIER_SCORE_COLUMN = "prediction_mlp_score"
REGRESSOR_RETURN_CANDIDATES = [
    "predicted_return",
    "prediction_mlp_reg_return",
    "score_mlp_reg_er",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-sectional diagnostic for incremental MLP regressor signal."
    )
    parser.add_argument("--classifier-predictions", type=Path, default=DEFAULT_CLASSIFIER_PREDICTIONS)
    parser.add_argument("--regressor-predictions", type=Path, default=DEFAULT_REGRESSOR_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def choose_regressor_column(df: pd.DataFrame) -> str:
    for column in REGRESSOR_RETURN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        "Regressor predictions must contain one of: "
        + ", ".join(REGRESSOR_RETURN_CANDIDATES)
    )


def load_and_align(classifier_path: Path, regressor_path: Path) -> tuple[pd.DataFrame, str]:
    classifier = pd.read_parquet(classifier_path)
    regressor = pd.read_parquet(regressor_path)

    require_columns(classifier, MERGE_KEYS + ["split", TARGET_COLUMN, CLASSIFIER_SCORE_COLUMN], "Classifier file")
    require_columns(regressor, MERGE_KEYS + ["split", TARGET_COLUMN], "Regressor file")
    regressor_column = choose_regressor_column(regressor)

    classifier = classifier.loc[classifier["split"].eq("validation")].copy()
    regressor = regressor.loc[regressor["split"].eq("validation")].copy()
    if classifier.empty:
        raise ValueError("Classifier validation predictions are empty.")
    if regressor.empty:
        raise ValueError("Regressor validation predictions are empty.")

    classifier["MthCalDt"] = pd.to_datetime(classifier["MthCalDt"], errors="raise")
    regressor["MthCalDt"] = pd.to_datetime(regressor["MthCalDt"], errors="raise")

    classifier = classifier[MERGE_KEYS + [TARGET_COLUMN, CLASSIFIER_SCORE_COLUMN]]
    regressor = regressor[MERGE_KEYS + [TARGET_COLUMN, regressor_column]]

    merged = classifier.merge(
        regressor,
        on=MERGE_KEYS,
        how="inner",
        suffixes=("_cls", "_reg"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No overlapping validation rows after merging classifier and regressor predictions.")

    target_delta = (merged[f"{TARGET_COLUMN}_cls"] - merged[f"{TARGET_COLUMN}_reg"]).abs().max()
    if pd.notna(target_delta) and target_delta > 1e-10:
        raise ValueError(f"Merged target returns disagree across files; max absolute difference is {target_delta:.6g}.")

    merged = merged.rename(
        columns={
            f"{TARGET_COLUMN}_cls": TARGET_COLUMN,
            CLASSIFIER_SCORE_COLUMN: "s_cls",
            regressor_column: "r_hat",
        }
    )
    return merged[MERGE_KEYS + [TARGET_COLUMN, "s_cls", "r_hat"]], regressor_column


def add_monthly_regressor_zscore(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("MthCalDt", observed=True)["r_hat"]
    month_mean = grouped.transform("mean")
    month_std = grouped.transform(lambda values: values.std(ddof=0))
    out["z_reg"] = (out["r_hat"] - month_mean) / month_std.replace(0.0, np.nan)
    return out


def fit_ols(y: np.ndarray, x_columns: list[np.ndarray]) -> tuple[np.ndarray, float]:
    x = np.column_stack([np.ones(len(y), dtype="float64"), *x_columns])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residual = y - fitted
    sse = float(np.dot(residual, residual))
    centered = y - y.mean()
    sst = float(np.dot(centered, centered))
    r2 = np.nan if sst <= 0 else 1.0 - sse / sst
    return beta, r2


def run_monthly_regressions(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for month, part in df.groupby("MthCalDt", sort=True, observed=True):
        part = part[[TARGET_COLUMN, "s_cls", "z_reg"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(part) <= 3:
            continue

        y = part[TARGET_COLUMN].to_numpy(dtype="float64")
        s_cls = part["s_cls"].to_numpy(dtype="float64")
        z_reg = part["z_reg"].to_numpy(dtype="float64")

        full_beta, full_r2 = fit_ols(y, [s_cls, z_reg])
        cls_beta, cls_r2 = fit_ols(y, [s_cls])
        reg_beta, reg_r2 = fit_ols(y, [z_reg])

        rows.append(
            {
                "month": pd.Timestamp(month).date().isoformat(),
                "n_obs": int(len(part)),
                "beta_cls": float(full_beta[1]),
                "beta_reg": float(full_beta[2]),
                "r2": float(full_r2),
                "beta_cls_only": float(cls_beta[1]),
                "r2_cls_only": float(cls_r2),
                "beta_reg_only": float(reg_beta[1]),
                "r2_reg_only": float(reg_r2),
            }
        )

    monthly = pd.DataFrame(rows)
    if monthly.empty:
        raise ValueError("No valid monthly cross-sectional regressions could be estimated.")
    return monthly


def monthly_tstat(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    std = values.std(ddof=1)
    if std == 0 or pd.isna(std):
        return float("nan")
    return float(values.mean() / (std / np.sqrt(len(values))))


def summarize(monthly: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "beta_cls_mean": float(monthly["beta_cls"].mean()),
        "beta_reg_mean": float(monthly["beta_reg"].mean()),
        "beta_cls_tstat": monthly_tstat(monthly["beta_cls"]),
        "beta_reg_tstat": monthly_tstat(monthly["beta_reg"]),
        "mean_r2": float(monthly["r2"].mean()),
        "num_months": int(len(monthly)),
        "mean_r2_cls_only": float(monthly["r2_cls_only"].mean()),
        "mean_r2_reg_only": float(monthly["r2_reg_only"].mean()),
    }
    return pd.DataFrame([summary])


def print_interpretation(summary: pd.DataFrame) -> None:
    row = summary.iloc[0]
    beta_reg = row["beta_reg_mean"]
    tstat_reg = row["beta_reg_tstat"]
    adds_signal = pd.notna(tstat_reg) and abs(tstat_reg) >= 1.96

    log("")
    log("Cross-sectional incremental-signal diagnostic")
    log(f"Months estimated: {int(row['num_months'])}")
    log(f"Mean beta_cls: {row['beta_cls_mean']:.6g} (t={row['beta_cls_tstat']:.3f})")
    log(f"Mean beta_reg: {beta_reg:.6g} (t={tstat_reg:.3f})")
    log(f"Mean full-model R^2: {row['mean_r2']:.6g}")
    log(f"Mean classifier-only R^2: {row['mean_r2_cls_only']:.6g}")
    log(f"Mean regressor-only R^2: {row['mean_r2_reg_only']:.6g}")
    if adds_signal:
        log("Interpretation: beta_reg is statistically different from zero at the 5% two-sided threshold.")
        log("The MLP regressor appears to add incremental information beyond the classifier score.")
    else:
        log("Interpretation: beta_reg is not statistically different from zero at the 5% two-sided threshold.")
        log("The diagnostic does not show clear incremental regressor signal beyond the classifier score.")


def main() -> None:
    args = parse_args()
    log("Loading saved validation predictions...")
    aligned, regressor_column = load_and_align(args.classifier_predictions, args.regressor_predictions)
    log(f"Aligned validation rows: {len(aligned):,}")
    log(f"Using regressor prediction column: {regressor_column}")

    aligned = add_monthly_regressor_zscore(aligned)
    monthly = run_monthly_regressions(aligned)
    summary = summarize(monthly)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    log(f"Wrote aggregate results to {args.output}")
    print_interpretation(summary)


if __name__ == "__main__":
    main()
