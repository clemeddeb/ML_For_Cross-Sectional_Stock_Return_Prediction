#!/usr/bin/env python3
"""Forecasting metrics from merged model prediction artifacts.

This script evaluates cross-sectional return forecasts using only saved
predictions. It creates report-ready CSV tables for monthly Rank ICs,
monthly top-bottom spreads, and split-level summary metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PREDICTIONS = Path("outputs/predictions/dl_xgb_predictions.parquet")
DEFAULT_TABLE_DIR = Path("outputs/tables")
VALIDATION_SPLITS = ("validation", "test")
MIN_MONTHLY_OBS = 5
TOP_BOTTOM_FRACTION = 0.20

BASE_COLUMNS = [
    "PERMNO",
    "MthCalDt",
    "split",
    "target_ret_1m",
    "top_bottom_label",
]

SCORE_COLUMNS = {
    "mlp_classifier": "prediction_mlp_classifier_score",
    "ft_classifier": "prediction_ft_classifier_score",
    "xgb_classifier": "prediction_xgb_classifier_score",
    "temporal_classifier": "prediction_temporal_classifier_score",
    "mlp_er_train": "prediction_mlp_er_train_score",
    "ft_er_train": "prediction_ft_er_train_score",
    "xgb_er_train": "prediction_xgb_er_train_score",
    "temporal_er_train": "prediction_temporal_er_train_score",
}

TEMPORAL_PROB_COLUMNS = [
    "prob_temporal_bottom",
    "prob_temporal_middle",
    "prob_temporal_top",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build forecasting summary tables from saved prediction artifacts."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--min-monthly-obs", type=int, default=MIN_MONTHLY_OBS)
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction artifact not found: {path}")
    df = pd.read_parquet(path)
    missing_base = sorted(set(BASE_COLUMNS) - set(df.columns))
    if missing_base:
        raise ValueError(f"Prediction artifact is missing required columns: {missing_base}")

    df = df.copy()
    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"], errors="raise")
    df["split"] = df["split"].astype("string")
    df["target_ret_1m"] = pd.to_numeric(df["target_ret_1m"], errors="coerce")

    duplicate_count = int(df.duplicated(["PERMNO", "MthCalDt"]).sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count:,} duplicate PERMNO-MthCalDt rows.")
    return df


def add_xgb_er_train_score(df: pd.DataFrame) -> None:
    score_col = SCORE_COLUMNS["xgb_er_train"]
    if score_col in df.columns:
        return
    fallback_col = "prediction_xgb_regressor"
    if fallback_col not in df.columns:
        raise ValueError(
            f"Missing {score_col}; expected fallback column {fallback_col} was not found."
        )
    df[score_col] = pd.to_numeric(df[fallback_col], errors="coerce")


def add_temporal_er_train_score(df: pd.DataFrame) -> None:
    score_col = SCORE_COLUMNS["temporal_er_train"]
    if score_col in df.columns:
        return

    missing_probs = sorted(set(TEMPORAL_PROB_COLUMNS) - set(df.columns))
    if missing_probs:
        raise ValueError(
            f"Missing {score_col}; cannot reconstruct it without {missing_probs}."
        )

    train = df.loc[df["split"].eq("train"), ["top_bottom_label", "target_ret_1m"]].dropna()
    class_means = train.groupby("top_bottom_label")["target_ret_1m"].mean()
    expected_labels = [0, 1, 2]
    missing_labels = sorted(set(expected_labels) - set(class_means.index.astype(int)))
    if missing_labels:
        raise ValueError(f"Training split has no rows for class labels: {missing_labels}")

    means = class_means.reindex(expected_labels).to_numpy(dtype=np.float64)
    probabilities = df[TEMPORAL_PROB_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df[score_col] = probabilities.to_numpy(dtype=np.float64) @ means


def ensure_score_columns(df: pd.DataFrame) -> None:
    add_xgb_er_train_score(df)
    add_temporal_er_train_score(df)

    missing_scores = sorted(set(SCORE_COLUMNS.values()) - set(df.columns))
    if missing_scores:
        raise ValueError(f"Missing required score columns: {missing_scores}")

    for column in SCORE_COLUMNS.values():
        df[column] = pd.to_numeric(df[column], errors="coerce")


def summarize_time_series(values: pd.Series) -> pd.Series:
    clean = values.dropna()
    n_obs = int(clean.size)
    mean = float(clean.mean()) if n_obs else np.nan
    std = float(clean.std(ddof=1)) if n_obs > 1 else np.nan
    t_stat = mean / (std / np.sqrt(n_obs)) if n_obs > 1 and std > 0 else np.nan
    return pd.Series({"mean": mean, "tstat": t_stat, "num_months": n_obs})


def compute_monthly_rank_ic(
    df: pd.DataFrame, score_columns: dict[str, str], min_monthly_obs: int
) -> pd.DataFrame:
    rows = []
    keys = ["split", "MthCalDt"]
    eval_df = df.loc[df["split"].isin(VALIDATION_SPLITS)]

    for model, score_col in score_columns.items():
        work = eval_df[keys + [score_col, "target_ret_1m"]].dropna().copy()
        if work.empty:
            continue

        grouped = work.groupby(keys, sort=True)
        work["_n_obs"] = grouped[score_col].transform("size")
        work["_score_nunique"] = grouped[score_col].transform("nunique")
        work["_ret_nunique"] = grouped["target_ret_1m"].transform("nunique")
        work = work.loc[
            work["_n_obs"].ge(min_monthly_obs)
            & work["_score_nunique"].gt(1)
            & work["_ret_nunique"].gt(1)
        ].copy()
        if work.empty:
            continue

        grouped = work.groupby(keys, sort=True)
        work["_score_rank"] = grouped[score_col].rank(method="average")
        work["_return_rank"] = grouped["target_ret_1m"].rank(method="average")
        corr = grouped[["_score_rank", "_return_rank"]].corr()
        rank_ic = corr.loc[(slice(None), slice(None), "_score_rank"), "_return_rank"]
        rank_ic = rank_ic.droplevel(2).rename("rank_ic_t").reset_index()
        rank_ic["model"] = model
        rank_ic["num_obs"] = grouped.size().to_numpy()
        rows.append(rank_ic[["model", "split", "MthCalDt", "rank_ic_t", "num_obs"]])

    if not rows:
        return pd.DataFrame(columns=["model", "split", "MthCalDt", "rank_ic_t", "num_obs"])
    return pd.concat(rows, ignore_index=True).sort_values(["model", "split", "MthCalDt"])


def compute_monthly_spread(
    df: pd.DataFrame, score_columns: dict[str, str], min_monthly_obs: int
) -> pd.DataFrame:
    rows = []
    keys = ["split", "MthCalDt"]
    eval_df = df.loc[df["split"].isin(VALIDATION_SPLITS)]

    for model, score_col in score_columns.items():
        work = eval_df[keys + [score_col, "target_ret_1m"]].dropna().copy()
        if work.empty:
            continue

        grouped = work.groupby(keys, sort=True)
        work["_n_obs"] = grouped[score_col].transform("size")
        work["_score_nunique"] = grouped[score_col].transform("nunique")
        work = work.loc[work["_n_obs"].ge(min_monthly_obs) & work["_score_nunique"].gt(1)].copy()
        if work.empty:
            continue

        grouped = work.groupby(keys, sort=True)
        work["_score_pct_rank"] = grouped[score_col].rank(method="first", pct=True)
        work["_bucket"] = np.select(
            [
                work["_score_pct_rank"].le(TOP_BOTTOM_FRACTION),
                work["_score_pct_rank"].gt(1.0 - TOP_BOTTOM_FRACTION),
            ],
            ["bottom", "top"],
            default="middle",
        )
        tails = work.loc[work["_bucket"].isin(["bottom", "top"])]
        means = (
            tails.groupby(keys + ["_bucket"], sort=True)["target_ret_1m"]
            .mean()
            .unstack("_bucket")
        )
        counts = tails.groupby(keys + ["_bucket"], sort=True)["target_ret_1m"].size().unstack("_bucket")
        means = means.dropna(subset=["bottom", "top"])
        counts = counts.reindex(means.index).fillna(0).astype(int)

        spread = means.assign(
            spread_t=means["top"] - means["bottom"],
            num_top=counts["top"],
            num_bottom=counts["bottom"],
            model=model,
        ).reset_index()
        rows.append(
            spread[["model", "split", "MthCalDt", "spread_t", "num_top", "num_bottom"]]
        )

    if not rows:
        return pd.DataFrame(
            columns=["model", "split", "MthCalDt", "spread_t", "num_top", "num_bottom"]
        )
    return pd.concat(rows, ignore_index=True).sort_values(["model", "split", "MthCalDt"])


def build_summary(monthly_ic: pd.DataFrame, monthly_spread: pd.DataFrame) -> pd.DataFrame:
    ic_summary = (
        monthly_ic.groupby(["model", "split"], sort=False)["rank_ic_t"]
        .apply(summarize_time_series)
        .unstack()
        .rename(columns={"mean": "mean_rank_ic", "tstat": "rank_ic_tstat"})
        .reset_index()
    )
    hit_rate = (
        monthly_ic.assign(hit=monthly_ic["rank_ic_t"].gt(0))
        .groupby(["model", "split"], sort=False)["hit"]
        .mean()
        .rename("hit_rate")
        .reset_index()
    )
    spread_summary = (
        monthly_spread.groupby(["model", "split"], sort=False)["spread_t"]
        .apply(summarize_time_series)
        .unstack()
        .rename(columns={"mean": "mean_top_bottom_spread", "tstat": "spread_tstat"})
        .drop(columns=["num_months"])
        .reset_index()
    )

    summary = (
        ic_summary.merge(hit_rate, on=["model", "split"], how="left")
        .merge(spread_summary, on=["model", "split"], how="left")
    )
    summary = summary[
        [
            "model",
            "split",
            "mean_rank_ic",
            "rank_ic_tstat",
            "hit_rate",
            "mean_top_bottom_spread",
            "spread_tstat",
            "num_months",
        ]
    ]
    summary["num_months"] = summary["num_months"].astype("Int64")
    return summary.sort_values(["mean_rank_ic", "model", "split"], ascending=[False, True, True])


def print_top_models(summary: pd.DataFrame, split: str) -> None:
    top = (
        summary.loc[summary["split"].eq(split)]
        .sort_values("mean_rank_ic", ascending=False)
        .head(5)
    )
    print(f"\nTop 5 models by Rank IC ({split}):")
    if top.empty:
        print("No models available.")
        return
    print(
        top[
            [
                "model",
                "mean_rank_ic",
                "rank_ic_tstat",
                "hit_rate",
                "mean_top_bottom_spread",
                "spread_tstat",
                "num_months",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )


def main() -> None:
    args = parse_args()
    if args.min_monthly_obs < 2:
        raise ValueError("--min-monthly-obs must be at least 2.")

    predictions = load_predictions(args.predictions)
    ensure_score_columns(predictions)

    monthly_ic = compute_monthly_rank_ic(predictions, SCORE_COLUMNS, args.min_monthly_obs)
    monthly_spread = compute_monthly_spread(predictions, SCORE_COLUMNS, args.min_monthly_obs)
    summary = build_summary(monthly_ic, monthly_spread)

    args.table_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.table_dir / "forecasting_summary.csv", index=False)
    monthly_ic.to_csv(args.table_dir / "monthly_rank_ic.csv", index=False)
    monthly_spread.to_csv(args.table_dir / "monthly_spread.csv", index=False)

    print(f"Saved summary table: {args.table_dir / 'forecasting_summary.csv'}")
    print(f"Saved monthly Rank ICs: {args.table_dir / 'monthly_rank_ic.csv'}")
    print(f"Saved monthly spreads: {args.table_dir / 'monthly_spread.csv'}")
    print_top_models(summary, "validation")
    print_top_models(summary, "test")


if __name__ == "__main__":
    main()
