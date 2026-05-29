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
DEFAULT_PREDICTION_DIR = Path("outputs/predictions")
VALIDATION_SPLITS = ("validation", "test")
MIN_MONTHLY_OBS = 5
TOP_BOTTOM_FRACTION = 0.20

SCORE_METADATA = [
    {
        "model": "xgb_classifier",
        "model_family": "XGB",
        "model_name": "XGB",
        "score": "classifier",
        "column": "prediction_xgb_classifier_score",
        "source": DEFAULT_PREDICTIONS,
    },
    {
        "model": "xgb_er_train",
        "model_family": "XGB",
        "model_name": "XGB",
        "score": "ER_train",
        "column": "prediction_xgb_er_train_score",
        "source": DEFAULT_PREDICTIONS,
    },
    {
        "model": "mlp_classifier",
        "model_family": "MLP",
        "model_name": "MLP",
        "score": "classifier",
        "column": "prediction_mlp_classifier_score",
        "source": DEFAULT_PREDICTIONS,
    },
    {
        "model": "mlp_er_train",
        "model_family": "MLP",
        "model_name": "MLP",
        "score": "ER_train",
        "column": "prediction_mlp_er_train_score",
        "source": DEFAULT_PREDICTIONS,
    },
]

FT_VARIANT_SPECS = [
    ("ft_base_seed362559", "FT-Transformer base seed362559"),
    ("ft_lr2e5_seed362559", "FT-Transformer lr2e-5 seed362559"),
    ("ft_lr5e5_seed362559", "FT-Transformer lr5e-5 seed362559"),
    ("ft_reg_seed362559", "FT-Transformer regularized seed362559"),
    ("ft_small_lr5e5_seed362559", "FT-Transformer small lr5e-5 seed362559"),
    ("ft_small_seed362559", "FT-Transformer small seed362559"),
    ("ft_transformer", "FT-Transformer main"),
]

TTT_VARIANT_SPECS = [
    ("temporal_tabular_small_ft_warmstart", "TTT small FT warmstart"),
    ("temporal_tabular_small_mlp_branch", "TTT small MLP branch"),
    ("temporal_tabular_small_mlp_warmstart", "TTT small MLP warmstart"),
    ("temporal_tabular_small_random", "TTT small random"),
    ("temporal_tabular_tiny_random", "TTT tiny random"),
    ("temporal_tabular_transformer_seed362559", "TTT full seed362559"),
]

for run_name, display_name in FT_VARIANT_SPECS:
    SCORE_METADATA.extend(
        [
            {
                "model": f"{run_name}_classifier",
                "model_family": "FT-Transformer",
                "model_name": display_name,
                "score": "classifier",
                "column": "prediction_ft_classifier_score",
                "source": DEFAULT_PREDICTION_DIR / f"{run_name}_predictions.parquet",
            },
            {
                "model": f"{run_name}_er_train",
                "model_family": "FT-Transformer",
                "model_name": display_name,
                "score": "ER_train",
                "column": "prediction_ft_er_train_score",
                "source": DEFAULT_PREDICTION_DIR / f"{run_name}_predictions.parquet",
            },
        ]
    )

for run_name, display_name in TTT_VARIANT_SPECS:
    SCORE_METADATA.extend(
        [
            {
                "model": f"{run_name}_classifier",
                "model_family": "TTT",
                "model_name": display_name,
                "score": "classifier",
                "column": "prediction_temporal_tabular_classifier_score",
                "source": DEFAULT_PREDICTION_DIR / f"{run_name}_predictions.parquet",
            },
            {
                "model": f"{run_name}_er_train",
                "model_family": "TTT",
                "model_name": display_name,
                "score": "ER_train",
                "column": "prediction_temporal_tabular_er_train_score",
                "source": DEFAULT_PREDICTION_DIR / f"{run_name}_predictions.parquet",
            },
        ]
    )

SCORE_COLUMNS = {item["model"]: item["column"] for item in SCORE_METADATA}

LEGACY_DLXGB_SCORE_COLUMNS = {
    "xgb_er_train": "prediction_xgb_er_train_score",
    "mlp_er_train": "prediction_mlp_er_train_score",
    "temporal_er_train": "prediction_temporal_er_train_score",
}

FAMILY_OUTPUT_NAMES = {
    "XGB": "xgb",
    "MLP": "mlp",
    "FT-Transformer": "ft_transformer",
    "TTT": "ttt",
}
FAMILY_ORDER = list(FAMILY_OUTPUT_NAMES)

BASE_COLUMNS = [
    "PERMNO",
    "MthCalDt",
    "split",
    "target_ret_1m",
    "top_bottom_label",
]

TEMPORAL_PROB_COLUMNS = [
    "prob_temporal_bottom",
    "prob_temporal_middle",
    "prob_temporal_top",
]

LEGACY_DLXGB_SCORE_METADATA = [
    {
        "model": "mlp_classifier",
        "model_family": "MLP",
        "model_name": "MLP",
        "score": "classifier",
        "column": "prediction_mlp_classifier_score",
        "source": DEFAULT_PREDICTIONS,
    },
    {
        "model": "mlp_er_train",
        "model_family": "MLP",
        "model_name": "MLP",
        "score": "ER_train",
        "column": "prediction_mlp_er_train_score",
        "source": DEFAULT_PREDICTIONS,
    },
    {
        "model": "xgb_classifier",
        "model_family": "XGB",
        "model_name": "XGB",
        "score": "classifier",
        "column": "prediction_xgb_classifier_score",
        "source": DEFAULT_PREDICTIONS,
    },
    {
        "model": "xgb_er_train",
        "model_family": "XGB",
        "model_name": "XGB",
        "score": "ER_train",
        "column": "prediction_xgb_er_train_score",
        "source": DEFAULT_PREDICTIONS,
    },
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


def add_xgb_er_train_score(df: pd.DataFrame, score_columns: dict[str, str]) -> None:
    score_col = score_columns.get("xgb_er_train")
    if score_col is None:
        return
    if score_col in df.columns:
        return
    fallback_col = "prediction_xgb_regressor"
    if fallback_col not in df.columns:
        raise ValueError(
            f"Missing {score_col}; expected fallback column {fallback_col} was not found."
        )
    df[score_col] = pd.to_numeric(df[fallback_col], errors="coerce")


def add_temporal_er_train_score(df: pd.DataFrame, score_columns: dict[str, str]) -> None:
    score_col = score_columns.get("temporal_er_train")
    if score_col is None:
        return
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


def ensure_score_columns(df: pd.DataFrame, score_columns: dict[str, str]) -> None:
    add_xgb_er_train_score(df, score_columns)
    add_temporal_er_train_score(df, score_columns)

    missing_scores = sorted(set(score_columns.values()) - set(df.columns))
    if missing_scores:
        raise ValueError(f"Missing required score columns: {missing_scores}")

    for column in score_columns.values():
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


def report_metadata(prediction_path: Path) -> list[dict[str, object]]:
    metadata = []
    for item in SCORE_METADATA:
        row = item.copy()
        if Path(row["source"]) == DEFAULT_PREDICTIONS:
            row["source"] = prediction_path
        metadata.append(row)
    return metadata


def build_forecasting_metrics(
    metadata: list[dict[str, object]], min_monthly_obs: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_ic_frames = []
    monthly_spread_frames = []

    by_source: dict[Path, list[dict[str, object]]] = {}
    for item in metadata:
        by_source.setdefault(Path(item["source"]), []).append(item)

    for source, items in by_source.items():
        predictions = load_predictions(source)
        score_columns = {str(item["model"]): str(item["column"]) for item in items}
        ensure_score_columns(predictions, score_columns)
        monthly_ic_frames.append(
            compute_monthly_rank_ic(predictions, score_columns, min_monthly_obs)
        )
        monthly_spread_frames.append(
            compute_monthly_spread(predictions, score_columns, min_monthly_obs)
        )

    monthly_ic = pd.concat(monthly_ic_frames, ignore_index=True)
    monthly_spread = pd.concat(monthly_spread_frames, ignore_index=True)
    summary = build_summary(monthly_ic, monthly_spread)
    return monthly_ic, monthly_spread, summary


def significance_stars(t_stat: float) -> str:
    if pd.isna(t_stat):
        return ""
    abs_t = abs(float(t_stat))
    if abs_t >= 2.576:
        return "***"
    if abs_t >= 1.960:
        return "**"
    if abs_t >= 1.645:
        return "*"
    return ""


def format_with_stars(value: float, t_stat: float, decimals: int = 4) -> str:
    if pd.isna(value):
        return ""
    stars = significance_stars(t_stat)
    formatted = f"{float(value):.{decimals}f}"
    return f"{formatted}^{{{stars}}}" if stars else formatted


def build_family_report_tables(
    summary: pd.DataFrame, metadata_rows: list[dict[str, object]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.DataFrame(metadata_rows)[["model", "model_family", "model_name", "score"]]
    wide = summary.pivot(index="model", columns="split")
    wide.columns = [f"{metric}_{split}" for metric, split in wide.columns]
    wide = wide.reset_index().merge(metadata, on="model", how="left")
    wide = wide.loc[wide["model_family"].notna()].copy()

    raw_columns = [
        "model_family",
        "model_name",
        "score",
        "model",
        "mean_rank_ic_validation",
        "rank_ic_tstat_validation",
        "hit_rate_validation",
        "mean_top_bottom_spread_validation",
        "spread_tstat_validation",
        "mean_rank_ic_test",
        "rank_ic_tstat_test",
        "hit_rate_test",
        "mean_top_bottom_spread_test",
        "spread_tstat_test",
        "num_months_validation",
        "num_months_test",
    ]
    raw = wide[raw_columns].copy()
    raw["model_family"] = pd.Categorical(raw["model_family"], FAMILY_ORDER, ordered=True)
    raw["dedupe_priority"] = np.where(
        raw["model_name"].eq("FT-Transformer main")
        | raw["model_name"].eq("TTT full seed362559"),
        0,
        1,
    )
    metric_signature_cols = [
        "model_family",
        "score",
        "mean_rank_ic_validation",
        "rank_ic_tstat_validation",
        "hit_rate_validation",
        "mean_top_bottom_spread_validation",
        "spread_tstat_validation",
        "mean_rank_ic_test",
        "rank_ic_tstat_test",
        "hit_rate_test",
        "mean_top_bottom_spread_test",
        "spread_tstat_test",
    ]
    raw = (
        raw.sort_values(
            ["model_family", "mean_rank_ic_validation", "dedupe_priority"],
            ascending=[True, False, True],
        )
        .drop_duplicates(metric_signature_cols, keep="first")
        .groupby("model_family", group_keys=False, observed=True)
        .head(3)
        .reset_index(drop=True)
    )
    raw["model_family"] = raw["model_family"].astype(str)
    raw = raw.drop(columns="dedupe_priority")

    display = raw.copy()
    display["Mean Rank IC (val)"] = display.apply(
        lambda row: format_with_stars(row["mean_rank_ic_validation"], row["rank_ic_tstat_validation"]),
        axis=1,
    )
    display["Mean Rank IC (test)"] = display.apply(
        lambda row: format_with_stars(row["mean_rank_ic_test"], row["rank_ic_tstat_test"]),
        axis=1,
    )
    display["IC hit rate"] = display["hit_rate_test"].map(
        lambda value: "" if pd.isna(value) else f"{float(value):.2%}"
    )
    display["Top-bottom spread"] = display.apply(
        lambda row: format_with_stars(
            row["mean_top_bottom_spread_test"], row["spread_tstat_test"]
        ),
        axis=1,
    )
    display = display.rename(
        columns={"model_family": "Model family", "model_name": "Model", "score": "Score"}
    )
    display = display[
        [
            "Model family",
            "Model",
            "Score",
            "Mean Rank IC (val)",
            "Mean Rank IC (test)",
            "IC hit rate",
            "Top-bottom spread",
        ]
    ]
    return raw, display


def build_best_family_comparison(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_counts = {"XGB": 1, "MLP": 1, "FT-Transformer": 2, "TTT": 2}
    selected = []
    for family in FAMILY_ORDER:
        n_rows = selection_counts.get(family, 0)
        if n_rows == 0:
            continue
        family_rows = raw.loc[raw["model_family"].eq(family)].head(n_rows)
        if not family_rows.empty:
            selected.append(family_rows)

    if selected:
        raw_out = pd.concat(selected, ignore_index=True)
    else:
        raw_out = pd.DataFrame(columns=raw.columns)

    raw_out = raw_out[
        [
            "model_family",
            "model_name",
            "score",
            "mean_rank_ic_validation",
            "rank_ic_tstat_validation",
            "hit_rate_validation",
            "mean_top_bottom_spread_validation",
            "mean_rank_ic_test",
            "rank_ic_tstat_test",
            "hit_rate_test",
            "mean_top_bottom_spread_test",
        ]
    ].rename(
        columns={
            "model_family": "model_family",
            "model_name": "model",
            "score": "score",
            "mean_rank_ic_validation": "val_rank_ic",
            "rank_ic_tstat_validation": "val_rank_ic_tstat",
            "hit_rate_validation": "val_ic_hit_rate",
            "mean_top_bottom_spread_validation": "val_top_bottom_spread",
            "mean_rank_ic_test": "test_rank_ic",
            "rank_ic_tstat_test": "test_rank_ic_tstat",
            "hit_rate_test": "test_ic_hit_rate",
            "mean_top_bottom_spread_test": "test_top_bottom_spread",
        }
    )

    display = raw_out.copy()
    for col in [
        "val_rank_ic",
        "val_rank_ic_tstat",
        "val_top_bottom_spread",
        "test_rank_ic",
        "test_rank_ic_tstat",
        "test_top_bottom_spread",
    ]:
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    for col in ["val_ic_hit_rate", "test_ic_hit_rate"]:
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.2%}")

    display = display.rename(
        columns={
            "model_family": "Model family",
            "model": "Model",
            "score": "Score",
            "val_rank_ic": "Val Rank IC",
            "val_rank_ic_tstat": "Val t-stat",
            "val_ic_hit_rate": "Val IC hit rate",
            "val_top_bottom_spread": "Val top-bottom spread",
            "test_rank_ic": "Test Rank IC",
            "test_rank_ic_tstat": "Test t-stat",
            "test_ic_hit_rate": "Test IC hit rate",
            "test_top_bottom_spread": "Test top-bottom spread",
        }
    )
    return raw_out, display


def write_family_report_tables(
    raw: pd.DataFrame,
    display: pd.DataFrame,
    best_raw: pd.DataFrame,
    best_display: pd.DataFrame,
    table_dir: Path,
) -> None:
    raw.to_csv(table_dir / "forecasting_family_top3_model_score_pairs_raw.csv", index=False)
    display.to_csv(table_dir / "forecasting_family_top3_model_score_pairs.csv", index=False)
    best_raw.to_csv(table_dir / "forecasting_best_family_model_comparison_raw.csv", index=False)
    best_display.to_csv(table_dir / "forecasting_best_family_model_comparison.csv", index=False)

    for family, group in display.groupby("Model family", sort=False):
        name = FAMILY_OUTPUT_NAMES[family]
        group.to_csv(table_dir / f"forecasting_{name}_top3_model_score_pairs.csv", index=False)


def print_family_report_tables(display: pd.DataFrame) -> None:
    print("\nTop 3 model/score pairs by validation Rank IC within each family:")
    for family, group in display.groupby("Model family", sort=False):
        print(f"\n{family}")
        print(group.to_string(index=False))


def print_best_family_comparison(display: pd.DataFrame) -> None:
    print("\nBest family model comparison:")
    print(display.to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.min_monthly_obs < 2:
        raise ValueError("--min-monthly-obs must be at least 2.")

    metadata = report_metadata(args.predictions)
    monthly_ic, monthly_spread, summary = build_forecasting_metrics(
        metadata, args.min_monthly_obs
    )
    family_raw, family_display = build_family_report_tables(summary, metadata)
    best_raw, best_display = build_best_family_comparison(family_raw)

    args.table_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.table_dir / "forecasting_summary.csv", index=False)
    monthly_ic.to_csv(args.table_dir / "monthly_rank_ic.csv", index=False)
    monthly_spread.to_csv(args.table_dir / "monthly_spread.csv", index=False)
    write_family_report_tables(family_raw, family_display, best_raw, best_display, args.table_dir)

    print(f"Saved summary table: {args.table_dir / 'forecasting_summary.csv'}")
    print(f"Saved monthly Rank ICs: {args.table_dir / 'monthly_rank_ic.csv'}")
    print(f"Saved monthly spreads: {args.table_dir / 'monthly_spread.csv'}")
    print(
        "Saved family top-3 tables: "
        f"{args.table_dir / 'forecasting_family_top3_model_score_pairs.csv'}"
    )
    print_family_report_tables(family_display)
    print_best_family_comparison(best_display)


if __name__ == "__main__":
    main()
