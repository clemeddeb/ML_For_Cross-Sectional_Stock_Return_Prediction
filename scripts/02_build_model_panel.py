#!/usr/bin/env python3
"""Build the base monthly modeling panel with one-month-ahead targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_INPUT = Path("Dataset/Processed/crsp_compustat_panel.parquet")
DEFAULT_OUTPUT = Path("Dataset/Processed/model_panel_monthly_base.parquet")
DEFAULT_SUMMARY = Path("outputs/sanity_checks/tables/model_panel_base_summary.csv")
DEFAULT_DISTRIBUTION = Path(
    "outputs/sanity_checks/tables/model_panel_target_distribution.csv"
)

KEY_COLUMNS = ["permno", "mthcaldt"]
TARGET_COLUMN = "target_ret_1m"
TARGET_MONTH_COLUMN = "target_month"
QUINTILE_COLUMN = "target_quintile"
TOP_BOTTOM_COLUMN = "top_bottom_label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the monthly base modeling panel from linked CRSP/Compustat data."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--distribution", type=Path, default=DEFAULT_DISTRIBUTION)
    return parser.parse_args()


def assert_unique_permno_month(df: pd.DataFrame) -> None:
    duplicates = df.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        sample = df.loc[duplicates, KEY_COLUMNS].head(20)
        raise ValueError(
            "Input panel has duplicate PERMNO-MthCalDt rows. "
            f"First duplicate keys:\n{sample.to_string(index=False)}"
        )


def add_next_month_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True).copy()
    out["_month_period"] = out["mthcaldt"].dt.to_period("M")
    out["_next_month_period"] = out.groupby("permno", observed=True)["_month_period"].shift(-1)
    out[TARGET_COLUMN] = out.groupby("permno", observed=True)["mthret"].shift(-1)
    out[TARGET_MONTH_COLUMN] = out["mthcaldt"] + pd.offsets.MonthEnd(1)
    consecutive = out["_next_month_period"].eq(out["_month_period"] + 1)
    out.loc[~consecutive, TARGET_COLUMN] = pd.NA
    return out


def assign_cross_sectional_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def month_quintiles(targets: pd.Series) -> pd.Series:
        valid = targets.notna()
        result = pd.Series(pd.NA, index=targets.index, dtype="Int64")
        if valid.sum() < 5:
            return result
        ranked = targets.loc[valid].rank(method="first")
        result.loc[valid] = pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5]).astype("Int64")
        return result

    out[QUINTILE_COLUMN] = (
        out.groupby("_month_period", group_keys=False)[TARGET_COLUMN].apply(month_quintiles)
    )
    out[TOP_BOTTOM_COLUMN] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[out[QUINTILE_COLUMN].eq(1), TOP_BOTTOM_COLUMN] = 0
    out.loc[out[QUINTILE_COLUMN].isin([2, 3, 4]), TOP_BOTTOM_COLUMN] = 1
    out.loc[out[QUINTILE_COLUMN].eq(5), TOP_BOTTOM_COLUMN] = 2
    return out


def output_columns(df: pd.DataFrame) -> list[str]:
    identifier_cols = [
        "permno",
        "gvkey",
        "mthcaldt",
        "ticker",
        "siccd",
        "naics",
    ]
    return_cols = ["mthret", "sprtrn"]
    comp_cols = [
        col
        for col in df.columns
        if col.startswith("comp_")
        or col in ["compustat_age_days", "compustat_match"]
    ]
    target_cols = [TARGET_MONTH_COLUMN, TARGET_COLUMN, QUINTILE_COLUMN, TOP_BOTTOM_COLUMN]
    requested = identifier_cols + return_cols + comp_cols + target_cols
    return [col for col in requested if col in df.columns]


def input_columns(path: Path) -> list[str]:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    identifier_cols = ["permno", "gvkey", "mthcaldt", "ticker", "siccd", "naics"]
    return_cols = ["mthret", "sprtrn"]
    comp_cols = [
        col
        for col in available
        if col.startswith("comp_")
        or col in ["compustat_age_days", "compustat_match"]
    ]
    requested = identifier_cols + return_cols + sorted(comp_cols)
    return [col for col in requested if col in available]


def build_summary(
    before_drop: pd.DataFrame,
    after_drop: pd.DataFrame,
    missing_target_rate: float,
) -> pd.DataFrame:
    rows = [
        {
            "section": "overall",
            "year": pd.NA,
            "number_of_rows": len(before_drop),
            "number_of_stocks": before_drop["permno"].nunique(dropna=True),
            "first_month": before_drop["mthcaldt"].min(),
            "last_month": before_drop["mthcaldt"].max(),
            "missing_target_rate_before_dropping": missing_target_rate,
            "rows_after_dropping_missing_targets": len(after_drop),
            "rows_per_year": pd.NA,
            "stocks_per_year": pd.NA,
        }
    ]

    yearly = (
        after_drop.assign(year=after_drop["mthcaldt"].dt.year)
        .groupby("year", dropna=True)
        .agg(rows_per_year=("permno", "size"), stocks_per_year=("permno", "nunique"))
        .reset_index()
        .sort_values("year")
    )
    for _, row in yearly.iterrows():
        rows.append(
            {
                "section": "year",
                "year": int(row["year"]),
                "number_of_rows": pd.NA,
                "number_of_stocks": pd.NA,
                "first_month": pd.NaT,
                "last_month": pd.NaT,
                "missing_target_rate_before_dropping": pd.NA,
                "rows_after_dropping_missing_targets": pd.NA,
                "rows_per_year": int(row["rows_per_year"]),
                "stocks_per_year": int(row["stocks_per_year"]),
            }
        )
    return pd.DataFrame(rows)


def build_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in [QUINTILE_COLUMN, TOP_BOTTOM_COLUMN]:
        counts = df[column].value_counts(dropna=False).sort_index()
        total = int(counts.sum())
        for label, count in counts.items():
            rows.append(
                {
                    "label_type": column,
                    "label": label,
                    "rows": int(count),
                    "pct": round(100 * int(count) / total, 4) if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing linked panel: {args.input}")

    print(f"Reading linked CRSP/Compustat panel: {args.input}", flush=True)
    columns = input_columns(args.input)
    panel = pd.read_parquet(args.input, columns=columns)
    panel["mthcaldt"] = pd.to_datetime(panel["mthcaldt"], errors="coerce")
    assert_unique_permno_month(panel)
    print(f"Input rows: {len(panel):,}", flush=True)
    print("Duplicate PERMNO-MthCalDt rows: 0", flush=True)

    print("Creating next-month target and cross-sectional labels...", flush=True)
    panel = add_next_month_target(panel)
    missing_target_rate = round(100 * panel[TARGET_COLUMN].isna().mean(), 4)
    missing_targets = int(panel[TARGET_COLUMN].isna().sum())
    print(
        f"Missing next-month targets before dropping: {missing_targets:,} ({missing_target_rate:.4f}%)",
        flush=True,
    )

    panel = assign_cross_sectional_labels(panel)
    model_panel = panel.dropna(subset=[TARGET_COLUMN]).copy()
    model_panel = model_panel[output_columns(model_panel)]
    model_panel = model_panel.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_panel.to_parquet(args.output, index=False)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(panel, model_panel, missing_target_rate)
    summary.to_csv(args.summary, index=False)

    distribution = build_distribution(model_panel)
    distribution.to_csv(args.distribution, index=False)

    duplicate_after = model_panel.duplicated(KEY_COLUMNS, keep=False).sum()
    print(f"Rows after dropping missing targets: {len(model_panel):,}", flush=True)
    print(f"Unique PERMNOs: {model_panel['permno'].nunique(dropna=True):,}", flush=True)
    print(
        f"Date range: {model_panel['mthcaldt'].min().date()} to {model_panel['mthcaldt'].max().date()}",
        flush=True,
    )
    print(f"Duplicate PERMNO-MthCalDt rows in output: {duplicate_after:,}", flush=True)
    print(f"Wrote model panel: {args.output}", flush=True)
    print(f"Wrote summary table: {args.summary}", flush=True)
    print(f"Wrote target distribution: {args.distribution}", flush=True)


if __name__ == "__main__":
    main()
