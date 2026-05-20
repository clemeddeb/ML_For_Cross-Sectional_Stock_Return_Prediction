#!/usr/bin/env python3
"""Create reproducible time-based train/validation/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_INPUT = Path("Dataset/Processed/model_panel_full_features.parquet")
DEFAULT_OUTPUT = Path("Dataset/Processed/model_panel_full_features_with_splits.parquet")
DEFAULT_TABLE_DIR = Path("outputs/sanity_checks/tables")
DEFAULT_PLOT_DIR = Path("outputs/sanity_checks/plots")
DEFAULT_FEATURE_GROUPS = DEFAULT_TABLE_DIR / "feature_groups.json"
DEFAULT_BATCH_SIZE = 100_000

KEY_COLUMNS = ["permno", "mthcaldt"]
TARGET_MONTH_COLUMN = "target_month"
TARGET_COLUMN = "target_ret_1m"
TARGET_LABEL_COLUMNS = ["target_quintile", "top_bottom_label"]

SAMPLE_START = pd.Timestamp("1990-01-01")
TRAIN_START = pd.Timestamp("1990-01-01")
TRAIN_END = pd.Timestamp("2010-12-31")
VALIDATION_START = pd.Timestamp("2011-01-01")
VALIDATION_END = pd.Timestamp("2015-12-31")
TEST_START = pd.Timestamp("2016-01-01")
TEST_END = pd.Timestamp("2024-11-30")

SPLIT_ORDER = ["train", "validation", "test"]
SPLIT_WINDOWS = {
    "train": (TRAIN_START, TRAIN_END),
    "validation": (VALIDATION_START, VALIDATION_END),
    "test": (TEST_START, TEST_END),
}
PLOT_COLORS = {
    "train": "#2f6f8f",
    "validation": "#b26a2c",
    "test": "#5f7f3a",
}


def next_calendar_month_end(dates: pd.Series) -> pd.Series:
    month_period = pd.to_datetime(dates, errors="coerce").dt.to_period("M")
    return month_period.add(1).dt.to_timestamp("M")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create time-based train/validation/test splits for the full feature panel."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--feature-groups", type=Path, default=DEFAULT_FEATURE_GROUPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def load_feature_groups(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Feature group metadata not found: {path}")
    with path.open(encoding="utf-8") as f:
        groups = json.load(f)
    return {key: list(value) for key, value in groups.items() if isinstance(value, list)}


def feature_group_lookup(groups: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, list[str]] = {}
    for group, columns in groups.items():
        if not group.endswith("_columns"):
            continue
        if group in {"identifier_columns", "target_columns", "excluded_columns"}:
            continue
        group_name = group.removesuffix("_columns")
        for column in columns:
            lookup.setdefault(column, []).append(group_name)
    return {column: ";".join(names) for column, names in lookup.items()}


def selected_model_features(groups: dict[str, list[str]], columns: list[str]) -> list[str]:
    feature_group_names = [
        "raw_return_columns",
        "return_feature_columns",
        "jkp_feature_columns",
        "selected_compustat_feature_columns",
        "selected_compustat_missing_indicator_columns",
    ]
    features: list[str] = []
    seen: set[str] = set()
    missing: list[str] = []
    available = set(columns)
    for group_name in feature_group_names:
        for feature in groups.get(group_name, []):
            if feature in seen:
                continue
            seen.add(feature)
            if feature not in available:
                missing.append(feature)
            else:
                features.append(feature)
    if missing:
        raise ValueError(f"Selected model features missing from input panel: {missing}")
    return features


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def assert_required_columns(columns: list[str]) -> None:
    required = KEY_COLUMNS + [TARGET_MONTH_COLUMN, TARGET_COLUMN] + TARGET_LABEL_COLUMNS
    missing = sorted(set(required).difference(columns))
    if missing:
        raise ValueError(f"Input panel is missing required columns: {missing}")


def assert_unique_permno_month(df: pd.DataFrame) -> None:
    duplicates = df.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        sample = df.loc[duplicates, KEY_COLUMNS].head(20)
        raise ValueError(
            "Panel has duplicate PERMNO-MthCalDt rows. "
            f"First duplicate keys:\n{sample.to_string(index=False)}"
        )


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mthcaldt"] = pd.to_datetime(out["mthcaldt"], errors="coerce")
    out[TARGET_MONTH_COLUMN] = next_calendar_month_end(out["mthcaldt"])
    invalid_dates = int(out["mthcaldt"].isna().sum())
    if invalid_dates:
        raise ValueError(f"Cannot create splits because {invalid_dates:,} rows have invalid dates.")

    out = out.loc[out[TARGET_MONTH_COLUMN].ge(SAMPLE_START)].copy()
    split = pd.Series(pd.NA, index=out.index, dtype="string")
    for split_name, (start, end) in SPLIT_WINDOWS.items():
        mask = out[TARGET_MONTH_COLUMN].between(start, end, inclusive="both")
        split.loc[mask] = split_name
    out["split"] = split
    out = out.loc[out["split"].notna()].copy()
    out["split"] = pd.Categorical(out["split"], categories=SPLIT_ORDER, ordered=True)
    return out


def split_values(dates: pd.Series) -> pd.Series:
    split = pd.Series(pd.NA, index=dates.index, dtype="string")
    for split_name, (start, end) in SPLIT_WINDOWS.items():
        split.loc[dates.between(start, end, inclusive="both")] = split_name
    return split


def validate_splits(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    warnings: list[str] = []
    assert_unique_permno_month(df)

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"{TARGET_COLUMN} must be nonmissing in the split output.")
    if not set(df["split"].astype(str)).issubset(set(SPLIT_ORDER)):
        raise ValueError("Every row must have one of train, validation, or test as split.")
    if df["split"].isna().any():
        raise ValueError("Every row must have a nonmissing split.")
    if df[TARGET_MONTH_COLUMN].lt(SAMPLE_START).any():
        raise ValueError("Rows with target_month before 1990 are included in the split output.")

    date_ranges = {
        split: df.loc[df["split"].eq(split), TARGET_MONTH_COLUMN] for split in SPLIT_ORDER
    }
    empty_splits = [split for split, values in date_ranges.items() if values.empty]
    if empty_splits:
        raise ValueError(f"Split output has empty split(s): {empty_splits}")
    if not date_ranges["train"].max() < date_ranges["validation"].min():
        raise ValueError("Train dates must be strictly before validation dates.")
    if not date_ranges["validation"].max() < date_ranges["test"].min():
        raise ValueError("Validation dates must be strictly before test dates.")

    for feature in feature_columns:
        train_nonmissing = df.loc[df["split"].eq("train"), feature].notna().sum()
        if train_nonmissing == 0:
            raise ValueError(f"Selected model feature is entirely missing in train: {feature}")
        for split in ["validation", "test"]:
            nonmissing = df.loc[df["split"].eq(split), feature].notna().sum()
            if nonmissing == 0:
                warnings.append(f"Selected model feature is entirely missing in {split}: {feature}")
    return warnings


def build_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLIT_ORDER:
        part = df.loc[df["split"].eq(split)]
        target = part[TARGET_COLUMN]
        quantiles = target.quantile([0.01, 0.05, 0.50, 0.95, 0.99])
        unique_months = int(part[TARGET_MONTH_COLUMN].nunique())
        rows.append(
            {
                "split": split,
                "first_month": part[TARGET_MONTH_COLUMN].min().date().isoformat(),
                "last_month": part[TARGET_MONTH_COLUMN].max().date().isoformat(),
                "rows": int(len(part)),
                "unique_permnos": int(part["permno"].nunique()),
                "unique_months": unique_months,
                "average_stocks_per_month": round(len(part) / unique_months, 4),
                "target_ret_1m_mean": target.mean(),
                "target_ret_1m_std": target.std(),
                "target_ret_1m_q01": quantiles.loc[0.01],
                "target_ret_1m_q05": quantiles.loc[0.05],
                "target_ret_1m_q50": quantiles.loc[0.50],
                "target_ret_1m_q95": quantiles.loc[0.95],
                "target_ret_1m_q99": quantiles.loc[0.99],
            }
        )
    return pd.DataFrame(rows)


def build_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLIT_ORDER:
        part = df.loc[df["split"].eq(split)]
        denominator = len(part)
        for label_column in TARGET_LABEL_COLUMNS:
            counts = part[label_column].value_counts(dropna=False).sort_index()
            for label_value, count in counts.items():
                rows.append(
                    {
                        "split": split,
                        "label_column": label_column,
                        "label_value": label_value,
                        "rows": int(count),
                        "percentage": round(100 * count / denominator, 6),
                    }
                )
    return pd.DataFrame(rows)


def build_feature_coverage(
    input_path: Path,
    feature_columns: list[str],
    group_lookup: dict[str, str],
    split_denominators: dict[str, int],
    batch_size: int,
) -> pd.DataFrame:
    nonmissing_counts = {
        feature: {split: 0 for split in SPLIT_ORDER} for feature in feature_columns
    }
    parquet = pq.ParquetFile(input_path)
    columns = ["mthcaldt"] + feature_columns
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        part = batch.to_pandas()
        part["mthcaldt"] = pd.to_datetime(part["mthcaldt"], errors="coerce")
        part[TARGET_MONTH_COLUMN] = next_calendar_month_end(part["mthcaldt"])
        part = part.loc[part[TARGET_MONTH_COLUMN].ge(SAMPLE_START)].copy()
        if part.empty:
            continue
        part["_split"] = split_values(part[TARGET_MONTH_COLUMN])
        part = part.loc[part["_split"].notna()]
        if part.empty:
            continue
        for split in SPLIT_ORDER:
            split_part = part.loc[part["_split"].eq(split), feature_columns]
            if split_part.empty:
                continue
            counts = split_part.notna().sum()
            for feature, count in counts.items():
                nonmissing_counts[feature][split] += int(count)

    rows = []
    for feature in feature_columns:
        row = {
            "feature": feature,
            "feature_group": group_lookup.get(feature, "unknown"),
        }
        for split in SPLIT_ORDER:
            denominator = split_denominators[split]
            row[f"nonmissing_pct_{split}"] = round(
                100 * nonmissing_counts[feature][split] / denominator, 6
            )
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_warnings(feature_coverage: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    train_missing = feature_coverage.loc[
        feature_coverage["nonmissing_pct_train"].eq(0), "feature"
    ].tolist()
    if train_missing:
        raise ValueError(f"Selected model features are entirely missing in train: {train_missing}")
    for split in ["validation", "test"]:
        column = f"nonmissing_pct_{split}"
        missing = feature_coverage.loc[feature_coverage[column].eq(0), "feature"].tolist()
        for feature in missing:
            warnings.append(f"Selected model feature is entirely missing in {split}: {feature}")
    return warnings


def build_stocks_per_month(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["split", TARGET_MONTH_COLUMN], observed=True)["permno"]
        .nunique()
        .rename("unique_permnos")
        .reset_index()
    )
    counts["month"] = counts[TARGET_MONTH_COLUMN].dt.date.astype(str)
    return counts[["split", "month", "unique_permnos"]]


def save_stocks_per_month_plot(stocks_per_month: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df = stocks_per_month.copy()
    plot_df["month"] = pd.to_datetime(plot_df["month"])
    for split in SPLIT_ORDER:
        part = plot_df.loc[plot_df["split"].eq(split)]
        ax.plot(
            part["month"],
            part["unique_permnos"],
            label=split,
            color=PLOT_COLORS[split],
            linewidth=1.6,
        )
    ax.set_title("Stocks per Month by Split")
    ax.set_xlabel("Month")
    ax.set_ylabel("Unique PERMNOs")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_target_distribution_plot(df: pd.DataFrame, path: Path) -> None:
    lower, upper = df[TARGET_COLUMN].quantile([0.01, 0.99])
    bins = np.linspace(lower, upper, 80)
    fig, ax = plt.subplots(figsize=(10, 5))
    for split in SPLIT_ORDER:
        values = df.loc[df["split"].eq(split), TARGET_COLUMN].clip(lower, upper)
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=PLOT_COLORS[split],
            label=split,
        )
    ax.set_title("Target Return Distribution by Split")
    ax.set_xlabel("target_ret_1m, clipped at global 1st/99th percentiles")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_target_volatility_by_year_plot(df: pd.DataFrame, path: Path) -> None:
    annual = (
        df.assign(year=df[TARGET_MONTH_COLUMN].dt.year)
        .groupby(["split", "year"], observed=True)[TARGET_COLUMN]
        .std()
        .rename("target_ret_1m_std")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    for split in SPLIT_ORDER:
        part = annual.loc[annual["split"].eq(split)]
        ax.plot(
            part["year"],
            part["target_ret_1m_std"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=PLOT_COLORS[split],
            label=split,
        )
    ax.set_title("Target Return Volatility by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cross-sectional target_ret_1m standard deviation")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_split_panel(input_path: Path, output_path: Path, batch_size: int) -> int:
    parquet = pq.ParquetFile(input_path)
    writer: pq.ParquetWriter | None = None
    output_rows = 0
    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            part = batch.to_pandas()
            part["mthcaldt"] = pd.to_datetime(part["mthcaldt"], errors="coerce")
            part[TARGET_MONTH_COLUMN] = next_calendar_month_end(part["mthcaldt"])
            part = part.loc[part[TARGET_MONTH_COLUMN].ge(SAMPLE_START)].copy()
            if part.empty:
                continue
            part["split"] = split_values(part[TARGET_MONTH_COLUMN])
            part = part.loc[part["split"].notna()].copy()
            if part.empty:
                continue
            part["split"] = pd.Categorical(part["split"], categories=SPLIT_ORDER, ordered=True)
            table = pa_table_from_pandas(part)
            if writer is None:
                writer = pq.ParquetWriter(temp_path, table.schema)
            writer.write_table(table)
            output_rows += len(part)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("No rows were written to the split output panel.")
    temp_path.replace(output_path)
    return output_rows


def pa_table_from_pandas(df: pd.DataFrame):
    import pyarrow as pa

    return pa.Table.from_pandas(df, preserve_index=False)


def write_metadata(
    path: Path,
    summary: pd.DataFrame,
    groups: dict[str, list[str]],
    feature_columns: list[str],
    warnings: list[str],
) -> None:
    feature_group_counts = {
        "raw_return": len(groups.get("raw_return_columns", [])),
        "return_feature": len(groups.get("return_feature_columns", [])),
        "jkp_feature": len(groups.get("jkp_feature_columns", [])),
        "selected_compustat_feature": len(groups.get("selected_compustat_feature_columns", [])),
        "selected_compustat_missing_indicator": len(
            groups.get("selected_compustat_missing_indicator_columns", [])
        ),
        "selected_model_features_total": len(feature_columns),
    }
    metadata = {
        "sample_start_date": SAMPLE_START.date().isoformat(),
        "split_date_column": TARGET_MONTH_COLUMN,
        "train_start": TRAIN_START.date().isoformat(),
        "train_end": TRAIN_END.date().isoformat(),
        "validation_start": VALIDATION_START.date().isoformat(),
        "validation_end": VALIDATION_END.date().isoformat(),
        "test_start": TEST_START.date().isoformat(),
        "test_end": TEST_END.date().isoformat(),
        "row_counts_by_split": {
            str(row["split"]): int(row["rows"]) for row in summary.to_dict(orient="records")
        },
        "feature_group_counts": feature_group_counts,
        "warnings": warnings,
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading panel schema: {args.input}", flush=True)
    columns = parquet_columns(args.input)
    assert_required_columns(columns)

    groups = load_feature_groups(args.feature_groups)
    feature_columns = selected_model_features(groups, columns)
    group_lookup = feature_group_lookup(groups)

    print("Loading key, target, and label columns for split diagnostics...", flush=True)
    diagnostics_input = pd.read_parquet(
        args.input, columns=KEY_COLUMNS + [TARGET_MONTH_COLUMN, TARGET_COLUMN] + TARGET_LABEL_COLUMNS
    )
    assert_unique_permno_month(diagnostics_input)

    input_rows = len(diagnostics_input)
    split_df = add_split(diagnostics_input)
    dropped_rows = input_rows - len(split_df)
    warnings = validate_splits(split_df, [])
    if dropped_rows:
        warnings.append(
            f"Dropped {dropped_rows:,} rows before {SAMPLE_START.date().isoformat()} "
            f"or outside the split date ranges."
        )

    split_summary = build_split_summary(split_df)
    label_distribution = build_label_distribution(split_df)
    split_denominators = split_summary.set_index("split")["rows"].to_dict()
    print("Computing feature coverage by split in batches...", flush=True)
    feature_coverage = build_feature_coverage(
        args.input,
        feature_columns,
        group_lookup,
        split_denominators,
        args.batch_size,
    )
    warnings.extend(coverage_warnings(feature_coverage))
    stocks_per_month = build_stocks_per_month(split_df)

    split_summary_path = args.table_dir / "split_summary.csv"
    label_distribution_path = args.table_dir / "target_label_distribution_by_split.csv"
    feature_coverage_path = args.table_dir / "feature_coverage_by_split.csv"
    stocks_per_month_path = args.table_dir / "stocks_per_month_by_split.csv"
    metadata_path = args.table_dir / "split_metadata.json"

    split_summary.to_csv(split_summary_path, index=False)
    label_distribution.to_csv(label_distribution_path, index=False)
    feature_coverage.to_csv(feature_coverage_path, index=False)
    stocks_per_month.to_csv(stocks_per_month_path, index=False)
    write_metadata(metadata_path, split_summary, groups, feature_columns, warnings)

    save_stocks_per_month_plot(
        stocks_per_month, args.plot_dir / "stocks_per_month_by_split.png"
    )
    save_target_distribution_plot(
        split_df, args.plot_dir / "target_distribution_by_split.png"
    )
    save_target_volatility_by_year_plot(
        split_df, args.plot_dir / "target_volatility_by_year.png"
    )

    print(f"Writing split panel in batches: {args.output}", flush=True)
    output_rows = write_split_panel(args.input, args.output, args.batch_size)
    if output_rows != len(split_df):
        raise ValueError(f"Split panel row count mismatch: {output_rows:,} != {len(split_df):,}")

    print(f"Input rows: {input_rows:,}", flush=True)
    print(f"Split output rows: {len(split_df):,}", flush=True)
    print(f"Selected model features checked: {len(feature_columns):,}", flush=True)
    if warnings:
        print("Warnings:", flush=True)
        for warning in warnings:
            print(f"- {warning}", flush=True)
    print(f"Wrote split summary: {split_summary_path}", flush=True)
    print(f"Wrote label distributions: {label_distribution_path}", flush=True)
    print(f"Wrote feature coverage: {feature_coverage_path}", flush=True)
    print(f"Wrote stocks per month: {stocks_per_month_path}", flush=True)
    print(f"Wrote split metadata: {metadata_path}", flush=True)
    print(f"Wrote split panel: {args.output}", flush=True)
    print(f"Wrote plots to: {args.plot_dir}", flush=True)


if __name__ == "__main__":
    main()
