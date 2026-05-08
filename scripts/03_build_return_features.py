#!/usr/bin/env python3
"""Add lagged return predictors to the monthly modeling panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_INPUT = Path("Dataset/Processed/model_panel_monthly_base.parquet")
DEFAULT_OUTPUT = Path("Dataset/Processed/model_panel_with_return_features.parquet")
DEFAULT_TABLE_DIR = Path("outputs/sanity_checks/tables")
DEFAULT_PLOT_DIR = Path("outputs/sanity_checks/plots")

KEY_COLUMNS = ["permno", "mthcaldt"]
TARGET_COLUMN = "target_ret_1m"
FEATURE_INPUT_COLUMNS = ["permno", "mthcaldt", "mthret", "sprtrn"]

LAG_FEATURES = ["ret_lag_1m", "ret_lag_2m", "ret_lag_3m", "ret_lag_6m"]
ROLLING_FEATURES = [
    "momentum_6m",
    "momentum_12m_excl_1m",
    "volatility_12m",
    "volatility_24m",
]
MARKET_FEATURES = ["market_ret_lag_1m", "excess_ret_lag_1m"]
BETA_FEATURES = ["beta_24m"]
RETURN_FEATURES = LAG_FEATURES + ROLLING_FEATURES + MARKET_FEATURES + BETA_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build lagged return features for the monthly CRSP modeling panel."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    return parser.parse_args()


def available_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def assert_required_columns(columns: list[str]) -> None:
    missing = set(FEATURE_INPUT_COLUMNS + [TARGET_COLUMN]).difference(columns)
    if missing:
        raise ValueError(f"Input panel is missing required columns: {sorted(missing)}")


def assert_unique_permno_month(df: pd.DataFrame) -> None:
    duplicates = df.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        sample = df.loc[duplicates, KEY_COLUMNS].head(20)
        raise ValueError(
            "Panel has duplicate PERMNO-MthCalDt rows. "
            f"First duplicate keys:\n{sample.to_string(index=False)}"
        )


def assert_no_target_leakage(feature_columns: list[str], source_columns: list[str]) -> None:
    if TARGET_COLUMN in feature_columns:
        raise ValueError(f"{TARGET_COLUMN} cannot be a return feature.")
    if TARGET_COLUMN in source_columns:
        raise ValueError(
            f"{TARGET_COLUMN} was passed into the return-feature builder. "
            "Return features must be built only from contemporaneous and lagged returns."
        )


def add_month_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mthcaldt"] = pd.to_datetime(out["mthcaldt"], errors="coerce")
    if out["mthcaldt"].isna().any():
        bad_rows = int(out["mthcaldt"].isna().sum())
        raise ValueError(f"Cannot build return features because {bad_rows:,} rows have invalid dates.")
    out["_month_id"] = out["mthcaldt"].dt.year * 12 + out["mthcaldt"].dt.month
    return out


def rolling_sum_by_permno(values: pd.Series, permno: pd.Series, window: int) -> pd.Series:
    rolled = values.groupby(permno, sort=False).rolling(window, min_periods=window).sum()
    return rolled.reset_index(level=0, drop=True).sort_index()


def rolling_std_by_permno(values: pd.Series, permno: pd.Series, window: int) -> pd.Series:
    rolled = values.groupby(permno, sort=False).rolling(window, min_periods=window).std()
    return rolled.reset_index(level=0, drop=True).sort_index()


def log1p_returns(returns: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(returns, errors="coerce")
    result = pd.Series(np.nan, index=returns.index, dtype="float64")
    valid = numeric.ge(-1.0) & numeric.notna()
    with np.errstate(divide="ignore", invalid="ignore"):
        result.loc[valid] = np.log1p(numeric.loc[valid])
    return result


def market_return_by_month(df: pd.DataFrame) -> pd.Series:
    nonmissing_unique_counts = df.groupby("_month_id")["sprtrn"].nunique(dropna=True)
    conflicting = nonmissing_unique_counts[nonmissing_unique_counts > 1]
    if not conflicting.empty:
        sample = conflicting.head(10).rename("unique_sprtrn_values").reset_index()
        raise ValueError(
            "sprtrn is not unique within month, so market_ret_lag_1m cannot be built cleanly. "
            f"First conflicting months:\n{sample.to_string(index=False)}"
        )
    return df.groupby("_month_id")["sprtrn"].first()


def build_return_features(feature_input: pd.DataFrame) -> pd.DataFrame:
    assert_no_target_leakage(RETURN_FEATURES, list(feature_input.columns))
    missing = set(FEATURE_INPUT_COLUMNS).difference(feature_input.columns)
    if missing:
        raise ValueError(f"Feature input is missing columns: {sorted(missing)}")

    out = add_month_id(feature_input[FEATURE_INPUT_COLUMNS])
    out["mthret"] = pd.to_numeric(out["mthret"], errors="coerce")
    out["sprtrn"] = pd.to_numeric(out["sprtrn"], errors="coerce")
    out = out.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)

    grouped = out.groupby("permno", sort=False)

    for lag in [1, 2, 3, 6]:
        feature = f"ret_lag_{lag}m"
        lagged_month_id = grouped["_month_id"].shift(lag)
        out[feature] = grouped["mthret"].shift(lag)
        valid = lagged_month_id.eq(out["_month_id"] - lag)
        out.loc[~valid, feature] = np.nan

    shifted_ret_1m = grouped["mthret"].shift(1)
    shifted_market_1m_by_permno = grouped["sprtrn"].shift(1)
    shifted_month_1m = grouped["_month_id"].shift(1)
    shifted_month_2m = grouped["_month_id"].shift(2)
    shifted_month_6m = grouped["_month_id"].shift(6)
    shifted_month_12m = grouped["_month_id"].shift(12)
    shifted_month_24m = grouped["_month_id"].shift(24)

    log_ret_lag_1m = log1p_returns(shifted_ret_1m)
    momentum_6m = np.expm1(rolling_sum_by_permno(log_ret_lag_1m, out["permno"], 6))
    valid_6m = shifted_month_1m.eq(out["_month_id"] - 1) & shifted_month_6m.eq(
        out["_month_id"] - 6
    )
    out["momentum_6m"] = momentum_6m.where(valid_6m)

    shifted_ret_2m = grouped["mthret"].shift(2)
    log_ret_lag_2m = log1p_returns(shifted_ret_2m)
    momentum_12m_excl_1m = np.expm1(
        rolling_sum_by_permno(log_ret_lag_2m, out["permno"], 11)
    )
    valid_12m_excl_1m = shifted_month_2m.eq(out["_month_id"] - 2) & shifted_month_12m.eq(
        out["_month_id"] - 12
    )
    out["momentum_12m_excl_1m"] = momentum_12m_excl_1m.where(valid_12m_excl_1m)

    volatility_12m = rolling_std_by_permno(shifted_ret_1m, out["permno"], 12)
    valid_12m = shifted_month_1m.eq(out["_month_id"] - 1) & shifted_month_12m.eq(
        out["_month_id"] - 12
    )
    out["volatility_12m"] = volatility_12m.where(valid_12m)

    volatility_24m = rolling_std_by_permno(shifted_ret_1m, out["permno"], 24)
    valid_24m = shifted_month_1m.eq(out["_month_id"] - 1) & shifted_month_24m.eq(
        out["_month_id"] - 24
    )
    out["volatility_24m"] = volatility_24m.where(valid_24m)

    monthly_market = market_return_by_month(out)
    out["market_ret_lag_1m"] = out["_month_id"].sub(1).map(monthly_market)
    out["excess_ret_lag_1m"] = out["ret_lag_1m"] - out["market_ret_lag_1m"]

    stock_ret = shifted_ret_1m
    market_ret = shifted_market_1m_by_permno
    sum_x = rolling_sum_by_permno(stock_ret, out["permno"], 24)
    sum_y = rolling_sum_by_permno(market_ret, out["permno"], 24)
    sum_xy = rolling_sum_by_permno(stock_ret * market_ret, out["permno"], 24)
    sum_y2 = rolling_sum_by_permno(market_ret * market_ret, out["permno"], 24)
    cov_num = sum_xy - (sum_x * sum_y / 24)
    var_num = sum_y2 - (sum_y * sum_y / 24)
    beta_24m = cov_num / var_num
    beta_24m = beta_24m.where(var_num.gt(1e-12))
    out["beta_24m"] = beta_24m.where(valid_24m)

    return out[KEY_COLUMNS + RETURN_FEATURES]


def build_missingness(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for feature in features:
        missing = int(df[feature].isna().sum())
        rows.append(
            {
                "feature": feature,
                "rows": total,
                "nonmissing": total - missing,
                "missing": missing,
                "missing_pct": round(100 * missing / total, 4) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_summary_stats(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce").dropna()
        if values.empty:
            rows.append({"feature": feature, "count": 0})
            continue
        quantiles = values.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
        rows.append(
            {
                "feature": feature,
                "count": int(values.count()),
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "p1": quantiles.loc[0.01],
                "p5": quantiles.loc[0.05],
                "median": quantiles.loc[0.5],
                "p95": quantiles.loc[0.95],
                "p99": quantiles.loc[0.99],
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def build_manual_checks(df: pd.DataFrame) -> pd.DataFrame:
    display_columns = [
        "permno",
        "mthcaldt",
        "mthret",
        TARGET_COLUMN,
        "ret_lag_1m",
        "momentum_6m",
        "momentum_12m_excl_1m",
    ]
    candidates = (
        df.loc[df["momentum_12m_excl_1m"].notna(), ["permno"]]
        .drop_duplicates()
        .head(3)["permno"]
        .tolist()
    )
    examples = []
    for permno in candidates:
        stock = df.loc[df["permno"].eq(permno), display_columns].sort_values("mthcaldt")
        first_valid_index = stock.index[stock["momentum_12m_excl_1m"].notna()][0]
        position = stock.index.get_loc(first_valid_index)
        start = max(0, position - 8)
        stop = min(len(stock), position + 5)
        examples.append(stock.iloc[start:stop])
    if not examples:
        return pd.DataFrame(columns=display_columns)
    return pd.concat(examples, ignore_index=True)


def build_monthly_coverage(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    working = df[["mthcaldt", "permno"] + features].copy()
    working["month"] = pd.to_datetime(working["mthcaldt"]).dt.to_period("M").dt.to_timestamp()
    grouped = working.groupby("month", sort=True)
    coverage = grouped.agg(rows=("permno", "size")).reset_index()
    for feature in features:
        nonmissing = grouped[feature].apply(lambda x: x.notna().sum()).reset_index(drop=True)
        coverage[f"{feature}_coverage_pct"] = 100 * nonmissing / coverage["rows"]
        coverage[f"{feature}_nonmissing"] = nonmissing
    return coverage


def save_coverage_plots(
    coverage: pd.DataFrame, features: list[str], plot_dir: Path
) -> list[Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    selected = [
        feature
        for feature in [
            "ret_lag_1m",
            "momentum_6m",
            "momentum_12m_excl_1m",
            "volatility_24m",
            "beta_24m",
        ]
        if feature in features
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    for feature in selected:
        ax.plot(coverage["month"], coverage[f"{feature}_coverage_pct"], label=feature, linewidth=1.2)
    ax.set_title("Return Feature Coverage Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Non-missing rows (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.25)
    path = plot_dir / "return_feature_coverage_over_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    coverage["average_return_feature_coverage_pct"] = coverage[
        [f"{feature}_coverage_pct" for feature in features]
    ].mean(axis=1)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(
        coverage["month"],
        coverage["average_return_feature_coverage_pct"],
        color="#1f77b4",
        linewidth=1.4,
        label="Average feature coverage",
    )
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Average non-missing feature rows (%)")
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(
        coverage["month"],
        coverage["rows"],
        color="#6b6b6b",
        linewidth=1.0,
        alpha=0.65,
        label="Panel rows",
    )
    ax2.set_ylabel("Panel rows")
    ax1.set_title("Average Return Feature Coverage and Panel Size")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right", fontsize=8)
    path = plot_dir / "return_feature_average_coverage_over_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    return paths


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing base modeling panel: {args.input}")

    columns = available_columns(args.input)
    assert_required_columns(columns)
    assert_no_target_leakage(RETURN_FEATURES, FEATURE_INPUT_COLUMNS)

    print(f"Reading base modeling panel: {args.input}", flush=True)
    panel = pd.read_parquet(args.input)
    panel["mthcaldt"] = pd.to_datetime(panel["mthcaldt"], errors="coerce")
    assert_unique_permno_month(panel)
    target_before = panel[TARGET_COLUMN].copy()
    print(f"Input rows: {len(panel):,}", flush=True)
    print("Duplicate PERMNO-MthCalDt rows in input: 0", flush=True)

    print("Building lagged return, momentum, volatility, market, and beta features...", flush=True)
    feature_input = panel[FEATURE_INPUT_COLUMNS].copy()
    features = build_return_features(feature_input)
    output = panel.merge(features, on=KEY_COLUMNS, how="left", validate="one_to_one")
    if not output[TARGET_COLUMN].equals(target_before):
        raise ValueError(f"{TARGET_COLUMN} changed while building return features.")

    duplicate_after = int(output.duplicated(KEY_COLUMNS).sum())
    if duplicate_after:
        raise ValueError(f"Output has {duplicate_after:,} duplicate PERMNO-MthCalDt rows.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)

    args.table_dir.mkdir(parents=True, exist_ok=True)
    missingness = build_missingness(output, RETURN_FEATURES)
    summary = build_summary_stats(output, RETURN_FEATURES)
    manual_checks = build_manual_checks(output)
    coverage = build_monthly_coverage(output, RETURN_FEATURES)

    missingness_path = args.table_dir / "return_feature_missingness.csv"
    summary_path = args.table_dir / "return_feature_summary.csv"
    manual_checks_path = args.table_dir / "return_feature_manual_checks.csv"
    coverage_path = args.table_dir / "return_feature_monthly_coverage.csv"

    missingness.to_csv(missingness_path, index=False)
    summary.to_csv(summary_path, index=False)
    manual_checks.to_csv(manual_checks_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    plot_paths = save_coverage_plots(coverage, RETURN_FEATURES, args.plot_dir)

    beta_created = "yes" if "beta_24m" in output.columns else "no"
    print(f"Output rows: {len(output):,}", flush=True)
    print(f"Generated return features: {len(RETURN_FEATURES):,}", flush=True)
    print(f"Duplicate PERMNO-MthCalDt rows in output: {duplicate_after:,}", flush=True)
    print(f"beta_24m created: {beta_created}", flush=True)
    print(f"Wrote return-feature panel: {args.output}", flush=True)
    print(f"Wrote missingness table: {missingness_path}", flush=True)
    print(f"Wrote summary table: {summary_path}", flush=True)
    print(f"Wrote manual validation examples: {manual_checks_path}", flush=True)
    print(f"Wrote monthly coverage table: {coverage_path}", flush=True)
    for path in plot_paths:
        print(f"Wrote coverage plot: {path}", flush=True)


if __name__ == "__main__":
    main()
