#!/usr/bin/env python3
"""Merge lagged JKP factor-state features onto the monthly modeling panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT = Path("Dataset/Processed/model_panel_with_return_features.parquet")
DEFAULT_OUTPUT = Path("Dataset/Processed/model_panel_with_return_jkp_features.parquet")
DEFAULT_TABLE_DIR = Path("outputs/sanity_checks/tables")
DEFAULT_PLOT_DIR = Path("outputs/sanity_checks/plots")

KEY_COLUMNS = ["permno", "mthcaldt"]
TARGET_COLUMN = "target_ret_1m"
JKP_REQUIRED_COLUMNS = ["date", "name", "ret"]
JKP_PREFERRED_PATH = Path("Dataset/Parquet/Predictors/[usa]_[all_factors]_[monthly]_[vw_cap].parquet")
JKP_SEARCH_ROOTS = [
    Path("Dataset/Processed"),
    Path("Dataset/Parquet/Predictors"),
    Path("Dataset/Predictors"),
]

SELECTED_FACTORS = {
    "market_equity": ("size", "Market equity size factor."),
    "be_me": ("value", "Book-to-market value factor."),
    "ret_12_1": ("momentum", "12-1 prior-return momentum factor."),
    "ret_6_1": ("momentum", "6-1 prior-return momentum factor."),
    "ret_1_0": ("reversal", "Short-term prior-return reversal factor."),
    "gp_at": ("profitability", "Gross-profitability-to-assets factor."),
    "op_at": ("profitability", "Operating-profitability-to-assets factor."),
    "inv_gr1a": ("investment", "Asset-investment growth factor."),
    "oaccruals_at": ("accruals", "Operating-accruals-to-assets factor."),
    "netis_at": ("issuance", "Net-issuance-to-assets factor."),
    "div12m_me": ("payout", "Dividend-yield factor."),
    "beta_60m": ("risk", "Five-year market-beta factor."),
    "ivol_capm_252d": ("risk", "CAPM idiosyncratic-volatility factor."),
    "rvol_21d": ("trading", "Recent realized-volatility factor."),
    "ami_126d": ("liquidity", "Amihud illiquidity factor."),
    "zero_trades_126d": ("liquidity", "Zero-trading-days liquidity factor."),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add lagged JKP factor-state features to the monthly modeling panel."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jkp-input", type=Path, default=None)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args()


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def locate_jkp_file(explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Specified JKP file does not exist: {explicit_path}")
        columns = set(parquet_columns(explicit_path))
        missing = set(JKP_REQUIRED_COLUMNS).difference(columns)
        if missing:
            raise ValueError(f"Specified JKP file is missing required columns: {sorted(missing)}")
        return explicit_path

    if JKP_PREFERRED_PATH.exists():
        columns = set(parquet_columns(JKP_PREFERRED_PATH))
        if set(JKP_REQUIRED_COLUMNS).issubset(columns):
            return JKP_PREFERRED_PATH

    candidates: list[Path] = []
    for root in JKP_SEARCH_ROOTS:
        if root.exists():
            candidates.extend(sorted(root.rglob("*.parquet")))

    matching = []
    for path in candidates:
        try:
            columns = set(parquet_columns(path))
        except Exception:
            continue
        if set(JKP_REQUIRED_COLUMNS).issubset(columns):
            score = 0
            path_text = str(path).lower()
            if "all_factors" in path_text:
                score += 10
            if "monthly" in path_text:
                score += 5
            if "vw_cap" in path_text:
                score += 3
            if {"location", "freq", "weighting"}.issubset(columns):
                score += 2
            matching.append((score, path))

    if not matching:
        raise FileNotFoundError(
            "Could not locate a parquet JKP factor file with date/name/ret columns "
            f"under: {[str(root) for root in JKP_SEARCH_ROOTS]}"
        )
    return sorted(matching, key=lambda item: (-item[0], str(item[1])))[0][1]


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
        raise ValueError(f"{TARGET_COLUMN} cannot be a JKP feature.")
    if TARGET_COLUMN in source_columns:
        raise ValueError(
            f"{TARGET_COLUMN} was passed into JKP feature construction. "
            "JKP factor states must be built from JKP factor returns only."
        )


def month_id(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    return dates.dt.year * 12 + dates.dt.month


def log1p_returns(returns: pd.DataFrame) -> pd.DataFrame:
    numeric = returns.apply(pd.to_numeric, errors="coerce")
    valid = numeric.ge(-1.0)
    return np.log1p(numeric.where(valid))


def schema_summary(jkp: pd.DataFrame, path: Path) -> pd.DataFrame:
    rows = []
    for column in jkp.columns:
        values = jkp[column]
        examples = values.dropna().astype("string").drop_duplicates().head(5).tolist()
        rows.append(
            {
                "source_file": str(path),
                "rows": len(jkp),
                "column": column,
                "dtype": str(values.dtype),
                "nonmissing": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "unique_values": int(values.nunique(dropna=True)),
                "example_values": "; ".join(examples),
            }
        )
    return pd.DataFrame(rows)


def normalize_jkp_dates_to_panel(jkp: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    out = jkp.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        bad_rows = int(out["date"].isna().sum())
        raise ValueError(f"JKP data has {bad_rows:,} rows with invalid dates.")

    panel_months = panel[["mthcaldt"]].drop_duplicates().copy()
    panel_months["mthcaldt"] = pd.to_datetime(panel_months["mthcaldt"], errors="coerce")
    panel_months["_month_period"] = panel_months["mthcaldt"].dt.to_period("M")
    if panel_months["_month_period"].duplicated().any():
        duplicates = panel_months.loc[panel_months["_month_period"].duplicated(keep=False)]
        raise ValueError(
            "Panel has more than one MthCalDt within a calendar month. "
            f"First duplicate months:\n{duplicates.head(20).to_string(index=False)}"
        )

    out["_jkp_raw_date"] = out["date"]
    out["_month_period"] = out["date"].dt.to_period("M")
    out = out.merge(panel_months, on="_month_period", how="left", validate="many_to_one")
    return out


def select_factors(jkp: pd.DataFrame) -> list[str]:
    available = set(jkp["name"].dropna().astype(str).unique())
    selected = [factor for factor in SELECTED_FACTORS if factor in available]
    missing = sorted(set(SELECTED_FACTORS).difference(available))
    if missing:
        print(f"Selected-factor names not found in JKP data and skipped: {missing}", flush=True)
    if not selected:
        raise ValueError("None of the documented selected JKP factors were found.")
    return selected


def pivot_jkp_returns(jkp: pd.DataFrame, selected_factors: list[str]) -> pd.DataFrame:
    selected = jkp.loc[jkp["name"].isin(selected_factors) & jkp["mthcaldt"].notna()].copy()
    duplicates = selected.duplicated(["mthcaldt", "name"], keep=False)
    if duplicates.any():
        sample = selected.loc[duplicates, ["mthcaldt", "name", "ret"]].head(20)
        raise ValueError(
            "JKP data has duplicate normalized month/factor rows. "
            f"First duplicates:\n{sample.to_string(index=False)}"
        )

    wide = selected.pivot(index="mthcaldt", columns="name", values="ret").sort_index()
    wide.index.name = "mthcaldt"
    wide = wide.reindex(columns=selected_factors)
    return wide


def build_jkp_feature_frame(wide_returns: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    assert_no_target_leakage([], list(wide_returns.columns))
    raw = wide_returns.apply(pd.to_numeric, errors="coerce").sort_index()
    month_ids = pd.Series(month_id(pd.Series(raw.index, index=raw.index)), index=raw.index)

    lag1 = raw.shift(1)
    valid_1m = month_ids.shift(1).eq(month_ids - 1)
    lag1 = lag1.where(valid_1m, np.nan)

    log_lag1 = log1p_returns(raw.shift(1))
    mom12 = np.expm1(log_lag1.rolling(12, min_periods=12).sum())
    valid_12m = month_ids.shift(1).eq(month_ids - 1) & month_ids.shift(12).eq(month_ids - 12)
    mom12 = mom12.where(valid_12m, np.nan)

    vol12 = raw.shift(1).rolling(12, min_periods=12).std()
    vol12 = vol12.where(valid_12m, np.nan)

    features = pd.DataFrame(index=raw.index)
    feature_columns = []
    for factor in raw.columns:
        for suffix, frame in [("lag1", lag1), ("mom12", mom12), ("vol12", vol12)]:
            column = f"jkp_{factor}_{suffix}"
            features[column] = frame[factor]
            feature_columns.append(column)

    assert_no_target_leakage(feature_columns, list(raw.columns))
    return features.reset_index(), feature_columns


def selected_feature_table(selected_factors: list[str]) -> pd.DataFrame:
    rows = []
    for factor in selected_factors:
        category, rationale = SELECTED_FACTORS[factor]
        rows.append(
            {
                "factor_name": factor,
                "economic_category": category,
                "selection_approach": "Option A: documented interpretable subset",
                "rationale": rationale,
                "lag1_feature": f"jkp_{factor}_lag1",
                "mom12_feature": f"jkp_{factor}_mom12",
                "vol12_feature": f"jkp_{factor}_vol12",
            }
        )
    return pd.DataFrame(rows)


def feature_missingness(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for column in feature_columns:
        missing = int(df[column].isna().sum())
        rows.append(
            {
                "feature": column,
                "rows": total,
                "nonmissing": total - missing,
                "missing": missing,
                "missing_pct": round(100 * missing / total, 4) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def panel_month_counts(panel_keys: pd.DataFrame) -> pd.DataFrame:
    return (
        panel_keys.assign(month=pd.to_datetime(panel_keys["mthcaldt"]).dt.to_period("M"))
        .groupby(["month", "mthcaldt"], sort=True)
        .size()
        .reset_index(name="panel_rows")
    )


def feature_missingness_from_months(
    panel_months: pd.DataFrame,
    jkp_features: pd.DataFrame,
    feature_columns: list[str],
    total_rows: int,
) -> pd.DataFrame:
    monthly = panel_months.merge(jkp_features, on="mthcaldt", how="left")
    rows = []
    for column in feature_columns:
        missing = int(monthly.loc[monthly[column].isna(), "panel_rows"].sum())
        rows.append(
            {
                "feature": column,
                "rows": total_rows,
                "nonmissing": total_rows - missing,
                "missing": missing,
                "missing_pct": round(100 * missing / total_rows, 4) if total_rows else 0.0,
            }
        )
    return pd.DataFrame(rows)


def date_coverage_table(
    panel_months: pd.DataFrame,
    normalized_jkp: pd.DataFrame,
    jkp_features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    raw_dates = (
        normalized_jkp.dropna(subset=["mthcaldt"])
        .groupby("mthcaldt", sort=True)["_jkp_raw_date"]
        .min()
        .reset_index(name="jkp_raw_date")
    )
    factor_counts = (
        normalized_jkp.dropna(subset=["mthcaldt"])
        .groupby("mthcaldt", sort=True)["name"]
        .nunique()
        .reset_index(name="raw_jkp_factor_count")
    )
    monthly = panel_months.merge(raw_dates, on="mthcaldt", how="left")
    monthly = monthly.merge(factor_counts, on="mthcaldt", how="left")
    monthly["has_jkp_month"] = monthly["jkp_raw_date"].notna()

    feature_coverage = jkp_features[["mthcaldt"]].copy()
    feature_coverage["jkp_feature_coverage_pct"] = (
        jkp_features[feature_columns].notna().mean(axis=1) * 100
    )
    monthly = monthly.merge(feature_coverage, on="mthcaldt", how="left")
    monthly["panel_month_match_rate_pct"] = round(100 * monthly["has_jkp_month"].mean(), 4)
    monthly["month"] = monthly["month"].astype(str)
    return monthly


def write_panel_with_jkp_features(
    input_path: Path,
    output_path: Path,
    jkp_features: pd.DataFrame,
    feature_columns: list[str],
    batch_size: int,
) -> int:
    feature_lookup = jkp_features.set_index("mthcaldt")[feature_columns].sort_index()
    feature_lookup.index = pd.to_datetime(feature_lookup.index)

    input_file = pq.ParquetFile(input_path)
    temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if temp_path.exists():
        temp_path.unlink()

    writer: pq.ParquetWriter | None = None
    rows_written = 0
    try:
        for batch in input_file.iter_batches(batch_size=batch_size):
            frame = batch.to_pandas()
            dates = pd.DatetimeIndex(pd.to_datetime(frame["mthcaldt"], errors="coerce"))
            if dates.isna().any():
                bad_rows = int(dates.isna().sum())
                raise ValueError(f"Encountered {bad_rows:,} invalid panel dates while writing output.")
            mapped = feature_lookup.reindex(dates).reset_index(drop=True)
            for column in feature_columns:
                frame[column] = mapped[column].to_numpy()

            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp_path, table.schema)
            writer.write_table(table)
            rows_written += len(frame)
    finally:
        if writer is not None:
            writer.close()

    temp_path.replace(output_path)
    return rows_written


def save_coverage_plot(coverage: pd.DataFrame, plot_dir: Path) -> Path:
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_data = coverage.copy()
    plot_data["mthcaldt"] = pd.to_datetime(plot_data["mthcaldt"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        plot_data["mthcaldt"],
        plot_data["jkp_feature_coverage_pct"],
        color="#1f77b4",
        linewidth=1.4,
        label="JKP feature coverage",
    )
    ax.set_title("JKP Feature Coverage Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Non-missing JKP feature cells (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    path = plot_dir / "jkp_feature_coverage_over_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing input panel: {args.input}")

    jkp_path = locate_jkp_file(args.jkp_input)
    print(f"Using JKP factor file: {jkp_path}", flush=True)

    input_columns = parquet_columns(args.input)
    required_panel_columns = set(KEY_COLUMNS + [TARGET_COLUMN])
    missing_panel_columns = required_panel_columns.difference(input_columns)
    if missing_panel_columns:
        raise ValueError(f"Input panel is missing required columns: {sorted(missing_panel_columns)}")

    print(f"Reading panel keys for validation: {args.input}", flush=True)
    panel_keys = pd.read_parquet(args.input, columns=KEY_COLUMNS + [TARGET_COLUMN])
    panel_keys["mthcaldt"] = pd.to_datetime(panel_keys["mthcaldt"], errors="coerce")
    assert_unique_permno_month(panel_keys)
    panel_months = panel_month_counts(panel_keys)
    print(f"Input rows: {len(panel_keys):,}", flush=True)
    print("Duplicate PERMNO-MthCalDt rows in input: 0", flush=True)

    print("Reading and reshaping JKP factor returns...", flush=True)
    jkp = pd.read_parquet(jkp_path)
    missing = set(JKP_REQUIRED_COLUMNS).difference(jkp.columns)
    if missing:
        raise ValueError(f"JKP file is missing required columns: {sorted(missing)}")
    raw_factor_count = int(jkp["name"].nunique(dropna=True))
    normalized_jkp = normalize_jkp_dates_to_panel(jkp, panel_keys)

    selected_factors = select_factors(normalized_jkp)
    wide_returns = pivot_jkp_returns(normalized_jkp, selected_factors)
    jkp_features, feature_columns = build_jkp_feature_frame(wide_returns)

    print(
        f"Raw JKP factors: {raw_factor_count:,}; selected factors: {len(selected_factors):,}; "
        f"created features: {len(feature_columns):,}",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("Writing output panel in parquet batches...", flush=True)
    rows_written = write_panel_with_jkp_features(
        args.input, args.output, jkp_features, feature_columns, args.batch_size
    )
    if rows_written != len(panel_keys):
        raise ValueError(f"Row count changed after JKP merge: {len(panel_keys):,} -> {rows_written:,}")

    output_keys = pd.read_parquet(args.output, columns=KEY_COLUMNS)
    duplicate_after = int(output_keys.duplicated(KEY_COLUMNS).sum())
    if duplicate_after:
        raise ValueError(f"Output has {duplicate_after:,} duplicate PERMNO-MthCalDt rows.")

    args.table_dir.mkdir(parents=True, exist_ok=True)
    schema_path = args.table_dir / "jkp_schema_summary.csv"
    selected_path = args.table_dir / "selected_jkp_features.csv"
    missingness_path = args.table_dir / "jkp_feature_missingness.csv"
    coverage_path = args.table_dir / "jkp_date_coverage.csv"

    schema_summary(jkp, jkp_path).to_csv(schema_path, index=False)
    selected_feature_table(selected_factors).to_csv(selected_path, index=False)
    feature_missingness_from_months(
        panel_months, jkp_features, feature_columns, len(panel_keys)
    ).to_csv(missingness_path, index=False)
    coverage = date_coverage_table(panel_months, normalized_jkp, jkp_features, feature_columns)
    coverage.to_csv(coverage_path, index=False)
    plot_path = save_coverage_plot(coverage, args.plot_dir)

    unmatched_months = coverage.loc[~coverage["has_jkp_month"], ["month", "mthcaldt"]]
    panel_match_rate = 100 * coverage["has_jkp_month"].mean()

    print(f"JKP raw date range: {jkp['date'].min()} to {jkp['date'].max()}", flush=True)
    print(
        f"Panel months with JKP coverage: {int(coverage['has_jkp_month'].sum()):,} / "
        f"{len(coverage):,} ({panel_match_rate:.4f}%)",
        flush=True,
    )
    print(f"Panel months without JKP coverage: {len(unmatched_months):,}", flush=True)
    if not unmatched_months.empty:
        print(unmatched_months.head(20).to_string(index=False), flush=True)
    print(f"Output rows: {rows_written:,}", flush=True)
    print(f"Duplicate PERMNO-MthCalDt rows in output: {duplicate_after:,}", flush=True)
    print(f"Wrote JKP feature panel: {args.output}", flush=True)
    print(f"Wrote JKP schema summary: {schema_path}", flush=True)
    print(f"Wrote selected JKP feature names: {selected_path}", flush=True)
    print(f"Wrote JKP feature missingness: {missingness_path}", flush=True)
    print(f"Wrote JKP date coverage: {coverage_path}", flush=True)
    print(f"Wrote JKP coverage plot: {plot_path}", flush=True)


if __name__ == "__main__":
    main()
