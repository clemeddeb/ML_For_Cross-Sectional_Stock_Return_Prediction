#!/usr/bin/env python3
"""Validate linked project datasets and write sanity-check outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PROCESSED_DIR = Path("Dataset/Processed")
DEFAULT_RAW_CRSP = Path("Dataset/Targets/monthly_crsp.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/sanity_checks")

RAW_CRSP_COLUMNS = [
    "PERMNO",
    "HdrCUSIP",
    "CUSIP",
    "Ticker",
    "TradingSymbol",
    "PERMCO",
    "SICCD",
    "NAICS",
    "MthCalDt",
    "MthRet",
    "sprtrn",
]

MONTHLY_DATASETS = {
    "crsp_with_gvkey": {
        "path": "crsp_with_gvkey.parquet",
        "columns": [
            "permno",
            "mthcaldt",
            "mthret",
            "sprtrn",
            "gvkey",
            "linktype",
            "linkprim",
        ],
    },
    "crsp_compustat_panel": {
        "path": "crsp_compustat_panel.parquet",
        "columns": [
            "permno",
            "mthcaldt",
            "mthret",
            "sprtrn",
            "gvkey",
            "linktype",
            "linkprim",
            "compustat_match",
            "compustat_age_days",
            "comp_datadate",
            "comp_available_date",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate linked datasets and write sanity-check tables/plots."
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--raw-crsp", type=Path, default=DEFAULT_RAW_CRSP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, plots_dir


def write_table(df: pd.DataFrame, tables_dir: Path, name: str) -> Path:
    path = tables_dir / name
    df.to_csv(path, index=False)
    return path


def save_plot(fig: plt.Figure, plots_dir: Path, name: str) -> Path:
    path = plots_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def month_end(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp("M")


def pct(numerator: int | float, denominator: int | float) -> float:
    return round(100 * numerator / denominator, 2) if denominator else 0.0


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.to_string(index=False)
    return f"```\n{text}\n```"


def load_link_summary(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "link_summary.json"
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    rows = []
    for dataset, values in summary.items():
        row = {"dataset": dataset}
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def duplicate_summary(df: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    working["month"] = month_end(working["mthcaldt"])
    date_counts = working.groupby(["permno", "mthcaldt"], dropna=False).size()
    month_counts = working.groupby(["permno", "month"], dropna=False).size()
    date_duplicates = date_counts[date_counts > 1]
    month_duplicates = month_counts[month_counts > 1]
    summary = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "rows": len(working),
                "permno_mthcaldt_keys": len(date_counts),
                "duplicate_permno_mthcaldt_keys": len(date_duplicates),
                "duplicate_permno_mthcaldt_rows": int(date_duplicates.sum()),
                "permno_month_keys": len(month_counts),
                "duplicate_permno_month_keys": len(month_duplicates),
                "duplicate_permno_month_rows": int(month_duplicates.sum()),
                "duplicate_extra_rows": int((month_duplicates - 1).sum()),
                "max_rows_per_permno_month": int(month_counts.max()) if len(month_counts) else 0,
            }
        ]
    )
    distribution = (
        month_duplicates.value_counts()
        .sort_index()
        .rename_axis("rows_per_permno_month")
        .reset_index(name="duplicate_key_count")
    )
    distribution.insert(0, "dataset", dataset)
    return summary, distribution


def coverage_by_year(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    working = df.copy()
    working["year"] = pd.to_datetime(working["mthcaldt"], errors="coerce").dt.year
    rows = []
    for year, group in working.groupby("year", dropna=True):
        row = {
            "dataset": dataset,
            "year": int(year),
            "rows": len(group),
            "unique_permnos": group["permno"].nunique(dropna=True),
        }
        if "gvkey" in group.columns:
            row["gvkey_match_pct"] = pct(group["gvkey"].notna().sum(), len(group))
        if "compustat_match" in group.columns:
            row["compustat_match_pct"] = pct(group["compustat_match"].fillna(False).sum(), len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def monthly_quality_summary(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    dates = pd.to_datetime(df["mthcaldt"], errors="coerce")
    row = {
        "dataset": dataset,
        "rows": len(df),
        "unique_permnos": df["permno"].nunique(dropna=True),
        "min_mthcaldt": dates.min(),
        "max_mthcaldt": dates.max(),
    }
    if "gvkey" in df.columns:
        row["gvkey_match_pct"] = pct(df["gvkey"].notna().sum(), len(df))
        row["gvkey_matched_rows"] = int(df["gvkey"].notna().sum())
    if "compustat_match" in df.columns:
        matched = df["compustat_match"].fillna(False)
        row["compustat_match_pct"] = pct(matched.sum(), len(df))
        row["compustat_matched_rows"] = int(matched.sum())
    return pd.DataFrame([row])


def compustat_lookahead_summary(panel: pd.DataFrame) -> pd.DataFrame:
    if "comp_available_date" not in panel.columns:
        return pd.DataFrame(
            [
                {
                    "dataset": "crsp_compustat_panel",
                    "lookahead_check_available": False,
                    "lookahead_violation_rows": 0,
                    "negative_age_rows": 0,
                }
            ]
        )

    matched = panel["compustat_match"].fillna(False)
    mthcaldt = pd.to_datetime(panel["mthcaldt"], errors="coerce")
    available = pd.to_datetime(panel["comp_available_date"], errors="coerce")
    ages = coerce_numeric(panel["compustat_age_days"])
    lookahead = matched & available.notna() & mthcaldt.notna() & available.gt(mthcaldt)
    negative_age = matched & ages.notna() & ages.lt(0)
    return pd.DataFrame(
        [
            {
                "dataset": "crsp_compustat_panel",
                "lookahead_check_available": True,
                "matched_rows_checked": int(matched.sum()),
                "lookahead_violation_rows": int(lookahead.sum()),
                "negative_age_rows": int(negative_age.sum()),
                "min_compustat_age_days": ages[matched].min(),
                "max_compustat_age_days": ages[matched].max(),
            }
        ]
    )


def return_summary(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    returns = coerce_numeric(df["mthret"]).dropna()
    if returns.empty:
        return pd.DataFrame([{"dataset": dataset, "nonmissing_returns": 0}])
    quantiles = returns.quantile([0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "nonmissing_returns": len(returns),
                "mean": returns.mean(),
                "std": returns.std(),
                "min": returns.min(),
                "p0_1": quantiles.loc[0.001],
                "p1": quantiles.loc[0.01],
                "p5": quantiles.loc[0.05],
                "median": quantiles.loc[0.5],
                "p95": quantiles.loc[0.95],
                "p99": quantiles.loc[0.99],
                "p99_9": quantiles.loc[0.999],
                "max": returns.max(),
            }
        ]
    )


def linktype_counts(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if "linktype" not in df.columns or "linkprim" not in df.columns:
        return pd.DataFrame()
    out = (
        df.groupby(["linktype", "linkprim"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["rows", "linktype", "linkprim"], ascending=[False, True, True])
    )
    out.insert(0, "dataset", dataset)
    return out


def validate_monthly(
    processed_dir: Path, tables_dir: Path, plots_dir: Path
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    duplicate_tables = []
    duplicate_distributions = []
    quality_tables = []
    coverage_tables = []
    return_tables = []
    link_tables = []

    for dataset, spec in MONTHLY_DATASETS.items():
        df = pd.read_parquet(processed_dir / spec["path"], columns=spec["columns"])
        loaded[dataset] = df

        summary, distribution = duplicate_summary(df, dataset)
        duplicate_tables.append(summary)
        duplicate_distributions.append(distribution)
        quality_tables.append(monthly_quality_summary(df, dataset))
        coverage_tables.append(coverage_by_year(df, dataset))
        return_tables.append(return_summary(df, dataset))
        link_tables.append(linktype_counts(df, dataset))

    duplicate_summary_df = pd.concat(duplicate_tables, ignore_index=True)
    duplicate_dist_df = pd.concat(duplicate_distributions, ignore_index=True)
    quality_df = pd.concat(quality_tables, ignore_index=True)
    coverage_df = pd.concat(coverage_tables, ignore_index=True)
    returns_df = pd.concat(return_tables, ignore_index=True)
    links_df = pd.concat(link_tables, ignore_index=True)
    lookahead_df = compustat_lookahead_summary(loaded["crsp_compustat_panel"])

    write_table(duplicate_summary_df, tables_dir, "monthly_duplicate_summary.csv")
    write_table(duplicate_dist_df, tables_dir, "monthly_duplicate_group_size_distribution.csv")
    write_table(quality_df, tables_dir, "monthly_quality_summary.csv")
    write_table(coverage_df, tables_dir, "monthly_coverage_by_year.csv")
    write_table(returns_df, tables_dir, "monthly_return_summary.csv")
    write_table(links_df, tables_dir, "monthly_linktype_counts.csv")
    write_table(lookahead_df, tables_dir, "compustat_lookahead_summary.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = coverage_df.pivot(index="year", columns="dataset", values="rows")
    pivot.plot(ax=ax)
    ax.set_title("Monthly linked rows by year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rows")
    save_plot(fig, plots_dir, "monthly_rows_by_year.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for dataset, df in loaded.items():
        returns = coerce_numeric(df["mthret"]).dropna().clip(-1, 1)
        ax.hist(returns, bins=120, alpha=0.45, label=dataset, density=True)
    ax.set_title("Monthly return distribution, clipped to [-1, 1]")
    ax.set_xlabel("Monthly return")
    ax.set_ylabel("Density")
    ax.legend()
    save_plot(fig, plots_dir, "monthly_return_distribution.png")

    panel = loaded["crsp_compustat_panel"]
    ages = coerce_numeric(panel["compustat_age_days"]).dropna()
    if not ages.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(ages, bins=80, color="#3d6fb6")
        ax.set_title("Compustat observation age in monthly panel")
        ax.set_xlabel("Age in days")
        ax.set_ylabel("Rows")
        save_plot(fig, plots_dir, "compustat_age_distribution.png")

    return loaded, duplicate_summary_df, quality_df, lookahead_df


def inspect_raw_crsp_duplicates(raw_crsp: Path, tables_dir: Path, plots_dir: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_crsp, usecols=RAW_CRSP_COLUMNS, dtype={"PERMNO": "Int64"})
    raw["month"] = month_end(raw["MthCalDt"])
    key = ["PERMNO", "month"]
    counts = raw.groupby(key, dropna=False).size()
    duplicate_keys = counts[counts > 1]
    duplicate_key_frame = duplicate_keys.reset_index(name="rows_per_permno_month")
    duplicate_rows = raw.merge(duplicate_key_frame[key], on=key, how="inner")
    row_columns = [c for c in RAW_CRSP_COLUMNS if c not in key]
    distinct_rows_per_key = (
        duplicate_rows.drop_duplicates(subset=RAW_CRSP_COLUMNS)
        .groupby(key, dropna=False)
        .size()
        .rename("distinct_raw_rows_per_key")
    )

    conflict_rows = []
    for column in [c for c in RAW_CRSP_COLUMNS if c not in ["PERMNO", "MthCalDt"]]:
        n_unique = duplicate_rows.groupby(key, dropna=False)[column].nunique(dropna=False)
        conflict_rows.append(
            {
                "column": column,
                "duplicate_keys_with_conflict": int((n_unique > 1).sum()),
            }
        )
    conflict_df = pd.DataFrame(conflict_rows)

    duplicate_key_details = duplicate_key_frame.merge(
        distinct_rows_per_key.reset_index(), on=key, how="left"
    )
    duplicate_key_details["year"] = duplicate_key_details["month"].dt.year
    duplicate_key_details = duplicate_key_details.sort_values(
        ["rows_per_permno_month", "PERMNO", "month"], ascending=[False, True, True]
    )

    metadata_columns = ["CUSIP", "Ticker", "TradingSymbol", "SICCD", "NAICS"]
    metadata_conflict_mask = pd.Series(False, index=duplicate_key_details.index)
    for column in metadata_columns:
        n_unique = duplicate_rows.groupby(key, dropna=False)[column].nunique(dropna=False)
        conflicted_keys = set(n_unique[n_unique > 1].index)
        metadata_conflict_mask = metadata_conflict_mask | duplicate_key_details.set_index(key).index.isin(conflicted_keys)

    placeholder = (
        duplicate_rows["CUSIP"].isna()
        & duplicate_rows["Ticker"].isna()
        & duplicate_rows["TradingSymbol"].isna()
        & duplicate_rows["SICCD"].fillna(0).eq(0)
        & duplicate_rows["NAICS"].fillna(0).eq(0)
    )
    non_placeholder = (
        duplicate_rows["CUSIP"].notna()
        | duplicate_rows["Ticker"].notna()
        | duplicate_rows["TradingSymbol"].notna()
        | ~duplicate_rows["SICCD"].fillna(0).eq(0)
        | ~duplicate_rows["NAICS"].fillna(0).eq(0)
    )
    placeholder_by_key = placeholder.groupby([duplicate_rows["PERMNO"], duplicate_rows["month"]], dropna=False).any()
    non_placeholder_by_key = non_placeholder.groupby(
        [duplicate_rows["PERMNO"], duplicate_rows["month"]], dropna=False
    ).any()
    both_placeholder = placeholder_by_key & non_placeholder_by_key

    raw_summary = pd.DataFrame(
        [
            {
                "raw_file": str(raw_crsp),
                "rows": len(raw),
                "permno_month_keys": len(counts),
                "duplicate_permno_month_keys": len(duplicate_keys),
                "duplicate_rows_in_duplicate_keys": int(duplicate_keys.sum()),
                "duplicate_extra_rows_by_key": int((duplicate_keys - 1).sum()),
                "unique_full_rows": len(raw.drop_duplicates(subset=RAW_CRSP_COLUMNS)),
                "duplicate_extra_rows_by_full_row": len(raw)
                - len(raw.drop_duplicates(subset=RAW_CRSP_COLUMNS)),
                "duplicate_keys_all_rows_exact_identical": int((distinct_rows_per_key == 1).sum()),
                "duplicate_keys_with_multiple_distinct_raw_rows": int((distinct_rows_per_key > 1).sum()),
                "duplicate_keys_with_placeholder_metadata_row": int(placeholder_by_key.sum()),
                "duplicate_keys_with_placeholder_and_nonplaceholder_metadata": int(both_placeholder.sum()),
            }
        ]
    )

    by_year = (
        duplicate_key_details.groupby("year", dropna=True)
        .agg(
            duplicate_permno_month_keys=("PERMNO", "size"),
            duplicate_rows_in_duplicate_keys=("rows_per_permno_month", "sum"),
        )
        .reset_index()
        .sort_values("year")
    )
    group_size_dist = (
        duplicate_keys.value_counts()
        .sort_index()
        .rename_axis("rows_per_permno_month")
        .reset_index(name="duplicate_key_count")
    )

    write_table(raw_summary, tables_dir, "raw_crsp_duplicate_summary.csv")
    write_table(conflict_df, tables_dir, "raw_crsp_duplicate_conflict_columns.csv")
    write_table(duplicate_key_details.head(500), tables_dir, "raw_crsp_top_duplicate_keys.csv")
    write_table(group_size_dist, tables_dir, "raw_crsp_duplicate_group_size_distribution.csv")
    write_table(by_year, tables_dir, "raw_crsp_duplicate_keys_by_year.csv")

    duplicate_sample = duplicate_rows.merge(
        duplicate_key_details.head(80)[key], on=key, how="inner"
    ).sort_values(["PERMNO", "month", "CUSIP", "Ticker", "TradingSymbol"])
    write_table(duplicate_sample, tables_dir, "raw_crsp_duplicate_sample_rows.csv")

    metadata_conflict_keys = duplicate_key_details[metadata_conflict_mask].head(80)[key]
    metadata_conflict_sample = duplicate_rows.merge(
        metadata_conflict_keys, on=key, how="inner"
    ).sort_values(["PERMNO", "month", "CUSIP", "Ticker", "TradingSymbol"])
    write_table(metadata_conflict_sample, tables_dir, "raw_crsp_metadata_conflict_sample_rows.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(by_year["year"], by_year["duplicate_permno_month_keys"], color="#a54242")
    ax.set_title("Raw CRSP duplicate PERMNO-month keys by year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Duplicate keys")
    save_plot(fig, plots_dir, "raw_crsp_duplicate_keys_by_year.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(group_size_dist["rows_per_permno_month"], group_size_dist["duplicate_key_count"], color="#7755a6")
    ax.set_title("Raw CRSP duplicate group sizes")
    ax.set_xlabel("Rows per PERMNO-month key")
    ax.set_ylabel("Duplicate key count")
    save_plot(fig, plots_dir, "raw_crsp_duplicate_group_sizes.png")

    return raw_summary


def validate_text_links(
    processed_dir: Path, tables_dir: Path, plots_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sec_path = processed_dir / "sec_10k_with_links.parquet"
    calls_path = processed_dir / "earnings_calls_with_gvkey.parquet"

    sec = pd.read_parquet(
        sec_path,
        columns=[
            "filing_date",
            "report_date",
            "report_year",
            "cik",
            "gvkey",
            "permno",
            "comp_report_distance_days",
            "linktype",
            "linkprim",
        ],
    )
    sec_summary = pd.DataFrame(
        [
            {
                "dataset": "sec_10k_with_links",
                "rows": len(sec),
                "unique_ciks": sec["cik"].nunique(dropna=True),
                "cik_nonmissing_pct": pct(sec["cik"].notna().sum(), len(sec)),
                "gvkey_match_pct": pct(sec["gvkey"].notna().sum(), len(sec)),
                "permno_match_pct": pct(sec["permno"].notna().sum(), len(sec)),
                "median_comp_report_distance_days": coerce_numeric(
                    sec["comp_report_distance_days"]
                ).median(),
                "p95_comp_report_distance_days": coerce_numeric(
                    sec["comp_report_distance_days"]
                ).quantile(0.95),
            }
        ]
    )
    sec_by_year = (
        sec.assign(year=pd.to_numeric(sec["report_year"], errors="coerce"))
        .groupby("year", dropna=True)
        .agg(
            rows=("gvkey", "size"),
            gvkey_matched=("gvkey", lambda s: s.notna().sum()),
            permno_matched=("permno", lambda s: s.notna().sum()),
        )
        .reset_index()
    )
    sec_by_year["gvkey_match_pct"] = sec_by_year.apply(
        lambda row: pct(row["gvkey_matched"], row["rows"]), axis=1
    )
    sec_by_year["permno_match_pct"] = sec_by_year.apply(
        lambda row: pct(row["permno_matched"], row["rows"]), axis=1
    )

    calls = pd.read_parquet(
        calls_path,
        columns=[
            "transcriptid",
            "permno",
            "mostimportantdateutc",
            "gvkey",
            "linktype",
            "linkprim",
            "word_count",
        ],
    )
    calls["year"] = pd.to_datetime(calls["mostimportantdateutc"], errors="coerce").dt.year
    calls_summary = pd.DataFrame(
        [
            {
                "dataset": "earnings_calls_with_gvkey",
                "rows": len(calls),
                "unique_transcripts": calls["transcriptid"].nunique(dropna=True),
                "unique_permnos": calls["permno"].nunique(dropna=True),
                "permno_nonmissing_pct": pct(calls["permno"].notna().sum(), len(calls)),
                "gvkey_match_pct": pct(calls["gvkey"].notna().sum(), len(calls)),
                "median_word_count": coerce_numeric(calls["word_count"]).median(),
            }
        ]
    )
    calls_by_year = (
        calls.groupby("year", dropna=True)
        .agg(
            rows=("gvkey", "size"),
            unique_permnos=("permno", "nunique"),
            gvkey_matched=("gvkey", lambda s: s.notna().sum()),
        )
        .reset_index()
    )
    calls_by_year["gvkey_match_pct"] = calls_by_year.apply(
        lambda row: pct(row["gvkey_matched"], row["rows"]), axis=1
    )

    link_tables = [linktype_counts(sec, "sec_10k_with_links"), linktype_counts(calls, "earnings_calls_with_gvkey")]

    write_table(sec_summary, tables_dir, "sec_10k_match_summary.csv")
    write_table(sec_by_year, tables_dir, "sec_10k_match_by_report_year.csv")
    write_table(calls_summary, tables_dir, "earnings_calls_match_summary.csv")
    write_table(calls_by_year, tables_dir, "earnings_calls_by_year.csv")
    write_table(pd.concat(link_tables, ignore_index=True), tables_dir, "text_linktype_counts.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sec_by_year["year"], sec_by_year["gvkey_match_pct"], label="gvkey")
    ax.plot(sec_by_year["year"], sec_by_year["permno_match_pct"], label="permno")
    ax.set_title("SEC 10-K link match rate by report year")
    ax.set_xlabel("Report year")
    ax.set_ylabel("Match rate (%)")
    ax.legend()
    save_plot(fig, plots_dir, "sec_10k_match_rate_by_year.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(calls_by_year["year"], calls_by_year["rows"], color="#4f8a6b")
    ax.set_title("Linked earnings calls by year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rows")
    save_plot(fig, plots_dir, "earnings_calls_by_year.png")

    text_summary = pd.concat([sec_summary, calls_summary], ignore_index=True, sort=False)
    warnings = []
    sec_row = sec_summary.iloc[0]
    calls_row = calls_summary.iloc[0]
    if sec_row["gvkey_match_pct"] < 80:
        warnings.append(
            {
                "severity": "warning",
                "dataset": "sec_10k_with_links",
                "check": "gvkey_match_pct",
                "message": f"SEC 10-K GVKEY match rate is {sec_row['gvkey_match_pct']:.2f}%; coverage is incomplete but non-blocking.",
            }
        )
    if sec_row["permno_match_pct"] < 80:
        warnings.append(
            {
                "severity": "warning",
                "dataset": "sec_10k_with_links",
                "check": "permno_match_pct",
                "message": f"SEC 10-K PERMNO match rate is {sec_row['permno_match_pct']:.2f}%; coverage is incomplete but non-blocking.",
            }
        )
    if calls_row["gvkey_match_pct"] < 95:
        warnings.append(
            {
                "severity": "warning",
                "dataset": "earnings_calls_with_gvkey",
                "check": "gvkey_match_pct",
                "message": f"Earnings-call GVKEY match rate is {calls_row['gvkey_match_pct']:.2f}%; review call coverage.",
            }
        )
    warnings_df = pd.DataFrame(warnings)
    if warnings_df.empty:
        warnings_df = pd.DataFrame(columns=["severity", "dataset", "check", "message"])
    write_table(warnings_df, tables_dir, "validation_warnings.csv")
    return text_summary, warnings_df


def build_failures(
    monthly_duplicates: pd.DataFrame,
    lookahead_summary: pd.DataFrame,
    link_summary: pd.DataFrame,
) -> pd.DataFrame:
    failures = []
    for _, row in monthly_duplicates.iterrows():
        if row["duplicate_permno_mthcaldt_keys"] > 0:
            failures.append(
                {
                    "severity": "failure",
                    "dataset": row["dataset"],
                    "check": "duplicate_permno_mthcaldt_keys",
                    "message": f"{int(row['duplicate_permno_mthcaldt_keys']):,} duplicate PERMNO-MthCalDt keys remain.",
                }
            )
        if row["duplicate_permno_month_keys"] > 0:
            failures.append(
                {
                    "severity": "failure",
                    "dataset": row["dataset"],
                    "check": "duplicate_permno_month_keys",
                    "message": f"{int(row['duplicate_permno_month_keys']):,} duplicate PERMNO-month keys remain.",
                }
            )

    if not lookahead_summary.empty:
        item = lookahead_summary.iloc[0]
        if int(item.get("lookahead_violation_rows", 0)) > 0:
            failures.append(
                {
                    "severity": "failure",
                    "dataset": "crsp_compustat_panel",
                    "check": "lookahead_violation_rows",
                    "message": f"{int(item['lookahead_violation_rows']):,} rows have Compustat availability dates after the CRSP month.",
                }
            )
        if int(item.get("negative_age_rows", 0)) > 0:
            failures.append(
                {
                    "severity": "failure",
                    "dataset": "crsp_compustat_panel",
                    "check": "negative_age_rows",
                    "message": f"{int(item['negative_age_rows']):,} matched rows have negative Compustat age.",
                }
            )

    if not link_summary.empty and "deduplicated_crsp_used" in link_summary.columns:
        monthly = link_summary[link_summary["dataset"].isin(["crsp_with_gvkey", "crsp_compustat_panel"])]
        if not monthly["deduplicated_crsp_used"].fillna(False).all():
            failures.append(
                {
                    "severity": "failure",
                    "dataset": "linked_monthly",
                    "check": "deduplicated_crsp_used",
                    "message": "One or more CRSP-derived linked datasets were not built from the deduplicated CRSP input.",
                }
            )

    if not failures:
        return pd.DataFrame(columns=["severity", "dataset", "check", "message"])
    return pd.DataFrame(failures)


def write_validation_report(
    output_dir: Path,
    link_summary: pd.DataFrame,
    monthly_duplicates: pd.DataFrame,
    monthly_quality: pd.DataFrame,
    lookahead_summary: pd.DataFrame,
    text_summary: pd.DataFrame,
    warnings: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    status = "FAILED" if not failures.empty else "PASSED"

    lines = [
        "# Linked Dataset Validation",
        "",
        f"Status: {status}",
        "",
        "## Core CRSP Checks",
        "",
        markdown_table(monthly_duplicates),
        "",
        markdown_table(monthly_quality),
        "",
        "## Compustat Look-Ahead Check",
        "",
        markdown_table(lookahead_summary),
        "",
    ]

    if not failures.empty:
        lines.extend(
            [
                "## Failures",
                "",
                markdown_table(failures),
                "",
            ]
        )

    lines.extend(["## Text Data Checks", "", markdown_table(text_summary), ""])

    if not warnings.empty:
        lines.extend(["## Warnings", "", markdown_table(warnings), ""])
    else:
        lines.extend(["## Warnings", "", "_No warnings._", ""])

    if not link_summary.empty:
        lines.extend(["## Existing link summary", "", markdown_table(link_summary), ""])

    (output_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    tables_dir, plots_dir = ensure_dirs(args.output_dir)

    link_summary = load_link_summary(args.processed_dir)
    if not link_summary.empty:
        write_table(link_summary, tables_dir, "link_summary_from_build.csv")

    _, monthly_duplicates, monthly_quality, lookahead_summary = validate_monthly(
        args.processed_dir, tables_dir, plots_dir
    )
    text_summary, warnings = validate_text_links(args.processed_dir, tables_dir, plots_dir)
    failures = build_failures(monthly_duplicates, lookahead_summary, link_summary)
    validation_status = pd.DataFrame(
        [
            {
                "status": "FAILED" if not failures.empty else "PASSED",
                "failure_count": len(failures),
                "warning_count": len(warnings),
            }
        ]
    )
    write_table(validation_status, tables_dir, "validation_status.csv")
    write_table(failures, tables_dir, "validation_failures.csv")
    write_table(
        pd.concat(
            [
                monthly_quality.assign(section="monthly"),
                text_summary.assign(section="text"),
            ],
            ignore_index=True,
            sort=False,
        ),
        tables_dir,
        "validation_match_summary.csv",
    )
    write_validation_report(
        args.output_dir,
        link_summary,
        monthly_duplicates,
        monthly_quality,
        lookahead_summary,
        text_summary,
        warnings,
        failures,
    )

    print(f"Wrote sanity-check tables to {tables_dir}")
    print(f"Wrote sanity-check plots to {plots_dir}")
    print(f"Wrote validation report to {args.output_dir / 'validation_report.md'}")
    if not failures.empty:
        raise SystemExit("Validation failed: see validation_failures.csv.")
    print("Validation passed.")


if __name__ == "__main__":
    main()
