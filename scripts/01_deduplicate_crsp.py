#!/usr/bin/env python3
"""Create a deduplicated monthly CRSP target file.

The raw CRSP CSV is left untouched. This script removes exact repeated rows,
then resolves remaining duplicate PERMNO-MthCalDt keys only when target/core
fields agree within each key.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("Dataset/Targets/monthly_crsp.csv")
DEFAULT_OUTPUT = Path("Dataset/Processed/crsp_monthly_deduped.parquet")
DEFAULT_AUDIT = Path("outputs/sanity_checks/tables/crsp_dedup_audit.csv")
DEFAULT_CONFLICTS = Path("outputs/sanity_checks/tables/crsp_core_field_conflicts.csv")

KEY_COLUMNS = ["PERMNO", "MthCalDt"]
CORE_COLUMNS = ["MthRet", "sprtrn", "PERMCO", "HdrCUSIP"]
METADATA_PRIORITY_COLUMNS = ["CUSIP", "Ticker", "TradingSymbol", "SICCD", "NAICS"]
TEXT_COLUMNS = ["HdrCUSIP", "CUSIP", "Ticker", "TradingSymbol"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate monthly CRSP rows before linking/modeling."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--conflicts", type=Path, default=DEFAULT_CONFLICTS)
    return parser.parse_args()


def is_placeholder(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text_lower = text.str.lower()
    numeric_zero = pd.to_numeric(series, errors="coerce").eq(0)
    placeholder = (
        series.isna()
        | text.isna()
        | text.eq("")
        | text_lower.eq("nan")
        | text_lower.eq("0")
        | numeric_zero
    )
    return placeholder.fillna(True)


def read_crsp(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw CRSP file: {path}")

    df = pd.read_csv(
        path,
        dtype={
            "PERMNO": "Int64",
            "PERMCO": "Int64",
            "HdrCUSIP": "string",
            "CUSIP": "string",
            "Ticker": "string",
            "TradingSymbol": "string",
        },
        low_memory=False,
    )
    missing = set(KEY_COLUMNS + CORE_COLUMNS + METADATA_PRIORITY_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Raw CRSP file is missing required columns: {sorted(missing)}")

    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"], errors="coerce")
    return df


def duplicate_key_count(df: pd.DataFrame) -> int:
    counts = df.groupby(KEY_COLUMNS, dropna=False).size()
    return int((counts > 1).sum())


def duplicate_key_rows(df: pd.DataFrame) -> int:
    counts = df.groupby(KEY_COLUMNS, dropna=False).size()
    return int(counts[counts > 1].sum())


def validate_keys(df: pd.DataFrame) -> None:
    missing_key_rows = df[KEY_COLUMNS].isna().any(axis=1)
    if missing_key_rows.any():
        count = int(missing_key_rows.sum())
        raise ValueError(
            f"Cannot deduplicate CRSP rows because {count:,} rows have missing PERMNO or MthCalDt."
        )


def find_core_conflicts(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[int, pd.Timestamp]]]:
    duplicated_keys = df.duplicated(KEY_COLUMNS, keep=False)
    duplicate_rows = df.loc[duplicated_keys].copy()
    if duplicate_rows.empty:
        return pd.DataFrame(columns=list(df.columns) + ["conflict_fields"]), []

    conflict_fields_by_key: dict[tuple[int, pd.Timestamp], list[str]] = {}
    grouped = duplicate_rows.groupby(KEY_COLUMNS, dropna=False)
    for column in CORE_COLUMNS:
        unique_counts = grouped[column].nunique(dropna=False)
        for key in unique_counts[unique_counts > 1].index:
            conflict_fields_by_key.setdefault(key, []).append(column)

    if not conflict_fields_by_key:
        return pd.DataFrame(columns=list(df.columns) + ["conflict_fields"]), []

    conflict_keys = list(conflict_fields_by_key)
    key_frame = pd.DataFrame(
        [
            {
                "PERMNO": permno,
                "MthCalDt": mthcaldt,
                "conflict_fields": ",".join(fields),
            }
            for (permno, mthcaldt), fields in conflict_fields_by_key.items()
        ]
    )
    conflicts = duplicate_rows.merge(key_frame, on=KEY_COLUMNS, how="inner")
    return conflicts, conflict_keys


def score_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    for column in METADATA_PRIORITY_COLUMNS:
        working[f"_has_{column}"] = (~is_placeholder(working[column])).astype("int8")

    sort_columns = (
        KEY_COLUMNS
        + [f"_has_{column}" for column in METADATA_PRIORITY_COLUMNS]
        + ["_original_row_number"]
    )
    ascending = [True, True] + [False] * len(METADATA_PRIORITY_COLUMNS) + [True]
    sorted_rows = working.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    deduped = sorted_rows.drop_duplicates(KEY_COLUMNS, keep="first")
    helper_columns = [c for c in deduped.columns if c.startswith("_has_")] + [
        "_original_row_number"
    ]
    deduped = deduped.drop(columns=helper_columns)
    return deduped.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)


def write_audit(path: Path, audit: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([audit]).to_csv(path, index=False)


def main() -> None:
    args = parse_args()

    print(f"Reading raw monthly CRSP file: {args.input}")
    raw = read_crsp(args.input)
    validate_keys(raw)
    raw["_original_row_number"] = range(len(raw))

    data_columns = [c for c in raw.columns if c != "_original_row_number"]
    rows_before = len(raw)
    duplicate_keys_before = duplicate_key_count(raw)
    print(f"Rows before: {rows_before:,}")
    print(f"Duplicate PERMNO-MthCalDt keys before: {duplicate_keys_before:,}")

    without_exact_duplicates = raw.drop_duplicates(subset=data_columns, keep="first").copy()
    rows_after_exact = len(without_exact_duplicates)
    exact_removed = rows_before - rows_after_exact
    print(f"Exact full-row duplicates removed: {exact_removed:,}")

    conflicts, conflict_keys = find_core_conflicts(without_exact_duplicates)
    if conflict_keys:
        args.conflicts.parent.mkdir(parents=True, exist_ok=True)
        conflicts.drop(columns="_original_row_number", errors="ignore").to_csv(
            args.conflicts, index=False
        )
        print(f"Core field conflicts found: {len(conflict_keys):,} PERMNO-MthCalDt keys")
        print(f"Wrote conflicting rows to: {args.conflicts}")
        raise SystemExit(
            "Stopped: duplicate PERMNO-MthCalDt rows disagree on MthRet, sprtrn, PERMCO, or HdrCUSIP."
        )

    print("Core field conflicts found: 0")
    print(
        "Resolving remaining duplicate keys by metadata quality: "
        "CUSIP, Ticker, TradingSymbol, SICCD, NAICS."
    )
    deduped = score_and_deduplicate(without_exact_duplicates)
    rows_after_final = len(deduped)
    duplicate_keys_after = duplicate_key_count(deduped)
    remaining_duplicate_rows = duplicate_key_rows(deduped)
    metadata_removed = rows_after_exact - rows_after_final

    args.output.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_parquet(args.output, index=False)

    audit = {
        "rows_before": rows_before,
        "rows_after_exact_duplicate_drop": rows_after_exact,
        "rows_after_permno_month_dedup": rows_after_final,
        "exact_duplicate_rows_removed": exact_removed,
        "metadata_duplicate_rows_removed": metadata_removed,
        "duplicate_permno_month_keys_before": duplicate_keys_before,
        "duplicate_permno_month_keys_after": duplicate_keys_after,
        "remaining_duplicate_permno_month_rows": remaining_duplicate_rows,
    }
    write_audit(args.audit, audit)

    print(f"Metadata duplicate rows removed: {metadata_removed:,}")
    print(f"Rows after PERMNO-MthCalDt deduplication: {rows_after_final:,}")
    print(f"Duplicate PERMNO-MthCalDt keys after: {duplicate_keys_after:,}")
    print(f"Remaining duplicate PERMNO-MthCalDt rows after: {remaining_duplicate_rows:,}")
    print(f"Wrote deduplicated parquet: {args.output}")
    print(f"Wrote audit table: {args.audit}")


if __name__ == "__main__":
    main()
