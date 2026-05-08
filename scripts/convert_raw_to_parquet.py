#!/usr/bin/env python3
"""Convert raw predictor and target datasets to a parquet mirror.

The script reads files from ``Dataset/Predictors`` and ``Dataset/Targets`` and
writes parquet equivalents under ``Dataset/Parquet``. Documentation files such
as PDFs are skipped. Existing parquet files are copied by default so downstream
work can point at a single parquet-only raw-data directory.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_SOURCES = [Path("Dataset/Predictors"), Path("Dataset/Targets")]
DEFAULT_OUTPUT = Path("Dataset/Parquet")

IDENTIFIER_COLUMNS = {
    "cik",
    "conm",
    "companyid",
    "companyname",
    "company_conformed_name",
    "cusip",
    "datafmt",
    "datacqtr",
    "datafqtr",
    "fic",
    "freq",
    "gvkey",
    "hdrcusip",
    "headline",
    "indfmt",
    "liid",
    "location",
    "name",
    "popsrc",
    "submission_type",
    "tic",
    "ticker",
    "tradingsymbol",
    "weighting",
}

DATE_COLUMNS = {
    "call_month",
    "datadate",
    "date",
    "filing_date",
    "linkdt",
    "linkenddt",
    "mostimportantdateutc",
    "mthcaldt",
    "report_date",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw predictor and target CSV/parquet files to parquet."
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        type=Path,
        default=DEFAULT_SOURCES,
        help="Directories to scan. Defaults to Dataset/Predictors Dataset/Targets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output root. Defaults to Dataset/Parquet.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parquet outputs.",
    )
    parser.add_argument(
        "--no-copy-existing-parquet",
        action="store_true",
        help="Do not copy source files that are already parquet.",
    )
    parser.add_argument(
        "--include-excel",
        action="store_true",
        help="Also convert .xlsx files, one parquet per sheet.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["snappy", "gzip", "brotli", "zstd", "none"],
        help="Parquet compression codec. Defaults to zstd.",
    )
    return parser.parse_args()


def safe_sheet_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "sheet"


def output_path_for(source: Path, source_root: Path, output_root: Path) -> Path:
    relative_parent = source.parent.relative_to(source_root)
    return output_root / source_root.name / relative_parent / source.with_suffix(".parquet").name


def compression_arg(codec: str) -> str | None:
    return None if codec == "none" else codec


def read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def dtype_map(columns: Iterable[str]) -> dict[str, str]:
    return {col: "string" for col in columns if col.lower() in IDENTIFIER_COLUMNS}


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col.lower() in DATE_COLUMNS:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def write_parquet(df: pd.DataFrame, path: Path, compression: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression=compression)


def convert_csv(source: Path, target: Path, compression: str | None) -> dict[str, object]:
    columns = read_header(source)
    dtypes = dtype_map(columns)
    try:
        df = pd.read_csv(source, dtype=dtypes, engine="pyarrow")
    except (ImportError, ValueError):
        df = pd.read_csv(source, dtype=dtypes, low_memory=False)

    df = normalize_dates(df)
    write_parquet(df, target, compression)
    return {"source": str(source), "target": str(target), "rows": len(df), "columns": len(df.columns)}


def copy_parquet(source: Path, target: Path) -> dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {"source": str(source), "target": str(target), "copied": True}


def convert_excel(source: Path, target: Path, compression: str | None) -> list[dict[str, object]]:
    outputs = []
    sheets = pd.read_excel(source, sheet_name=None)
    for sheet_name, df in sheets.items():
        sheet_target = target.with_name(f"{target.stem}__{safe_sheet_name(sheet_name)}.parquet")
        df = normalize_dates(df)
        write_parquet(df, sheet_target, compression)
        outputs.append(
            {
                "source": str(source),
                "sheet": sheet_name,
                "target": str(sheet_target),
                "rows": len(df),
                "columns": len(df.columns),
            }
        )
    return outputs


def should_skip(path: Path, args: argparse.Namespace) -> bool:
    suffix = path.suffix.lower()
    if path.name.startswith("."):
        return True
    if suffix in {".csv", ".parquet"}:
        return False
    if suffix == ".xlsx" and args.include_excel:
        return False
    return True


def main() -> None:
    args = parse_args()
    compression = compression_arg(args.compression)
    results: list[dict[str, object]] = []
    skipped: list[str] = []

    for source_root in args.source_dirs:
        for source in sorted(source_root.rglob("*")):
            if not source.is_file() or should_skip(source, args):
                continue

            target = output_path_for(source, source_root, args.output_dir)
            if target.exists() and not args.overwrite:
                skipped.append(str(target))
                continue

            suffix = source.suffix.lower()
            print(f"Processing {source} -> {target}", flush=True)
            if suffix == ".csv":
                results.append(convert_csv(source, target, compression))
            elif suffix == ".parquet":
                if args.no_copy_existing_parquet:
                    skipped.append(str(source))
                else:
                    results.append(copy_parquet(source, target))
            elif suffix == ".xlsx":
                results.extend(convert_excel(source, target, compression))

    summary = {
        "converted_or_copied": results,
        "skipped_existing": skipped,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "conversion_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary to {summary_path}")
    print(f"Converted/copied {len(results)} file(s); skipped {len(skipped)} existing output(s).")


if __name__ == "__main__":
    main()
