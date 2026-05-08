#!/usr/bin/env python3
"""Build project-level linked datasets using SEC CIK and WRDS CCM links.

Outputs:
  - crsp_with_gvkey.parquet: monthly CRSP targets with CCM gvkey links.
  - crsp_compustat_panel.parquet: CRSP rows matched to lagged Compustat data.
  - sec_10k_with_links.parquet: 10-K filings linked to gvkey and PERMNO.
  - earnings_calls_with_gvkey.parquet: call transcripts linked to gvkey.
  - link_summary.json: row counts and match rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_CCM = Path("Dataset/Linking/ccm_links.parquet")
DEFAULT_CRSP = Path("Dataset/Targets/monthly_crsp.csv")
DEFAULT_COMPUSTAT = Path("Dataset/Predictors/CompFirmCharac.csv")
DEFAULT_10K = Path("Dataset/Predictors/10K_fillings.parquet")
DEFAULT_CALLS = Path("Dataset/Predictors/sm-calls_with_connectors.parquet")
DEFAULT_OUTPUT_DIR = Path("Dataset/Processed")

DEFAULT_COMPUSTAT_COLUMNS = [
    "gvkey",
    "datadate",
    "fyearq",
    "fqtr",
    "fyr",
    "tic",
    "cusip",
    "conm",
    "cik",
    "fic",
    "curcdq",
    "exchg",
    "costat",
    "saley",
    "revty",
    "cogsy",
    "xopry",
    "xsgay",
    "xrdy",
    "capxy",
    "dpcy",
    "chechy",
    "dlcchy",
    "dltisy",
    "dltry",
    "wcapcy",
    "oancfy",
    "ivncfy",
    "fincfy",
    "niy",
    "ibcy",
    "epspxy",
    "cshpry",
    "dvy",
    "dvpy",
    "prstkcy",
    "sstky",
    "txty",
    "xinty",
    "piy",
    "oiadpy",
    "oibdpy",
    "recchy",
    "invchy",
    "aqcy",
]

LINKPRIM_RANK = {"P": 0, "C": 1}
LINKTYPE_RANK = {"LC": 0, "LU": 1, "LS": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build linked CRSP/Compustat/SEC/call datasets."
    )
    parser.add_argument("--ccm-path", type=Path, default=DEFAULT_CCM)
    parser.add_argument("--crsp-path", type=Path, default=DEFAULT_CRSP)
    parser.add_argument("--compustat-path", type=Path, default=DEFAULT_COMPUSTAT)
    parser.add_argument("--sec-10k-path", type=Path, default=DEFAULT_10K)
    parser.add_argument("--calls-path", type=Path, default=DEFAULT_CALLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--compustat-lag-months",
        type=int,
        default=3,
        help="Months added to Compustat datadate before data is usable.",
    )
    parser.add_argument(
        "--max-compustat-age-days",
        type=int,
        default=548,
        help="Maximum age of the latest available Compustat observation.",
    )
    parser.add_argument(
        "--sec-match-tolerance-days",
        type=int,
        default=370,
        help="Maximum distance between 10-K report_date and Compustat datadate.",
    )
    parser.add_argument(
        "--compustat-cols",
        nargs="+",
        default=DEFAULT_COMPUSTAT_COLUMNS,
        help="Compustat columns for the panel. Use 'all' to include all columns.",
    )
    parser.add_argument(
        "--drop-text",
        action="store_true",
        help="Drop long text fields from SEC and earnings-call linked outputs.",
    )
    return parser.parse_args()


def normalize_gvkey(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.zfill(6)


def normalize_cik(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    normalized = normalized.mask(normalized.isin(["", "<NA>", "nan", "None"]))
    return normalized.str.zfill(10)


def lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df


def existing_columns(path: Path, requested: Iterable[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns
    available = {c.lower(): c for c in header}
    cols = []
    missing = []
    for col in requested:
        key = col.lower()
        if key in available:
            cols.append(available[key])
        else:
            missing.append(col)
    if missing:
        print(f"Skipping missing Compustat columns: {', '.join(missing)}")
    return cols


def prepare_ccm(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing CCM link file: {path}. "
            "Run scripts/download_ccm_links.py first."
        )

    ccm = lower_columns(pd.read_parquet(path))
    rename = {}
    if "lpermno" in ccm.columns:
        rename["lpermno"] = "permno"
    if "lpermco" in ccm.columns:
        rename["lpermco"] = "permco"
    ccm = ccm.rename(columns=rename)

    required = {"gvkey", "permno", "permco", "linkdt", "linkenddt", "linktype", "linkprim"}
    missing = required.difference(ccm.columns)
    if missing:
        raise ValueError(f"CCM file is missing required columns: {sorted(missing)}")

    ccm["gvkey"] = normalize_gvkey(ccm["gvkey"])
    ccm["ccm_permno"] = pd.to_numeric(ccm["permno"], errors="coerce").astype("Int64")
    ccm["ccm_permco"] = pd.to_numeric(ccm["permco"], errors="coerce").astype("Int64")
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"], errors="coerce")
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"], errors="coerce").fillna(
        pd.Timestamp("2100-12-31")
    )
    ccm["linkprim"] = ccm["linkprim"].astype("string").str.strip()
    ccm["linktype"] = ccm["linktype"].astype("string").str.strip()
    if "liid" not in ccm.columns:
        ccm["liid"] = pd.NA
    ccm["liid"] = ccm["liid"].astype("string").str.strip()

    ccm = ccm[
        ccm["gvkey"].notna()
        & ccm["ccm_permno"].notna()
        & ccm["linkdt"].notna()
        & ccm["linktype"].isin(LINKTYPE_RANK)
        & ccm["linkprim"].isin(LINKPRIM_RANK)
    ]
    return ccm[
        ["gvkey", "ccm_permno", "ccm_permco", "liid", "linktype", "linkprim", "linkdt", "linkenddt"]
    ].reset_index(drop=True)


def select_valid_links(
    candidates: pd.DataFrame,
    row_id_col: str,
    date_col: str,
    selected_cols: list[str],
) -> pd.DataFrame:
    date = candidates[date_col]
    valid = candidates[
        candidates["linkdt"].notna()
        & date.ge(candidates["linkdt"])
        & date.le(candidates["linkenddt"])
    ].copy()

    if valid.empty:
        return pd.DataFrame(columns=[row_id_col] + selected_cols)

    valid["_linkprim_rank"] = valid["linkprim"].map(LINKPRIM_RANK).fillna(99)
    valid["_linktype_rank"] = valid["linktype"].map(LINKTYPE_RANK).fillna(99)
    valid = valid.sort_values(
        [row_id_col, "_linkprim_rank", "_linktype_rank", "linkdt"],
        ascending=[True, True, True, False],
    )
    return valid.drop_duplicates(row_id_col, keep="first")[[row_id_col] + selected_cols]


def link_by_permno_date(
    observations: pd.DataFrame,
    ccm: pd.DataFrame,
    permno_col: str,
    date_col: str,
) -> pd.DataFrame:
    obs = observations.copy()
    obs["_row_id"] = range(len(obs))
    obs[permno_col] = pd.to_numeric(obs[permno_col], errors="coerce").astype("Int64")
    obs[date_col] = pd.to_datetime(obs[date_col], errors="coerce")

    candidates = obs.merge(
        ccm,
        left_on=permno_col,
        right_on="ccm_permno",
        how="left",
        suffixes=("", "_ccm"),
    )
    selected = select_valid_links(
        candidates,
        "_row_id",
        date_col,
        ["gvkey", "ccm_permco", "liid", "linktype", "linkprim", "linkdt", "linkenddt"],
    )
    out = obs.merge(selected, on="_row_id", how="left").drop(columns="_row_id")
    return out


def link_by_gvkey_date(
    observations: pd.DataFrame,
    ccm: pd.DataFrame,
    gvkey_col: str,
    date_col: str,
) -> pd.DataFrame:
    obs = observations.copy()
    obs["_row_id"] = range(len(obs))
    obs[gvkey_col] = normalize_gvkey(obs[gvkey_col])
    obs[date_col] = pd.to_datetime(obs[date_col], errors="coerce")

    candidates = obs.merge(
        ccm,
        left_on=gvkey_col,
        right_on="gvkey",
        how="left",
        suffixes=("", "_ccm"),
    )
    selected = select_valid_links(
        candidates,
        "_row_id",
        date_col,
        ["ccm_permno", "ccm_permco", "liid", "linktype", "linkprim", "linkdt", "linkenddt"],
    )
    out = obs.merge(selected, on="_row_id", how="left").drop(columns="_row_id")
    return out.rename(columns={"ccm_permno": "permno", "ccm_permco": "permco"})


def load_crsp(path: Path) -> pd.DataFrame:
    crsp = pd.read_csv(
        path,
        dtype={
            "PERMNO": "Int64",
            "PERMCO": "Int64",
            "HdrCUSIP": "string",
            "CUSIP": "string",
            "Ticker": "string",
            "TradingSymbol": "string",
        },
    )
    crsp = lower_columns(crsp)
    crsp["mthcaldt"] = pd.to_datetime(crsp["mthcaldt"], errors="coerce")
    return crsp


def load_compustat(path: Path, compustat_cols: list[str]) -> pd.DataFrame:
    if compustat_cols == ["all"]:
        usecols = None
    else:
        required = ["gvkey", "datadate", "cik", "tic", "cusip", "conm", "fyearq", "fqtr", "fyr"]
        usecols = existing_columns(path, dict.fromkeys(required + compustat_cols).keys())

    dtype = {
        "gvkey": "string",
        "cik": "string",
        "tic": "string",
        "cusip": "string",
        "conm": "string",
    }
    try:
        comp = pd.read_csv(path, usecols=usecols, dtype=dtype, engine="pyarrow")
    except (ImportError, ValueError):
        comp = pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False)
    comp = lower_columns(comp)
    comp["gvkey"] = normalize_gvkey(comp["gvkey"])
    comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")
    if "cik" in comp.columns:
        comp["cik_norm"] = normalize_cik(comp["cik"])
    return comp


def prefix_compustat_columns(comp: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in comp.columns:
        if col == "gvkey":
            continue
        if col == "datadate":
            rename[col] = "comp_datadate"
        elif col == "cik_norm":
            rename[col] = "comp_cik_norm"
        elif not col.startswith("comp_"):
            rename[col] = f"comp_{col}"
    return comp.rename(columns=rename)


def build_compustat_panel(
    crsp_with_gvkey: pd.DataFrame,
    comp: pd.DataFrame,
    lag_months: int,
    max_age_days: int,
) -> pd.DataFrame:
    comp_panel = comp.copy()
    comp_panel["comp_available_date"] = comp_panel["datadate"] + pd.DateOffset(months=lag_months)
    comp_panel = prefix_compustat_columns(comp_panel)

    left = crsp_with_gvkey[crsp_with_gvkey["gvkey"].notna()].copy()
    # pandas merge_asof requires the as-of key to be globally sorted even when
    # matching within by-groups.
    left = left.sort_values(["mthcaldt", "gvkey"])
    right = comp_panel.dropna(subset=["gvkey", "comp_available_date"]).sort_values(
        ["comp_available_date", "gvkey"]
    )

    panel = pd.merge_asof(
        left,
        right,
        by="gvkey",
        left_on="mthcaldt",
        right_on="comp_available_date",
        direction="backward",
    )
    panel["compustat_age_days"] = (
        panel["mthcaldt"] - panel["comp_available_date"]
    ).dt.days
    panel["compustat_match"] = panel["compustat_age_days"].between(
        0, max_age_days, inclusive="both"
    )

    stale = ~panel["compustat_match"].fillna(False)
    comp_cols = [c for c in panel.columns if c.startswith("comp_")]
    panel.loc[stale, comp_cols] = pd.NA
    panel.loc[stale, "compustat_age_days"] = pd.NA
    return panel


def link_sec_10k(sec_path: Path, comp: pd.DataFrame, ccm: pd.DataFrame, drop_text: bool, tolerance_days: int) -> pd.DataFrame:
    sec = pd.read_parquet(sec_path)
    if drop_text and "text" in sec.columns:
        sec = sec.drop(columns="text")
    sec = lower_columns(sec)
    sec["filing_date"] = pd.to_datetime(sec["filing_date"], errors="coerce")
    sec["report_date"] = pd.to_datetime(sec["report_date"], errors="coerce")
    sec["cik_norm"] = normalize_cik(sec["cik"])
    sec["_row_id"] = range(len(sec))

    comp_ids = comp[
        ["gvkey", "datadate", "cik_norm", "tic", "cusip", "conm", "fyearq", "fqtr"]
    ].dropna(subset=["cik_norm", "datadate"])
    annual_ids = comp_ids[comp_ids["fqtr"].eq(4)]
    if annual_ids.empty:
        annual_ids = comp_ids

    pieces = []
    comp_groups = {key: group.sort_values("datadate") for key, group in annual_ids.groupby("cik_norm")}
    for cik, sec_group in sec.groupby("cik_norm", dropna=False):
        right = comp_groups.get(cik)
        left = sec_group.sort_values("report_date")
        if right is None or left["report_date"].isna().all():
            for col in ["gvkey", "datadate", "tic", "cusip", "conm", "fyearq", "fqtr"]:
                left[col] = pd.NA
            pieces.append(left)
            continue
        matched = pd.merge_asof(
            left,
            right,
            left_on="report_date",
            right_on="datadate",
            direction="nearest",
            tolerance=pd.Timedelta(days=tolerance_days),
            suffixes=("", "_comp"),
        )
        pieces.append(matched)

    linked = pd.concat(pieces, ignore_index=True).sort_values("_row_id")
    linked["comp_report_distance_days"] = (
        linked["report_date"] - linked["datadate"]
    ).abs().dt.days
    linked = linked.rename(
        columns={
            "datadate": "comp_datadate",
            "tic": "comp_tic",
            "cusip": "comp_cusip",
            "conm": "comp_conm",
            "fyearq": "comp_fyearq",
            "fqtr": "comp_fqtr",
        }
    ).drop(columns="_row_id")
    linked["filing_month"] = linked["filing_date"] + pd.offsets.MonthEnd(0)

    return link_by_gvkey_date(linked, ccm, "gvkey", "filing_date")


def link_calls(calls_path: Path, ccm: pd.DataFrame, drop_text: bool) -> pd.DataFrame:
    calls = pd.read_parquet(calls_path)
    if drop_text and "text" in calls.columns:
        calls = calls.drop(columns="text")
    calls = lower_columns(calls)
    calls["mostimportantdateutc"] = pd.to_datetime(calls["mostimportantdateutc"], errors="coerce")
    calls["call_month"] = calls["mostimportantdateutc"] + pd.offsets.MonthEnd(0)
    return link_by_permno_date(calls, ccm, "permno", "mostimportantdateutc")


def pct(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 2) if denominator else 0.0


def write_summary(output_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    summary = {}
    for name, df in outputs.items():
        row_count = len(df)
        item = {"rows": row_count}
        if "gvkey" in df.columns:
            matched = int(df["gvkey"].notna().sum())
            item["gvkey_matched_rows"] = matched
            item["gvkey_match_pct"] = pct(matched, row_count)
        if "permno" in df.columns:
            matched = int(df["permno"].notna().sum())
            item["permno_nonmissing_rows"] = matched
            item["permno_nonmissing_pct"] = pct(matched, row_count)
        if "compustat_match" in df.columns:
            matched = int(df["compustat_match"].fillna(False).sum())
            item["compustat_matched_rows"] = matched
            item["compustat_match_pct"] = pct(matched, row_count)
        summary[name] = item

    with (output_dir / "link_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CCM links...", flush=True)
    ccm = prepare_ccm(args.ccm_path)
    print(f"Loaded {len(ccm):,} CCM links.", flush=True)

    print("Loading monthly CRSP targets...", flush=True)
    crsp = load_crsp(args.crsp_path)
    print(f"Loaded {len(crsp):,} CRSP rows. Linking CRSP to gvkey...", flush=True)
    crsp_with_gvkey = link_by_permno_date(crsp, ccm, "permno", "mthcaldt")
    crsp_with_gvkey.to_parquet(args.output_dir / "crsp_with_gvkey.parquet", index=False)
    print("Wrote crsp_with_gvkey.parquet.", flush=True)

    print("Loading Compustat firm characteristics...", flush=True)
    comp = load_compustat(args.compustat_path, args.compustat_cols)
    print(f"Loaded {len(comp):,} Compustat rows. Building lagged panel...", flush=True)
    panel = build_compustat_panel(
        crsp_with_gvkey,
        comp,
        lag_months=args.compustat_lag_months,
        max_age_days=args.max_compustat_age_days,
    )
    panel.to_parquet(args.output_dir / "crsp_compustat_panel.parquet", index=False)
    print("Wrote crsp_compustat_panel.parquet.", flush=True)

    print("Linking SEC 10-K filings by CIK, then to CCM...", flush=True)
    sec_10k = link_sec_10k(
        args.sec_10k_path,
        comp,
        ccm,
        drop_text=args.drop_text,
        tolerance_days=args.sec_match_tolerance_days,
    )
    sec_10k.to_parquet(args.output_dir / "sec_10k_with_links.parquet", index=False)
    print("Wrote sec_10k_with_links.parquet.", flush=True)

    print("Linking earnings calls by PERMNO to CCM...", flush=True)
    calls = link_calls(args.calls_path, ccm, drop_text=args.drop_text)
    calls.to_parquet(args.output_dir / "earnings_calls_with_gvkey.parquet", index=False)
    print("Wrote earnings_calls_with_gvkey.parquet.", flush=True)

    outputs = {
        "crsp_with_gvkey": crsp_with_gvkey,
        "crsp_compustat_panel": panel,
        "sec_10k_with_links": sec_10k,
        "earnings_calls_with_gvkey": calls,
    }
    write_summary(args.output_dir, outputs)

    for name, df in outputs.items():
        print(f"{name}: {len(df):,} rows")
    print(f"Wrote linked outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
