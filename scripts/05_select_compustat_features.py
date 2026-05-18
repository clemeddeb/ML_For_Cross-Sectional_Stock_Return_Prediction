#!/usr/bin/env python3
"""Select and clean Compustat accounting features for the modeling panel."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT = Path("Dataset/Processed/model_panel_with_return_jkp_features.parquet")
DEFAULT_OUTPUT = Path("Dataset/Processed/model_panel_full_features.parquet")
DEFAULT_TABLE_DIR = Path("outputs/sanity_checks/tables")

KEY_COLUMNS = ["permno", "mthcaldt"]
IDENTIFIER_COLUMNS = ["permno", "gvkey", "mthcaldt", "ticker", "siccd", "naics"]
TARGET_COLUMNS = ["target_month", "target_ret_1m", "target_quintile", "top_bottom_label"]
RAW_RETURN_COLUMNS = ["mthret", "sprtrn"]
RETURN_FEATURE_COLUMNS = [
    "ret_lag_1m",
    "ret_lag_2m",
    "ret_lag_3m",
    "ret_lag_6m",
    "momentum_6m",
    "momentum_12m_excl_1m",
    "volatility_12m",
    "volatility_24m",
    "market_ret_lag_1m",
    "excess_ret_lag_1m",
    "beta_24m",
]
TARGET_COLUMN = "target_ret_1m"
TRAIN_SELECTION_START = pd.Timestamp("1990-01-01")
TRAIN_SELECTION_END = pd.Timestamp("2010-12-31")

COMPUSTAT_IDENTIFIER_COLUMNS = {
    "comp_cik",
    "comp_cik_norm",
    "comp_conm",
    "comp_cusip",
    "comp_tic",
}
COMPUSTAT_DATE_COLUMNS = {"comp_available_date", "comp_datadate"}
COMPUSTAT_CATEGORICAL_COLUMNS = {
    "comp_costat",
    "comp_curcdq",
    "comp_exchg",
    "comp_fic",
    "comp_fqtr",
    "comp_fyearq",
    "comp_fyr",
}
COMPUSTAT_HELPER_COLUMNS = {"compustat_age_days", "compustat_match"}

MISSING_INDICATOR_MIN_RATE = 0.10
MAX_MISSING_INDICATORS = 10
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and clean Compustat accounting features for modeling."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--modeling-start-date", type=str, default="1990-01-01")
    parser.add_argument("--train-diagnostic-end-date", type=str, default="2014-12-31")
    parser.add_argument("--missingness-threshold", type=float, default=0.60)
    parser.add_argument("--winsor-lower", type=float, default=0.01)
    parser.add_argument("--winsor-upper", type=float, default=0.99)
    parser.add_argument("--near-zero-variance-threshold", type=float, default=1e-12)
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args()


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def parquet_row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def present(columns: list[str], requested: list[str]) -> list[str]:
    available = set(columns)
    return [column for column in requested if column in available]


def assert_no_target_leakage(source_columns: list[str], context: str) -> None:
    forbidden = set(TARGET_COLUMNS)
    leaked = sorted(forbidden.intersection(source_columns))
    if leaked:
        raise ValueError(f"Target/label columns cannot be used for {context}: {leaked}")


def assert_unique_permno_month(df: pd.DataFrame) -> None:
    duplicates = df.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        sample = df.loc[duplicates, KEY_COLUMNS].head(20)
        raise ValueError(
            "Panel has duplicate PERMNO-MthCalDt rows. "
            f"First duplicate keys:\n{sample.to_string(index=False)}"
        )


def compustat_like_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if column.startswith("comp_") or column.startswith("compustat_")]


def exclusion_reason(column: str) -> str | None:
    lower = column.lower()
    if column in COMPUSTAT_IDENTIFIER_COLUMNS or any(token in lower for token in ["cusip", "cik"]):
        return "identifier_or_name"
    if column in COMPUSTAT_DATE_COLUMNS or "date" in lower:
        return "date_or_availability_metadata"
    if column in COMPUSTAT_HELPER_COLUMNS:
        return "linking_or_validation_helper"
    if column in COMPUSTAT_CATEGORICAL_COLUMNS:
        return "categorical_or_fiscal_metadata"
    return None


def read_panel_keys(input_path: Path) -> pd.DataFrame:
    keys = pd.read_parquet(input_path, columns=KEY_COLUMNS + TARGET_COLUMNS)
    keys["mthcaldt"] = pd.to_datetime(keys["mthcaldt"], errors="coerce")
    if keys["mthcaldt"].isna().any():
        bad_rows = int(keys["mthcaldt"].isna().sum())
        raise ValueError(f"Input panel has {bad_rows:,} rows with invalid MthCalDt.")
    assert_unique_permno_month(keys)
    return keys


def example_values(series: pd.Series, limit: int = 5) -> str:
    return "; ".join(series.dropna().astype("string").drop_duplicates().head(limit).tolist())


def pct_nonmissing(series: pd.Series) -> float:
    return round(100 * series.notna().mean(), 4) if len(series) else 0.0


def build_all_compustat_like_table(
    data: pd.DataFrame,
    compustat_columns: list[str],
    model_mask: pd.Series,
) -> pd.DataFrame:
    rows = []
    for column in compustat_columns:
        series = data[column]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "exclusion_reason": exclusion_reason(column) or "",
                "nonmissing_pct_full_sample": pct_nonmissing(series),
                "nonmissing_pct_modeling_sample": pct_nonmissing(series.loc[model_mask]),
                "example_nonmissing_values": example_values(series),
            }
        )
    return pd.DataFrame(rows)


def numeric_accounting_candidates(
    data: pd.DataFrame,
    compustat_columns: list[str],
    selection_mask: pd.Series,
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    selected = []
    for column in compustat_columns:
        reason = exclusion_reason(column)
        series = data[column]
        numeric = pd.to_numeric(series, errors="coerce") if reason is None else None
        full_nonmissing = int(series.notna().sum())
        if numeric is None:
            model_numeric_nonmissing = pd.NA
            numeric_parse_rate = pd.NA
            model_missing_pct = round(100 * series.loc[selection_mask].isna().mean(), 4)
            variance = pd.NA
            unique_values = pd.NA
            candidate = False
        else:
            selection_series = series.loc[selection_mask]
            selection_numeric = numeric.loc[selection_mask]
            original_nonmissing = int(selection_series.notna().sum())
            numeric_nonmissing = int(selection_numeric.notna().sum())
            numeric_parse_rate = numeric_nonmissing / original_nonmissing if original_nonmissing else 0.0
            model_numeric = selection_numeric
            model_numeric_nonmissing = int(model_numeric.notna().sum())
            model_missing_pct = round(100 * model_numeric.isna().mean(), 4) if len(model_numeric) else 100.0
            variance = model_numeric.var(skipna=True)
            unique_values = int(model_numeric.nunique(dropna=True))
            candidate = numeric_parse_rate >= 0.995
            if candidate:
                selected.append(column)
        rows.append(
            {
                "feature": column,
                "raw_column": column,
                "feature_type": "raw_compustat_column",
                "source_columns": column,
                "initial_candidate": reason is None,
                "numeric_candidate": candidate,
                "rows": len(series),
                "full_nonmissing": full_nonmissing,
                "full_nonmissing_pct": pct_nonmissing(series),
                "modeling_nonmissing": model_numeric_nonmissing,
                "modeling_missing_pct": model_missing_pct,
                "numeric_parse_rate": round(numeric_parse_rate, 6) if not pd.isna(numeric_parse_rate) else pd.NA,
                "modeling_unique_values": unique_values,
                "modeling_variance": variance,
                "selected": False,
                "status": "raw_available_for_engineering" if candidate else "excluded",
                "exclusion_reason": reason or ("" if candidate else "not_safely_numeric"),
                "selection_note": "raw level not selected directly; transformed or ratio features are preferred",
            }
        )
    return pd.DataFrame(rows), selected


def to_numeric_frame(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return data[columns].apply(pd.to_numeric, errors="coerce").astype("float64")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.where(denominator.abs() > EPS)
    return numerator / denom


def log_positive(series: pd.Series) -> pd.Series:
    return np.log(series.where(series > 0))


def add_feature(
    features: dict[str, pd.Series],
    statuses: list[dict[str, object]],
    name: str,
    feature_type: str,
    required: list[str],
    available: set[str],
    builder: Callable[[], pd.Series],
    description: str,
) -> None:
    missing = [column for column in required if column not in available]
    if missing:
        statuses.append(
            {
                "feature": name,
                "feature_type": feature_type,
                "required_columns": ",".join(required),
                "created": False,
                "skipped_reason": f"missing required columns: {','.join(missing)}",
                "description": description,
            }
        )
        return
    values = builder().replace([np.inf, -np.inf], np.nan).astype("float64")
    features[name] = values
    statuses.append(
        {
            "feature": name,
            "feature_type": feature_type,
            "required_columns": ",".join(required),
            "created": True,
            "skipped_reason": "",
            "description": description,
        }
    )


def build_engineered_compustat_features(
    numeric: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features: dict[str, pd.Series] = {}
    statuses: list[dict[str, object]] = []
    available = set(numeric.columns)
    n = numeric

    add_feature(
        features,
        statuses,
        "acct_log_sales",
        "transformed_level",
        ["comp_saley"],
        available,
        lambda: log_positive(n["comp_saley"]),
        "log sales; size proxy because total assets are unavailable in the current panel",
    )
    add_feature(
        features,
        statuses,
        "acct_log_revenue",
        "transformed_level",
        ["comp_revty"],
        available,
        lambda: log_positive(n["comp_revty"]),
        "log revenue; alternative size proxy",
    )
    add_feature(
        features,
        statuses,
        "acct_log_common_shares",
        "transformed_level",
        ["comp_cshpry"],
        available,
        lambda: log_positive(n["comp_cshpry"]),
        "log common shares used in EPS calculation; share-base scale proxy",
    )
    add_feature(
        features,
        statuses,
        "acct_epspxy",
        "per_share_raw",
        ["comp_epspxy"],
        available,
        lambda: n["comp_epspxy"],
        "earnings per share, already scale-adjusted",
    )

    ratio_specs: list[tuple[str, list[str], Callable[[], pd.Series], str]] = [
        (
            "acct_gross_margin",
            ["comp_saley", "comp_cogsy"],
            lambda: safe_divide(n["comp_saley"] - n["comp_cogsy"], n["comp_saley"]),
            "gross profit over sales",
        ),
        ("acct_cogs_to_sales", ["comp_cogsy", "comp_saley"], lambda: safe_divide(n["comp_cogsy"], n["comp_saley"]), "cost of goods sold over sales"),
        ("acct_operating_margin", ["comp_oiadpy", "comp_saley"], lambda: safe_divide(n["comp_oiadpy"], n["comp_saley"]), "operating income after depreciation over sales"),
        ("acct_ebitda_margin", ["comp_oibdpy", "comp_saley"], lambda: safe_divide(n["comp_oibdpy"], n["comp_saley"]), "operating income before depreciation over sales"),
        ("acct_net_income_margin", ["comp_niy", "comp_saley"], lambda: safe_divide(n["comp_niy"], n["comp_saley"]), "net income over sales"),
        ("acct_income_before_extra_margin", ["comp_ibcy", "comp_saley"], lambda: safe_divide(n["comp_ibcy"], n["comp_saley"]), "income before extraordinary items over sales"),
        ("acct_pretax_margin", ["comp_piy", "comp_saley"], lambda: safe_divide(n["comp_piy"], n["comp_saley"]), "pretax income over sales"),
        ("acct_operating_cf_margin", ["comp_oancfy", "comp_saley"], lambda: safe_divide(n["comp_oancfy"], n["comp_saley"]), "operating cash flow over sales"),
        ("acct_free_cash_flow_margin", ["comp_oancfy", "comp_capxy", "comp_saley"], lambda: safe_divide(n["comp_oancfy"] - n["comp_capxy"], n["comp_saley"]), "operating cash flow minus capex over sales"),
        ("acct_total_expense_to_sales", ["comp_xopry", "comp_saley"], lambda: safe_divide(n["comp_xopry"], n["comp_saley"]), "operating expense over sales"),
        ("acct_sga_to_sales", ["comp_xsgay", "comp_saley"], lambda: safe_divide(n["comp_xsgay"], n["comp_saley"]), "selling, general, and administrative expense over sales"),
        ("acct_rd_to_sales", ["comp_xrdy", "comp_saley"], lambda: safe_divide(n["comp_xrdy"], n["comp_saley"]), "R&D expense over sales"),
        ("acct_capex_to_sales", ["comp_capxy", "comp_saley"], lambda: safe_divide(n["comp_capxy"], n["comp_saley"]), "capital expenditure over sales"),
        ("acct_interest_to_sales", ["comp_xinty", "comp_saley"], lambda: safe_divide(n["comp_xinty"], n["comp_saley"]), "interest expense over sales"),
        ("acct_tax_to_sales", ["comp_txty", "comp_saley"], lambda: safe_divide(n["comp_txty"], n["comp_saley"]), "income taxes over sales"),
        ("acct_dividends_to_sales", ["comp_dvy", "comp_saley"], lambda: safe_divide(n["comp_dvy"], n["comp_saley"]), "cash dividends over sales"),
        ("acct_preferred_dividends_to_sales", ["comp_dvpy", "comp_saley"], lambda: safe_divide(n["comp_dvpy"], n["comp_saley"]), "preferred dividends over sales"),
        ("acct_dividends_to_earnings", ["comp_dvy", "comp_niy"], lambda: safe_divide(n["comp_dvy"], n["comp_niy"]), "cash dividends over earnings"),
        ("acct_stock_issuance_to_sales", ["comp_sstky", "comp_saley"], lambda: safe_divide(n["comp_sstky"], n["comp_saley"]), "sale of common/preferred stock over sales"),
        ("acct_stock_repurchases_to_sales", ["comp_prstkcy", "comp_saley"], lambda: safe_divide(n["comp_prstkcy"], n["comp_saley"]), "stock repurchases over sales"),
        ("acct_long_debt_issuance_to_sales", ["comp_dltisy", "comp_saley"], lambda: safe_divide(n["comp_dltisy"], n["comp_saley"]), "long-term debt issuance over sales"),
        ("acct_debt_reduction_to_sales", ["comp_dltry", "comp_saley"], lambda: safe_divide(n["comp_dltry"], n["comp_saley"]), "long-term debt reduction over sales"),
        ("acct_short_debt_change_to_sales", ["comp_dlcchy", "comp_saley"], lambda: safe_divide(n["comp_dlcchy"], n["comp_saley"]), "current-debt change over sales"),
        ("acct_cash_change_to_sales", ["comp_chechy", "comp_saley"], lambda: safe_divide(n["comp_chechy"], n["comp_saley"]), "cash-change flow over sales"),
        ("acct_receivables_change_to_sales", ["comp_recchy", "comp_saley"], lambda: safe_divide(n["comp_recchy"], n["comp_saley"]), "accounts-receivable change over sales"),
        ("acct_inventory_change_to_sales", ["comp_invchy", "comp_saley"], lambda: safe_divide(n["comp_invchy"], n["comp_saley"]), "inventory change over sales"),
        ("acct_acquisition_to_sales", ["comp_aqcy", "comp_saley"], lambda: safe_divide(n["comp_aqcy"], n["comp_saley"]), "acquisitions over sales"),
        ("acct_working_capital_change_to_sales", ["comp_wcapcy", "comp_saley"], lambda: safe_divide(n["comp_wcapcy"], n["comp_saley"]), "working-capital change over sales"),
        ("acct_investing_cf_to_sales", ["comp_ivncfy", "comp_saley"], lambda: safe_divide(n["comp_ivncfy"], n["comp_saley"]), "investing cash flow over sales"),
        ("acct_financing_cf_to_sales", ["comp_fincfy", "comp_saley"], lambda: safe_divide(n["comp_fincfy"], n["comp_saley"]), "financing cash flow over sales"),
        ("acct_accruals_to_sales", ["comp_niy", "comp_oancfy", "comp_saley"], lambda: safe_divide(n["comp_niy"] - n["comp_oancfy"], n["comp_saley"]), "net income minus operating cash flow over sales"),
    ]
    for name, required, builder, description in ratio_specs:
        add_feature(features, statuses, name, "constructed_ratio", required, available, builder, description)

    requested_specs: list[tuple[str, list[str], str]] = [
        ("requested_operating_income_to_assets", ["comp_oiadpy", "comp_atq"], "operating income over assets"),
        ("requested_net_income_to_assets", ["comp_niy", "comp_atq"], "net income over assets"),
        ("requested_gross_profit_to_assets", ["comp_saley", "comp_cogsy", "comp_atq"], "gross profit over assets"),
        ("requested_debt_to_assets", ["comp_dlcq", "comp_dlttq", "comp_atq"], "debt over assets"),
        ("requested_debt_to_equity", ["comp_dlcq", "comp_dlttq", "comp_ceqq"], "debt over equity"),
        ("requested_cash_to_assets", ["comp_cheq", "comp_atq"], "cash over assets"),
        ("requested_current_ratio", ["comp_actq", "comp_lctq"], "current assets over current liabilities"),
        ("requested_asset_growth", ["comp_atq"], "asset growth"),
        ("requested_capex_to_assets", ["comp_capxy", "comp_atq"], "capex over assets"),
        ("requested_log_assets", ["comp_atq"], "log assets"),
        ("requested_book_to_market", ["comp_ceqq", "market_equity"], "book equity over market equity"),
        ("requested_dividends_to_assets", ["comp_dvy", "comp_atq"], "dividends over assets"),
        ("requested_accruals_to_assets", ["comp_niy", "comp_oancfy", "comp_atq"], "accruals over assets"),
    ]
    for name, required, description in requested_specs:
        add_feature(
            features,
            statuses,
            name,
            "requested_ratio_unavailable",
            required,
            available,
            lambda: pd.Series(np.nan, index=n.index, dtype="float64"),
            description,
        )
        if name in features:
            features.pop(name)

    feature_frame = pd.DataFrame(features, index=n.index)
    return feature_frame, pd.DataFrame(statuses)


def redundant_columns(feature_data: pd.DataFrame, columns: list[str], mask: pd.Series) -> list[str]:
    if len(columns) < 2:
        return []
    corr = feature_data.loc[mask, columns].corr()
    redundant: list[str] = []
    for i, left in enumerate(columns):
        if left in redundant:
            continue
        for right in columns[i + 1 :]:
            if right in redundant:
                continue
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) >= 0.999999999:
                redundant.append(right)
    return redundant


def select_engineered_features(
    feature_data: pd.DataFrame,
    ratio_status: pd.DataFrame,
    selection_mask: pd.Series,
    train_diag_mask: pd.Series,
    missingness_threshold: float,
    near_zero_variance_threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    status_lookup = ratio_status.set_index("feature").to_dict(orient="index")
    rows = []
    selected = []
    for feature in feature_data.columns:
        values = feature_data[feature]
        model_values = values.loc[selection_mask]
        train_values = values.loc[train_diag_mask]
        missing_rate = model_values.isna().mean() if len(model_values) else 1.0
        variance = model_values.var(skipna=True)
        unique_values = int(model_values.nunique(dropna=True))
        selected_flag = True
        exclusion = ""
        if missing_rate >= missingness_threshold:
            selected_flag = False
            exclusion = f"modeling_missingness_ge_{missingness_threshold:.0%}"
        elif unique_values <= 1:
            selected_flag = False
            exclusion = "near_zero_variance"
        elif pd.isna(variance) or variance <= near_zero_variance_threshold:
            selected_flag = False
            exclusion = "near_zero_variance"

        if selected_flag:
            selected.append(feature)

        meta = status_lookup.get(feature, {})
        rows.append(
            {
                "feature": feature,
                "raw_column": "",
                "feature_type": meta.get("feature_type", "engineered_feature"),
                "source_columns": meta.get("required_columns", ""),
                "initial_candidate": True,
                "numeric_candidate": True,
                "rows": len(values),
                "full_nonmissing": int(values.notna().sum()),
                "full_nonmissing_pct": pct_nonmissing(values),
                "modeling_nonmissing": int(model_values.notna().sum()),
                "modeling_missing_pct": round(100 * missing_rate, 4),
                "numeric_parse_rate": 1.0,
                "modeling_unique_values": unique_values,
                "modeling_variance": variance,
                "train_diag_missing_pct": round(100 * train_values.isna().mean(), 4) if len(train_values) else pd.NA,
                "train_diag_variance": train_values.var(skipna=True),
                "selected": selected_flag,
                "status": "selected" if selected_flag else "excluded",
                "exclusion_reason": exclusion,
                "selection_note": meta.get("description", ""),
            }
        )

    redundant = redundant_columns(feature_data, selected, selection_mask)
    if redundant:
        redundant_set = set(redundant)
        for row in rows:
            if row["feature"] in redundant_set and row["selected"]:
                row["selected"] = False
                row["status"] = "excluded"
                row["exclusion_reason"] = "duplicate_or_perfectly_correlated"
        selected = [feature for feature in selected if feature not in redundant_set]

    return pd.DataFrame(rows), selected


def selected_feature_metadata(
    diagnostics: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    selected = diagnostics.loc[diagnostics["feature"].isin(selected_features)].copy()
    selected["coverage_pct_modeling_sample"] = 100 - selected["modeling_missing_pct"]
    selected["missing_indicator"] = ""
    selected = selected.sort_values(
        ["feature_type", "coverage_pct_modeling_sample", "feature"],
        ascending=[True, False, True],
    )
    return selected[
        [
            "feature",
            "feature_type",
            "source_columns",
            "coverage_pct_modeling_sample",
            "modeling_missing_pct",
            "train_diag_missing_pct",
            "missing_indicator",
            "selection_note",
        ]
    ].reset_index(drop=True)


def choose_missing_indicators(
    diagnostics: pd.DataFrame,
    selected_features: list[str],
) -> list[str]:
    selected = diagnostics.loc[diagnostics["feature"].isin(selected_features)].copy()
    selected = selected.loc[selected["modeling_missing_pct"].ge(100 * MISSING_INDICATOR_MIN_RATE)]
    selected = selected.sort_values(["modeling_missing_pct", "feature"], ascending=[False, True])
    return selected["feature"].head(MAX_MISSING_INDICATORS).tolist()


def clean_selected_features(
    keys: pd.DataFrame,
    feature_data: pd.DataFrame,
    selected_features: list[str],
    indicator_source_features: list[str],
    winsor_lower: float,
    winsor_upper: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert_no_target_leakage(selected_features, "Compustat cleaning")
    working = pd.concat([keys[KEY_COLUMNS], feature_data[selected_features]], axis=1)
    cleaned = working[KEY_COLUMNS].copy()
    missing_rows = []
    full_missing_rows = []
    grouped = working.groupby("mthcaldt", sort=False)

    for feature in selected_features:
        raw = working[feature].astype("float64")
        lower = grouped[feature].transform(lambda x: x.quantile(winsor_lower))
        upper = grouped[feature].transform(lambda x: x.quantile(winsor_upper))
        winsorized = raw.clip(lower=lower, upper=upper)
        medians = winsorized.groupby(working["mthcaldt"], sort=False).transform("median")
        imputed = winsorized.fillna(medians)
        cleaned[feature] = imputed

        full_missing_months = (
            working.groupby("mthcaldt", sort=True)[feature]
            .apply(lambda x: x.isna().all())
            .loc[lambda x: x]
            .index
        )
        for month in full_missing_months:
            full_missing_rows.append({"feature": feature, "mthcaldt": month})

        before_missing = int(raw.isna().sum())
        after_missing = int(imputed.isna().sum())
        missing_rows.append(
            {
                "feature": feature,
                "rows": len(raw),
                "missing_before": before_missing,
                "missing_before_pct": round(100 * before_missing / len(raw), 4) if len(raw) else 0.0,
                "missing_after": after_missing,
                "missing_after_pct": round(100 * after_missing / len(raw), 4) if len(raw) else 0.0,
                "full_missing_months_after_monthly_imputation": int(len(full_missing_months)),
            }
        )

    for feature in indicator_source_features:
        indicator = f"{feature}_missing"
        raw = working[feature].astype("float64")
        cleaned[indicator] = raw.isna().astype("int8")

    full_missing = pd.DataFrame(full_missing_rows)
    if full_missing.empty:
        full_missing = pd.DataFrame(columns=["feature", "mthcaldt"])

    return cleaned, pd.DataFrame(missing_rows), full_missing


def final_missingness(
    input_path: Path,
    input_feature_columns: list[str],
    cleaned_features: pd.DataFrame,
    cleaned_feature_columns: list[str],
    indicator_columns: list[str],
    batch_size: int,
) -> pd.DataFrame:
    input_missing = pd.Series(0, index=input_feature_columns, dtype="int64")
    row_count = 0
    input_file = pq.ParquetFile(input_path)
    for batch in input_file.iter_batches(columns=input_feature_columns, batch_size=batch_size):
        frame = batch.to_pandas()
        input_missing = input_missing.add(frame.isna().sum(), fill_value=0).astype("int64")
        row_count += len(frame)

    rows = []
    for column in input_feature_columns:
        missing = int(input_missing[column])
        rows.append(
            {
                "feature": column,
                "group": "pass_through",
                "rows": row_count,
                "nonmissing": row_count - missing,
                "missing": missing,
                "missing_pct": round(100 * missing / row_count, 4) if row_count else 0.0,
            }
        )
    for column in cleaned_feature_columns + indicator_columns:
        missing = int(cleaned_features[column].isna().sum())
        rows.append(
            {
                "feature": column,
                "group": "selected_compustat",
                "rows": len(cleaned_features),
                "nonmissing": len(cleaned_features) - missing,
                "missing": missing,
                "missing_pct": round(100 * missing / len(cleaned_features), 4) if len(cleaned_features) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_final_panel(
    input_path: Path,
    output_path: Path,
    pass_through_columns: list[str],
    cleaned_features: pd.DataFrame,
    cleaned_feature_columns: list[str],
    indicator_columns: list[str],
    batch_size: int,
) -> int:
    lookup_columns = cleaned_feature_columns + indicator_columns
    input_file = pq.ParquetFile(input_path)
    temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if temp_path.exists():
        temp_path.unlink()

    writer: pq.ParquetWriter | None = None
    rows_written = 0
    try:
        for batch in input_file.iter_batches(columns=pass_through_columns, batch_size=batch_size):
            frame = batch.to_pandas()
            frame["mthcaldt"] = pd.to_datetime(frame["mthcaldt"], errors="coerce")
            mapped = cleaned_features.iloc[rows_written : rows_written + len(frame)].reset_index(
                drop=True
            )
            if not frame[KEY_COLUMNS].reset_index(drop=True).equals(
                mapped[KEY_COLUMNS].reset_index(drop=True)
            ):
                raise ValueError("Cleaned Compustat features are not row-aligned with the input panel.")
            for column in lookup_columns:
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


def build_summary(
    input_rows: int,
    output_rows: int,
    total_compustat_like: int,
    candidate_count: int,
    constructed_count: int,
    engineered_count: int,
    selected_count: int,
    return_feature_count: int,
    jkp_feature_count: int,
    final_feature_count: int,
    duplicate_count: int,
    warnings: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "input_rows": input_rows,
                "output_rows": output_rows,
                "row_count_changed": input_rows != output_rows,
                "total_compustat_like_columns": total_compustat_like,
                "compustat_candidate_features_after_exclusions": candidate_count,
                "constructed_ratios_created": constructed_count,
                "engineered_compustat_features_created": engineered_count,
                "selected_compustat_features": selected_count,
                "return_features_preserved": return_feature_count,
                "jkp_features_preserved": jkp_feature_count,
                "final_total_feature_count": final_feature_count,
                "duplicate_permno_mthcaldt_rows": duplicate_count,
                "warnings": " | ".join(warnings),
            }
        ]
    )


def write_feature_groups(
    path: Path,
    identifier_columns: list[str],
    target_columns: list[str],
    raw_return_columns: list[str],
    return_feature_columns: list[str],
    jkp_feature_columns: list[str],
    selected_compustat_features: list[str],
    missing_indicator_columns: list[str],
    excluded_columns: list[str],
    args: argparse.Namespace,
) -> None:
    groups = {
        "identifier_columns": identifier_columns,
        "target_columns": target_columns,
        "raw_return_columns": raw_return_columns,
        "return_feature_columns": return_feature_columns,
        "jkp_feature_columns": jkp_feature_columns,
        "selected_compustat_feature_columns": selected_compustat_features,
        "selected_compustat_missing_indicator_columns": missing_indicator_columns,
        "excluded_columns": excluded_columns,
        "cleaning_rules": {
            "candidate_detection": "comp_* and compustat_* columns excluding identifiers, dates, categorical metadata, and linking helpers",
            "feature_selection_window_start": TRAIN_SELECTION_START.date().isoformat(),
            "feature_selection_window_end": TRAIN_SELECTION_END.date().isoformat(),
            "missingness_threshold": args.missingness_threshold,
            "near_zero_variance_threshold": args.near_zero_variance_threshold,
            "winsorization": f"cross-sectional by mthcaldt at {args.winsor_lower:.2%}/{args.winsor_upper:.2%}",
            "imputation": "cross-sectional median by mthcaldt; full-missing months remain missing",
            "missing_indicators": f"added for up to {MAX_MISSING_INDICATORS} selected Compustat features with training-window missingness >= {MISSING_INDICATOR_MIN_RATE:.0%}",
            "raw_level_policy": "prefer constructed ratios; keep transformed levels only for size proxies and per-share EPS",
            "target_leakage_guard": f"{TARGET_COLUMNS} are excluded from selection, winsorization, imputation, and scaling",
        },
    }
    path.write_text(json.dumps(groups, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing input panel: {args.input}")

    columns = parquet_columns(args.input)
    input_rows = parquet_row_count(args.input)
    required = set(KEY_COLUMNS + TARGET_COLUMNS + RAW_RETURN_COLUMNS)
    missing_required = required.difference(columns)
    if missing_required:
        raise ValueError(f"Input panel is missing required columns: {sorted(missing_required)}")

    modeling_start = pd.Timestamp(args.modeling_start_date)

    identifier_columns = present(columns, IDENTIFIER_COLUMNS)
    target_columns = present(columns, TARGET_COLUMNS)
    raw_return_columns = present(columns, RAW_RETURN_COLUMNS)
    return_feature_columns = present(columns, RETURN_FEATURE_COLUMNS)
    jkp_feature_columns = [column for column in columns if column.startswith("jkp_")]
    compustat_columns = compustat_like_columns(columns)
    assert_no_target_leakage(compustat_columns, "Compustat feature selection")

    print(f"Reading panel keys: {args.input}", flush=True)
    keys = read_panel_keys(args.input)
    model_mask = keys["mthcaldt"].ge(modeling_start)
    train_selection_mask = keys["mthcaldt"].between(
        TRAIN_SELECTION_START,
        TRAIN_SELECTION_END,
        inclusive="both",
    )
    train_diag_mask = train_selection_mask
    print(f"Input rows: {len(keys):,}", flush=True)
    print("Duplicate PERMNO-MthCalDt rows in input: 0", flush=True)
    print(f"Training-window rows for selection: {int(train_selection_mask.sum()):,}", flush=True)

    print("Reading Compustat-like columns for diagnostics and engineering...", flush=True)
    compustat_data = pd.read_parquet(args.input, columns=compustat_columns)
    all_compustat_like = build_all_compustat_like_table(compustat_data, compustat_columns, model_mask)
    df_train = compustat_data.loc[train_selection_mask]
    if df_train.empty:
        raise ValueError(
            f"No rows found in the feature-selection training window "
            f"{TRAIN_SELECTION_START.date()} to {TRAIN_SELECTION_END.date()}."
        )
    raw_diagnostics, numeric_raw_columns = numeric_accounting_candidates(
        compustat_data, compustat_columns, train_selection_mask
    )
    numeric = to_numeric_frame(compustat_data, numeric_raw_columns)
    engineered, ratio_status = build_engineered_compustat_features(numeric)
    engineered_diagnostics, selected_features = select_engineered_features(
        engineered,
        ratio_status,
        train_selection_mask,
        train_diag_mask,
        args.missingness_threshold,
        args.near_zero_variance_threshold,
    )
    diagnostics = pd.concat([raw_diagnostics, engineered_diagnostics], ignore_index=True)

    created_engineered_count = int(
        (
            ratio_status["created"].fillna(False)
            & ratio_status["feature_type"].isin(["constructed_ratio", "transformed_level", "per_share_raw"])
        ).sum()
    )
    created_ratio_count = int(
        (
            ratio_status["created"].fillna(False)
            & ratio_status["feature_type"].eq("constructed_ratio")
        ).sum()
    )
    candidate_count = len(numeric_raw_columns)
    selected_count = len(selected_features)
    if selected_count == 0:
        raise ValueError("No Compustat accounting features survived revised selection.")

    warnings: list[str] = []
    if selected_count < 20:
        warnings.append(
            f"Selected Compustat feature count is {selected_count}, below the desired 20-80 range; current input exposes only {len(compustat_columns)} Compustat-like columns and lacks assets/debt/equity levels."
        )
    unavailable_requested = ratio_status.loc[
        ratio_status["feature_type"].eq("requested_ratio_unavailable") & ~ratio_status["created"]
    ]
    if not unavailable_requested.empty:
        warnings.append(
            f"{len(unavailable_requested)} requested asset/debt/equity-style ratios were skipped because required raw columns are absent from the current panel."
        )

    print(
        f"Compustat-like columns: {len(compustat_columns):,}; numeric raw candidates: {candidate_count:,}; "
        f"constructed ratios created: {created_ratio_count:,}; "
        f"engineered accounting features created: {created_engineered_count:,}; "
        f"selected features: {selected_count:,}",
        flush=True,
    )

    indicator_source_features = choose_missing_indicators(engineered_diagnostics, selected_features)
    indicator_columns = [f"{feature}_missing" for feature in indicator_source_features]
    selected_table = selected_feature_metadata(engineered_diagnostics, selected_features)
    selected_table.loc[
        selected_table["feature"].isin(indicator_source_features), "missing_indicator"
    ] = selected_table.loc[
        selected_table["feature"].isin(indicator_source_features), "feature"
    ].map(lambda x: f"{x}_missing")

    print("Winsorizing and imputing selected Compustat features by month...", flush=True)
    cleaned_features, missing_before_after, full_missing_months = clean_selected_features(
        keys,
        engineered,
        selected_features,
        indicator_source_features,
        args.winsor_lower,
        args.winsor_upper,
    )

    entirely_missing = [column for column in selected_features if cleaned_features[column].isna().all()]
    if entirely_missing:
        raise ValueError(f"Selected cleaned features are entirely missing: {entirely_missing}")

    pass_through_columns = (
        identifier_columns
        + target_columns
        + raw_return_columns
        + return_feature_columns
        + jkp_feature_columns
    )
    seen = set()
    pass_through_columns = [c for c in pass_through_columns if not (c in seen or seen.add(c))]
    excluded_columns = [column for column in columns if column not in pass_through_columns]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("Writing final modeling panel in parquet batches...", flush=True)
    output_rows = write_final_panel(
        args.input,
        args.output,
        pass_through_columns,
        cleaned_features,
        selected_features,
        indicator_columns,
        args.batch_size,
    )
    if output_rows != input_rows:
        raise ValueError(f"Row count changed after final write: {input_rows:,} -> {output_rows:,}")

    output_keys = pd.read_parquet(args.output, columns=KEY_COLUMNS)
    duplicate_after = int(output_keys.duplicated(KEY_COLUMNS).sum())
    if duplicate_after:
        raise ValueError(f"Final panel has {duplicate_after:,} duplicate PERMNO-MthCalDt rows.")

    output_columns = parquet_columns(args.output)
    if len(present(output_columns, return_feature_columns)) != len(return_feature_columns):
        raise ValueError("Not all return features were preserved in final panel.")
    if len([c for c in output_columns if c.startswith("jkp_")]) != len(jkp_feature_columns):
        raise ValueError("Not all JKP features were preserved in final panel.")

    final_feature_columns = (
        raw_return_columns + return_feature_columns + jkp_feature_columns + selected_features + indicator_columns
    )
    final_feature_count = len(final_feature_columns)

    args.table_dir.mkdir(parents=True, exist_ok=True)
    all_compustat_like_path = args.table_dir / "all_compustat_like_columns.csv"
    diagnostics_path = args.table_dir / "compustat_candidate_diagnostics.csv"
    ratio_status_path = args.table_dir / "constructed_compustat_ratios_status.csv"
    missing_path = args.table_dir / "compustat_missingness_before_after.csv"
    full_missing_months_path = args.table_dir / "compustat_full_missing_months.csv"
    selected_path = args.table_dir / "selected_compustat_features.csv"
    groups_path = args.table_dir / "feature_groups.json"
    final_missingness_path = args.table_dir / "final_feature_missingness.csv"
    summary_path = args.table_dir / "model_panel_full_features_summary.csv"

    all_compustat_like.to_csv(all_compustat_like_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    ratio_status.to_csv(ratio_status_path, index=False)
    missing_before_after.to_csv(missing_path, index=False)
    full_missing_months.to_csv(full_missing_months_path, index=False)
    selected_table.to_csv(selected_path, index=False)
    write_feature_groups(
        groups_path,
        identifier_columns,
        target_columns,
        raw_return_columns,
        return_feature_columns,
        jkp_feature_columns,
        selected_features,
        indicator_columns,
        excluded_columns,
        args,
    )
    final_missingness(
        args.input,
        raw_return_columns + return_feature_columns + jkp_feature_columns,
        cleaned_features,
        selected_features,
        indicator_columns,
        args.batch_size,
    ).to_csv(final_missingness_path, index=False)
    build_summary(
        input_rows,
        output_rows,
        len(compustat_columns),
        candidate_count,
        created_ratio_count,
        created_engineered_count,
        selected_count,
        len(return_feature_columns),
        len(jkp_feature_columns),
        final_feature_count,
        duplicate_after,
        warnings,
    ).to_csv(summary_path, index=False)

    print(f"Output rows: {output_rows:,}", flush=True)
    print(f"Return features preserved: {len(return_feature_columns):,}", flush=True)
    print(f"JKP features preserved: {len(jkp_feature_columns):,}", flush=True)
    print(f"Final total feature count: {final_feature_count:,}", flush=True)
    print(f"Duplicate PERMNO-MthCalDt rows in output: {duplicate_after:,}", flush=True)
    if warnings:
        print("Warnings:", flush=True)
        for warning in warnings:
            print(f"- {warning}", flush=True)
    print(f"Wrote final modeling panel: {args.output}", flush=True)
    print(f"Wrote all Compustat-like column diagnostics: {all_compustat_like_path}", flush=True)
    print(f"Wrote Compustat candidate diagnostics: {diagnostics_path}", flush=True)
    print(f"Wrote constructed ratio status: {ratio_status_path}", flush=True)
    print(f"Wrote Compustat missingness before/after: {missing_path}", flush=True)
    print(f"Wrote full-missing Compustat month log: {full_missing_months_path}", flush=True)
    print(f"Wrote selected Compustat features: {selected_path}", flush=True)
    print(f"Wrote feature groups metadata: {groups_path}", flush=True)
    print(f"Wrote final feature missingness: {final_missingness_path}", flush=True)
    print(f"Wrote final panel summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
