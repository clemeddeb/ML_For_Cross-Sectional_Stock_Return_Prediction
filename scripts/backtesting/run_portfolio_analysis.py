#!/usr/bin/env python3
"""Run raw score-based portfolio analysis at a fixed transaction cost.

This script replaces the exploratory portfolio-analysis notebooks with a
reproducible pipeline. It evaluates raw model-score portfolios only. It does
not implement volatility scaling, drawdown de-risking, or any other risk
overlay.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Path("/tmp/matplotlib-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

plt.style.use("default")
plt.rcParams.update(
    {
        "figure.figsize": (10, 5),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


REQUIRED_COLUMNS = ["PERMNO", "MthCalDt", "split", "target_ret_1m"]
TARGET_AND_ID_COLUMNS = {
    "PERMNO",
    "permno",
    "gvkey",
    "GVKEY",
    "MthCalDt",
    "mthcaldt",
    "month",
    "date",
    "split",
    "target_ret_1m",
    "target_quintile",
    "top_bottom_label",
}
EXPECTED_PREDICTION_FILES = [
    "baseline_predictions.parquet",
    "baseline_predictions_with_boosting.parquet",
    "boosting_predictions.parquet",
    "dl_xgb_predictions.parquet",
    "ft_transformer_predictions.parquet",
    "temporal_tabular_backtest_predictions.parquet",
    "temporal_tabular_transformer_seed362559_predictions.parquet",
]
SPEC_COLUMNS = [
    "spec_id",
    "source_file",
    "score_label",
    "score_column",
    "model_family",
    "rule",
    "q",
    "gate_type",
    "threshold",
    "z_threshold",
    "one_way_cost_bps",
]
SPLITS = ["validation", "test"]
BENCHMARK_TICKER = "SPY"
BASELINE_FAMILIES = ["Naive Momentum", "XGBoost"]
COMPARISON_MODEL_FAMILIES = ["Naive Momentum", "XGBoost", "MLP", "FT-Transformer", "TTT", "Ensemble"]


@dataclass(frozen=True)
class ScoreSpec:
    source_file: str
    score_label: str
    score_column: str
    model_family: str
    sign_meaningful: bool


@dataclass(frozen=True)
class StrategySpec:
    rule: str
    q: float | None
    gate_type: str
    threshold: float | None
    z_threshold: float | None
    one_way_cost_bps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw score portfolio analysis.")
    parser.add_argument("--prediction-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/backtests"))
    parser.add_argument("--cost-bps", type=float, default=25.0)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.02])
    parser.add_argument("--z-thresholds", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.00])
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.10, 0.20])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-specs", type=int, default=None)
    parser.add_argument("--min-rank-ic", type=float, default=0.0)
    parser.add_argument("--min-rank-ic-tstat", type=float, default=1.96)
    parser.add_argument("--min-max-dd", type=float, default=-0.40)
    parser.add_argument("--max-turnover", type=float, default=1.75)
    return parser.parse_args()


def log(message: str, debug: bool = True) -> None:
    if debug:
        print(message, flush=True)


def import_yfinance() -> Any:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for the SPY benchmark. Install it with:\n"
            "  python3 -m pip install yfinance"
        ) from exc
    return yf


def safe_slug(text: str) -> str:
    out = []
    for char in str(text).lower():
        if char.isalnum():
            out.append(char)
        elif char in {"_", "-", "."}:
            out.append("_")
    return "".join(out).strip("_")


def family_from_source_and_score(source_file: str, score_label: str) -> str:
    text = f"{source_file} {score_label}".lower()
    if "naive_momentum" in text:
        return "Naive Momentum"
    if "naive_reversal" in text:
        return "Naive Reversal"
    if "temporal_tabular" in text or "temporal" in text or "ttt" in text:
        return "TTT"
    if "ft_" in text or "ft-" in text or "ft_transformer" in text:
        return "FT-Transformer"
    if "mlp" in text:
        return "MLP"
    if "xgb" in text:
        return "XGBoost"
    if "ensemble" in text:
        return "Ensemble"
    if any(token in text for token in ["ridge", "elastic", "logistic", "naive", "gb", "boosting"]):
        return "Baseline"
    return "Other"


def score_label_from_column(source_file: str, column: str) -> str:
    stem = Path(source_file).stem.replace("_predictions", "")
    label = column
    for prefix in ["prediction_", "score_"]:
        if label.startswith(prefix):
            label = label[len(prefix) :]
    return f"{stem}__{label}"


def is_score_column(column: str, series: pd.Series) -> bool:
    if column in TARGET_AND_ID_COLUMNS:
        return False
    if column.startswith("prob_"):
        return False
    if not (column.startswith("prediction_") or column.startswith("score_")):
        return False
    if not pd.api.types.is_numeric_dtype(series):
        return False
    return series.notna().any()


def discover_prediction_files(prediction_dir: Path, warnings: list[str]) -> list[Path]:
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")
    found = sorted(prediction_dir.glob("*.parquet"))
    names = {path.name for path in found}
    for expected in EXPECTED_PREDICTION_FILES:
        if expected not in names:
            warnings.append(f"Expected prediction file is missing: {prediction_dir / expected}")
    if not found:
        raise FileNotFoundError(f"No parquet prediction files found in {prediction_dir}")
    names = {path.name for path in found}
    skip = set()
    for path in found:
        if "debug" in path.name:
            skip.add(path.name)
    if "baseline_predictions_with_boosting.parquet" in names:
        skip.update({"baseline_predictions.parquet", "boosting_predictions.parquet"})
    if "temporal_tabular_backtest_predictions.parquet" in names:
        skip.update(
            {
                name
                for name in names
                if name.startswith("temporal_tabular_")
                and name != "temporal_tabular_backtest_predictions.parquet"
            }
        )
    if "ft_transformer_predictions.parquet" in names:
        skip.add("ft_small_seed362559_predictions.parquet")
    skip.add("xgb_debug_predictions.parquet")
    selected = [path for path in found if path.name not in skip]
    for name in sorted(skip):
        if name in names:
            warnings.append(f"Skipped redundant/debug prediction artifact: {prediction_dir / name}")
    return selected


def normalize_prediction_frame(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    rename = {}
    if "permno" in df.columns and "PERMNO" not in df.columns:
        rename["permno"] = "PERMNO"
    if "mthcaldt" in df.columns and "MthCalDt" not in df.columns:
        rename["mthcaldt"] = "MthCalDt"
    df = df.rename(columns=rename).copy()
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"], errors="raise")
    df["split"] = df["split"].astype(str)
    df["target_ret_1m"] = pd.to_numeric(df["target_ret_1m"], errors="coerce")
    duplicate_count = int(df.duplicated(["PERMNO", "MthCalDt"]).sum())
    if duplicate_count:
        raise ValueError(f"{path.name} has {duplicate_count:,} duplicate PERMNO-MthCalDt rows.")
    return df


def add_probability_derived_scores(df: pd.DataFrame, source_file: str) -> list[str]:
    created = []
    prob_prefixes = sorted(
        {
            col[: -len("_top")]
            for col in df.columns
            if col.startswith("prob_") and col.endswith("_top")
        }
    )
    for prefix in prob_prefixes:
        bottom = f"{prefix}_bottom"
        top = f"{prefix}_top"
        if bottom not in df.columns:
            continue
        base = prefix.removeprefix("prob_")
        if f"score_{base}_classifier" in df.columns or f"prediction_{base}_classifier_score" in df.columns:
            continue
        derived = f"score_{safe_slug(Path(source_file).stem)}_{safe_slug(prefix)}_classifier"
        if derived not in df.columns:
            df[derived] = pd.to_numeric(df[top], errors="coerce") - pd.to_numeric(
                df[bottom], errors="coerce"
            )
            created.append(derived)
    return created


def add_er_train_scores(df: pd.DataFrame, source_file: str) -> list[str]:
    if "top_bottom_label" not in df.columns:
        return []
    train = df.loc[df["split"].eq("train"), ["top_bottom_label", "target_ret_1m"]].dropna()
    if train.empty:
        return []
    means = train.groupby("top_bottom_label")["target_ret_1m"].mean()
    created = []
    prob_prefixes = sorted(
        {
            col[: -len("_top")]
            for col in df.columns
            if col.startswith("prob_") and col.endswith("_top")
        }
    )
    for prefix in prob_prefixes:
        cols = [f"{prefix}_bottom", f"{prefix}_middle", f"{prefix}_top"]
        if not all(col in df.columns for col in cols):
            continue
        base = prefix.removeprefix("prob_")
        if f"score_{base}_er_train" in df.columns or f"prediction_{base}_er_train_score" in df.columns:
            continue
        if not set([0, 1, 2]).issubset(set(means.index.astype(int))):
            continue
        derived = f"score_{safe_slug(Path(source_file).stem)}_{safe_slug(prefix)}_er_train"
        if derived not in df.columns:
            mu = means.reindex([0, 1, 2]).to_numpy(dtype=float)
            probs = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            df[derived] = probs @ mu
            created.append(derived)
    return created


def discover_scores(df: pd.DataFrame, path: Path) -> list[ScoreSpec]:
    source_file = path.name
    add_probability_derived_scores(df, source_file)
    add_er_train_scores(df, source_file)
    specs = []
    for column in sorted(df.columns):
        if not is_score_column(column, df[column]):
            continue
        label = score_label_from_column(source_file, column)
        family = family_from_source_and_score(source_file, label)
        sign_meaningful = not column.startswith("prob_")
        specs.append(
            ScoreSpec(
                source_file=source_file,
                score_label=label,
                score_column=column,
                model_family=family,
                sign_meaningful=sign_meaningful,
            )
        )
    return specs


def build_strategy_grid(args: argparse.Namespace) -> list[StrategySpec]:
    specs = []
    base_rules = ["long_only", "long_short", "gross_normalized_long_short", "long_short_130_30"]
    short_rules = ["long_short", "gross_normalized_long_short", "long_short_130_30"]
    for q in args.quantiles:
        for rule in base_rules:
            specs.append(StrategySpec(rule, q, "none", None, None, args.cost_bps))
        for rule in short_rules:
            specs.append(StrategySpec(rule, q, "sign", 0.0, None, args.cost_bps))
        for threshold in args.thresholds:
            for rule in short_rules:
                specs.append(StrategySpec(rule, q, "absolute", float(threshold), None, args.cost_bps))
    for z_threshold in args.z_thresholds:
        for rule in short_rules:
            specs.append(StrategySpec(rule, None, "zscore", None, float(z_threshold), args.cost_bps))

    unique = {}
    for spec in specs:
        unique[(spec.rule, spec.q, spec.gate_type, spec.threshold, spec.z_threshold, spec.one_way_cost_bps)] = spec
    return list(unique.values())


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    if wealth.empty:
        return np.nan
    return float((wealth / wealth.cummax() - 1.0).min())


def average_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    if wealth.empty:
        return np.nan
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.mean())


def annualized_return(monthly_returns: pd.Series) -> float:
    clean = monthly_returns.dropna()
    if clean.empty:
        return np.nan
    total = float((1.0 + clean).prod())
    if total <= 0:
        return np.nan
    return total ** (12.0 / len(clean)) - 1.0


def annualized_volatility(monthly_returns: pd.Series) -> float:
    clean = monthly_returns.dropna()
    if len(clean) <= 1:
        return np.nan
    return float(clean.std(ddof=1) * math.sqrt(12.0))


def sharpe_ratio(monthly_returns: pd.Series) -> float:
    ann_ret = annualized_return(monthly_returns)
    ann_vol = annualized_volatility(monthly_returns)
    if pd.isna(ann_ret) or pd.isna(ann_vol) or ann_vol == 0:
        return np.nan
    return float(ann_ret / ann_vol)


def summarize_time_series(values: pd.Series) -> dict[str, float]:
    clean = values.dropna()
    n_obs = int(clean.size)
    mean = float(clean.mean()) if n_obs else np.nan
    std = float(clean.std(ddof=1)) if n_obs > 1 else np.nan
    t_stat = mean / (std / math.sqrt(n_obs)) if n_obs > 1 and std and std > 0 else np.nan
    return {"mean": mean, "tstat": t_stat, "num_months": n_obs}


def compute_rank_ic(df: pd.DataFrame, score: ScoreSpec) -> pd.DataFrame:
    rows = []
    work = df.loc[df["split"].isin(SPLITS), ["split", "MthCalDt", score.score_column, "target_ret_1m"]].dropna()
    for (split, month), part in work.groupby(["split", "MthCalDt"], sort=True):
        if len(part) < 5 or part[score.score_column].nunique() <= 1 or part["target_ret_1m"].nunique() <= 1:
            rank_ic = np.nan
        else:
            rank_ic = part[score.score_column].corr(part["target_ret_1m"], method="spearman")
        rows.append(
            {
                "source_file": score.source_file,
                "score_label": score.score_label,
                "split": split,
                "month": month,
                "rank_ic_t": rank_ic,
                "num_obs": len(part),
            }
        )
    monthly = pd.DataFrame(rows)
    if monthly.empty:
        return pd.DataFrame()
    summary_rows = []
    for split, part in monthly.groupby("split", sort=True):
        stats = summarize_time_series(part["rank_ic_t"])
        summary_rows.append(
            {
                "source_file": score.source_file,
                "score_label": score.score_label,
                "model_family": score.model_family,
                "split": split,
                "mean_rank_ic": stats["mean"],
                "rank_ic_tstat": stats["tstat"],
                "rank_ic_hit_rate": float(part["rank_ic_t"].gt(0).mean()),
                "rank_ic_months": stats["num_months"],
            }
        )
    return pd.DataFrame(summary_rows)


def portfolio_weights(part: pd.DataFrame, score_col: str, spec: StrategySpec) -> pd.Series:
    scores = pd.to_numeric(part[score_col], errors="coerce")
    valid = scores.notna()
    weights = pd.Series(0.0, index=part.index)
    if valid.sum() < 10:
        return weights

    if spec.gate_type == "zscore":
        valid_scores = scores.loc[valid]
        std = valid_scores.std(ddof=1)
        if pd.isna(std) or std == 0:
            return weights
        z = (valid_scores - valid_scores.mean()) / std
        long_idx = z[z >= float(spec.z_threshold)].index
        short_idx = z[z <= -float(spec.z_threshold)].index
    else:
        q = float(spec.q)
        ranks = scores[valid].rank(method="first", pct=True)
        long_idx = ranks[ranks >= 1.0 - q].index
        short_idx = ranks[ranks <= q].index
        if spec.gate_type == "sign":
            long_idx = long_idx[scores.loc[long_idx] > 0]
            short_idx = short_idx[scores.loc[short_idx] < 0]
        elif spec.gate_type == "absolute":
            threshold = float(spec.threshold)
            long_idx = long_idx[scores.loc[long_idx] > threshold]
            short_idx = short_idx[scores.loc[short_idx] < -threshold]

    if spec.rule == "long_only":
        if len(long_idx) > 0:
            weights.loc[long_idx] = 1.0 / len(long_idx)
        return weights
    if spec.rule == "long_short":
        long_exposure, short_exposure = 1.0, 1.0
    elif spec.rule == "gross_normalized_long_short":
        long_exposure, short_exposure = 0.5, 0.5
    elif spec.rule == "long_short_130_30":
        long_exposure, short_exposure = 1.3, 0.3
    else:
        raise ValueError(f"Unknown rule: {spec.rule}")
    if len(long_idx) > 0:
        weights.loc[long_idx] = long_exposure / len(long_idx)
    if len(short_idx) > 0:
        weights.loc[short_idx] = -short_exposure / len(short_idx)
    return weights


def portfolio_weights_precomputed(part: pd.DataFrame, spec: StrategySpec) -> pd.Series:
    scores = part["_score"]
    valid = scores.notna()
    weights = pd.Series(0.0, index=part.index)
    if valid.sum() < 10:
        return weights

    if spec.gate_type == "zscore":
        z = part["_zscore"]
        long_idx = z[z >= float(spec.z_threshold)].index
        short_idx = z[z <= -float(spec.z_threshold)].index
    else:
        ranks = part["_rank_pct"]
        q = float(spec.q)
        long_idx = ranks[ranks >= 1.0 - q].index
        short_idx = ranks[ranks <= q].index
        if spec.gate_type == "sign":
            long_idx = long_idx[scores.loc[long_idx] > 0]
            short_idx = short_idx[scores.loc[short_idx] < 0]
        elif spec.gate_type == "absolute":
            threshold = float(spec.threshold)
            long_idx = long_idx[scores.loc[long_idx] > threshold]
            short_idx = short_idx[scores.loc[short_idx] < -threshold]

    if spec.rule == "long_only":
        if len(long_idx) > 0:
            weights.loc[long_idx] = 1.0 / len(long_idx)
        return weights
    if spec.rule == "long_short":
        long_exposure, short_exposure = 1.0, 1.0
    elif spec.rule == "gross_normalized_long_short":
        long_exposure, short_exposure = 0.5, 0.5
    elif spec.rule == "long_short_130_30":
        long_exposure, short_exposure = 1.3, 0.3
    else:
        raise ValueError(f"Unknown rule: {spec.rule}")
    if len(long_idx) > 0:
        weights.loc[long_idx] = long_exposure / len(long_idx)
    if len(short_idx) > 0:
        weights.loc[short_idx] = -short_exposure / len(short_idx)
    return weights


def portfolio_weights_numpy(
    scores: np.ndarray,
    ranks: np.ndarray,
    zscores: np.ndarray,
    spec: StrategySpec,
) -> np.ndarray:
    valid = np.isfinite(scores)
    weights = np.zeros(len(scores), dtype=float)
    if valid.sum() < 10:
        return weights

    if spec.gate_type == "zscore":
        long_mask = np.isfinite(zscores) & (zscores >= float(spec.z_threshold))
        short_mask = np.isfinite(zscores) & (zscores <= -float(spec.z_threshold))
    else:
        q = float(spec.q)
        long_mask = np.isfinite(ranks) & (ranks >= 1.0 - q)
        short_mask = np.isfinite(ranks) & (ranks <= q)
        if spec.gate_type == "sign":
            long_mask &= scores > 0
            short_mask &= scores < 0
        elif spec.gate_type == "absolute":
            threshold = float(spec.threshold)
            long_mask &= scores > threshold
            short_mask &= scores < -threshold

    if spec.rule == "long_only":
        n_long = int(long_mask.sum())
        if n_long:
            weights[long_mask] = 1.0 / n_long
        return weights
    if spec.rule == "long_short":
        long_exposure, short_exposure = 1.0, 1.0
    elif spec.rule == "gross_normalized_long_short":
        long_exposure, short_exposure = 0.5, 0.5
    elif spec.rule == "long_short_130_30":
        long_exposure, short_exposure = 1.3, 0.3
    else:
        raise ValueError(f"Unknown rule: {spec.rule}")
    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    if n_long:
        weights[long_mask] = long_exposure / n_long
    if n_short:
        weights[short_mask] = -short_exposure / n_short
    return weights


def turnover_drift_adjusted(
    current_weights: pd.Series,
    previous_weights: pd.Series | None,
    previous_returns: pd.Series | None,
    previous_gross_return: float | None,
) -> tuple[float, str]:
    if previous_weights is None or previous_returns is None or previous_gross_return is None:
        return float(current_weights.abs().sum()), "initial_abs_weight"
    denom = 1.0 + previous_gross_return
    if denom <= 0 or not np.isfinite(denom):
        aligned = pd.concat([previous_weights.rename("prev"), current_weights.rename("curr")], axis=1).fillna(0.0)
        return float((aligned["curr"] - aligned["prev"]).abs().sum()), "non_drift_fallback"
    drifted = previous_weights.mul(1.0 + previous_returns.reindex(previous_weights.index).fillna(0.0)) / denom
    aligned = pd.concat([drifted.rename("drifted"), current_weights.rename("curr")], axis=1).fillna(0.0)
    return float((aligned["curr"] - aligned["drifted"]).abs().sum()), "drift_adjusted"


def turnover_drift_adjusted_sparse(
    current_permnos: np.ndarray,
    current_weights: np.ndarray,
    previous_permnos: np.ndarray | None,
    previous_weights: np.ndarray | None,
    previous_returns: np.ndarray | None,
    previous_gross_return: float | None,
) -> tuple[float, str]:
    if previous_permnos is None or previous_weights is None or previous_returns is None or previous_gross_return is None:
        return float(np.abs(current_weights).sum()), "initial_abs_weight"
    denom = 1.0 + previous_gross_return
    if denom <= 0 or not np.isfinite(denom):
        drifted = previous_weights
        method = "non_drift_fallback"
    else:
        drifted = previous_weights * (1.0 + previous_returns) / denom
        method = "drift_adjusted_sparse"

    common, curr_idx, prev_idx = np.intersect1d(
        current_permnos, previous_permnos, assume_unique=False, return_indices=True
    )
    turnover = float(np.abs(current_weights).sum() + np.abs(drifted).sum())
    if len(common):
        turnover -= float(np.abs(current_weights[curr_idx]).sum() + np.abs(drifted[prev_idx]).sum())
        turnover += float(np.abs(current_weights[curr_idx] - drifted[prev_idx]).sum())
    return turnover, method


def spec_id(score: ScoreSpec, strategy: StrategySpec) -> str:
    q = "na" if strategy.q is None else f"{strategy.q:g}"
    threshold = "na" if strategy.threshold is None else f"{strategy.threshold:g}"
    z_threshold = "na" if strategy.z_threshold is None else f"{strategy.z_threshold:g}"
    return "__".join(
        [
            safe_slug(score.score_label),
            strategy.rule,
            f"q{q}",
            strategy.gate_type,
            f"thr{threshold}",
            f"z{z_threshold}",
            f"cost{strategy.one_way_cost_bps:g}",
        ]
    )


def build_portfolio_returns(
    df: pd.DataFrame,
    score: ScoreSpec,
    strategy: StrategySpec,
) -> pd.DataFrame:
    rows = []
    previous_weights_by_split: dict[str, pd.Series] = {}
    previous_returns_by_split: dict[str, pd.Series] = {}
    previous_gross_by_split: dict[str, float] = {}
    cost_rate = strategy.one_way_cost_bps / 10_000.0
    sid = spec_id(score, strategy)
    work = df.loc[df["split"].isin(SPLITS)].sort_values(["split", "MthCalDt", "PERMNO"]).copy()
    for (split, month), part in work.groupby(["split", "MthCalDt"], sort=True):
        weights = portfolio_weights(part, score.score_column, strategy)
        returns = pd.to_numeric(part["target_ret_1m"], errors="coerce").fillna(0.0)
        gross_return = float((weights * returns).sum())
        current_weights = pd.Series(weights.to_numpy(dtype=float), index=part["PERMNO"].to_numpy())
        current_returns = pd.Series(returns.to_numpy(dtype=float), index=part["PERMNO"].to_numpy())
        turnover, turnover_method = turnover_drift_adjusted(
            current_weights,
            previous_weights_by_split.get(split),
            previous_returns_by_split.get(split),
            previous_gross_by_split.get(split),
        )
        long_weight = float(weights.clip(lower=0).sum())
        short_weight = float(-weights.clip(upper=0).sum())
        gross_exposure = float(weights.abs().sum())
        net_exposure = float(weights.sum())
        trading_cost = cost_rate * turnover
        rows.append(
            {
                "spec_id": sid,
                "source_file": score.source_file,
                "score_label": score.score_label,
                "score_column": score.score_column,
                "model_family": score.model_family,
                "rule": strategy.rule,
                "q": strategy.q,
                "gate_type": strategy.gate_type,
                "threshold": strategy.threshold,
                "z_threshold": strategy.z_threshold,
                "one_way_cost_bps": strategy.one_way_cost_bps,
                "split": split,
                "month": month,
                "gross_return": gross_return,
                "turnover": turnover,
                "turnover_method": turnover_method,
                "trading_cost": trading_cost,
                "net_return": gross_return - trading_cost,
                "long_weight": long_weight,
                "short_weight": short_weight,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "n_long": int((weights > 0).sum()),
                "n_short": int((weights < 0).sum()),
            }
        )
        previous_weights_by_split[split] = current_weights
        previous_returns_by_split[split] = current_returns
        previous_gross_by_split[split] = gross_return
    return pd.DataFrame(rows)


def build_portfolio_returns_for_score(
    df: pd.DataFrame,
    score: ScoreSpec,
    strategies: list[StrategySpec],
) -> pd.DataFrame:
    rows = []
    previous_permnos: dict[tuple[str, str], np.ndarray] = {}
    previous_weights: dict[tuple[str, str], np.ndarray] = {}
    previous_returns: dict[tuple[str, str], np.ndarray] = {}
    previous_gross: dict[tuple[str, str], float] = {}
    states = [(spec_id(score, strategy), strategy) for strategy in strategies]
    work = df.loc[df["split"].isin(SPLITS)].sort_values(["split", "MthCalDt", "PERMNO"]).copy()
    work["_score"] = pd.to_numeric(work[score.score_column], errors="coerce")
    grouped_score = work.groupby(["split", "MthCalDt"], sort=False)["_score"]
    work["_rank_pct"] = grouped_score.rank(method="first", pct=True)
    monthly_mean = grouped_score.transform("mean")
    monthly_std = grouped_score.transform("std")
    work["_zscore"] = (work["_score"] - monthly_mean) / monthly_std.replace(0.0, np.nan)

    for (split, month), part in work.groupby(["split", "MthCalDt"], sort=True):
        permnos = part["PERMNO"].to_numpy()
        returns_np = pd.to_numeric(part["target_ret_1m"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        scores_np = part["_score"].to_numpy(dtype=float)
        ranks_np = part["_rank_pct"].to_numpy(dtype=float)
        zscores_np = part["_zscore"].to_numpy(dtype=float)
        for sid, strategy in states:
            weights_np = portfolio_weights_numpy(scores_np, ranks_np, zscores_np, strategy)
            gross_return = float(weights_np @ returns_np)
            nonzero = weights_np != 0.0
            current_permnos = permnos[nonzero]
            current_weights = weights_np[nonzero]
            current_returns = returns_np[nonzero]
            key = (sid, split)
            turnover, turnover_method = turnover_drift_adjusted_sparse(
                current_permnos,
                current_weights,
                previous_permnos.get(key),
                previous_weights.get(key),
                previous_returns.get(key),
                previous_gross.get(key),
            )
            long_weight = float(weights_np[weights_np > 0].sum())
            short_weight = float(-weights_np[weights_np < 0].sum())
            gross_exposure = float(np.abs(weights_np).sum())
            net_exposure = float(weights_np.sum())
            trading_cost = (strategy.one_way_cost_bps / 10_000.0) * turnover
            rows.append(
                {
                    "spec_id": sid,
                    "source_file": score.source_file,
                    "score_label": score.score_label,
                    "score_column": score.score_column,
                    "model_family": score.model_family,
                    "rule": strategy.rule,
                    "q": strategy.q,
                    "gate_type": strategy.gate_type,
                    "threshold": strategy.threshold,
                    "z_threshold": strategy.z_threshold,
                    "one_way_cost_bps": strategy.one_way_cost_bps,
                    "split": split,
                    "month": month,
                    "gross_return": gross_return,
                    "turnover": turnover,
                    "turnover_method": turnover_method,
                    "trading_cost": trading_cost,
                    "net_return": gross_return - trading_cost,
                    "long_weight": long_weight,
                    "short_weight": short_weight,
                    "gross_exposure": gross_exposure,
                    "net_exposure": net_exposure,
                    "n_long": int((weights_np > 0).sum()),
                    "n_short": int((weights_np < 0).sum()),
                }
            )
            previous_permnos[key] = current_permnos
            previous_weights[key] = current_weights
            previous_returns[key] = current_returns
            previous_gross[key] = gross_return
    return pd.DataFrame(rows)


def alpha_beta(strategy: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, float]:
    if benchmark.empty:
        return {"annualized_alpha": np.nan, "beta": np.nan, "alpha_tstat": np.nan, "alpha_beta_months": 0}
    reg = strategy[["month", "net_return"]].merge(benchmark, on="month", how="inner").dropna()
    if len(reg) < 3:
        return {"annualized_alpha": np.nan, "beta": np.nan, "alpha_tstat": np.nan, "alpha_beta_months": len(reg)}
    y = reg["net_return"].to_numpy(dtype=float)
    x = reg["benchmark_return"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coef
    n, k = X.shape
    sigma2 = float((resid @ resid) / max(n - k, 1))
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se_alpha = math.sqrt(cov[0, 0]) if cov[0, 0] >= 0 else np.nan
    except np.linalg.LinAlgError:
        se_alpha = np.nan
    return {
        "annualized_alpha": float(12.0 * coef[0]),
        "beta": float(coef[1]),
        "alpha_tstat": float(coef[0] / se_alpha) if se_alpha and not pd.isna(se_alpha) else np.nan,
        "alpha_beta_months": len(reg),
    }


def summarize_spec(monthly: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = SPEC_COLUMNS + ["split"]
    for keys, part in monthly.groupby(group_cols, dropna=False, sort=True):
        data = dict(zip(group_cols, keys))
        net = part["net_return"]
        gross = part["gross_return"]
        exposure = part["gross_exposure"].replace(0.0, np.nan)
        exposure_normalized = net / exposure
        row = {
            **data,
            "months": len(part),
            "net_annualized_return": annualized_return(net),
            "net_annualized_volatility": annualized_volatility(net),
            "net_sharpe": sharpe_ratio(net),
            "gross_annualized_return": annualized_return(gross),
            "gross_annualized_volatility": annualized_volatility(gross),
            "gross_sharpe": sharpe_ratio(gross),
            "max_drawdown": max_drawdown(net),
            "average_drawdown": average_drawdown(net),
            "average_monthly_turnover": float(part["turnover"].mean()),
            "average_gross_exposure": float(part["gross_exposure"].mean()),
            "average_net_exposure": float(part["net_exposure"].mean()),
            "exposure_normalized_annualized_return": annualized_return(exposure_normalized),
            "exposure_normalized_sharpe": sharpe_ratio(exposure_normalized),
            "average_n_long": float(part["n_long"].mean()),
            "average_n_short": float(part["n_short"].mean()),
            "min_n_long": int(part["n_long"].min()),
            "min_n_short": int(part["n_short"].min()),
        }
        if "split" in benchmark.columns and benchmark["split"].ne("").any():
            benchmark_part = benchmark.loc[benchmark["split"].eq(data["split"])]
        else:
            benchmark_part = benchmark
        row.update(alpha_beta(part, benchmark_part))
        rows.append(row)
    return pd.DataFrame(rows)


def adjusted_close_from_yfinance(benchmark: pd.DataFrame) -> pd.Series:
    if benchmark.empty:
        raise RuntimeError("yfinance returned no SPY data.")
    if isinstance(benchmark.columns, pd.MultiIndex):
        if ("Adj Close", BENCHMARK_TICKER) in benchmark.columns:
            adj_close = benchmark[("Adj Close", BENCHMARK_TICKER)]
        elif ("Close", BENCHMARK_TICKER) in benchmark.columns:
            adj_close = benchmark[("Close", BENCHMARK_TICKER)]
        else:
            ticker_frame = benchmark.xs(BENCHMARK_TICKER, axis=1, level=-1)
            if "Adj Close" in ticker_frame.columns:
                adj_close = ticker_frame["Adj Close"]
            elif "Close" in ticker_frame.columns:
                adj_close = ticker_frame["Close"]
            else:
                adj_close = ticker_frame.iloc[:, 0]
    elif "Adj Close" in benchmark.columns:
        adj_close = benchmark["Adj Close"]
    elif "Close" in benchmark.columns:
        adj_close = benchmark["Close"]
    else:
        raise RuntimeError("Unable to locate SPY adjusted close in yfinance output.")
    adj_close = pd.to_numeric(adj_close, errors="coerce").dropna()
    if adj_close.empty:
        raise RuntimeError("SPY adjusted close series is empty after cleaning.")
    adj_close.index = pd.to_datetime(adj_close.index).tz_localize(None)
    return adj_close


def fetch_spy_returns(return_periods: pd.Series) -> pd.DataFrame:
    yf = import_yfinance()
    periods = pd.PeriodIndex(return_periods.dropna().astype(str).drop_duplicates().sort_values(), freq="M")
    if periods.empty:
        return pd.DataFrame(columns=["return_period", "benchmark_return"])
    start = (periods.min() - 2).to_timestamp("M").date().isoformat()
    end = ((periods.max() + 1).to_timestamp("M") + pd.Timedelta(days=5)).date().isoformat()
    benchmark = yf.download(
        BENCHMARK_TICKER,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )
    adj_close = adjusted_close_from_yfinance(benchmark)
    monthly = adj_close.resample("ME").last().pct_change().dropna().rename("benchmark_return").to_frame()
    monthly = monthly.reset_index()
    monthly = monthly.rename(columns={monthly.columns[0]: "return_month"})
    monthly["return_month"] = pd.to_datetime(monthly["return_month"], errors="raise")
    monthly["return_period"] = monthly["return_month"].dt.to_period("M")
    aligned = pd.DataFrame({"return_period": periods}).merge(
        monthly[["return_period", "benchmark_return"]],
        on="return_period",
        how="left",
        validate="one_to_one",
    )
    if aligned["benchmark_return"].isna().any():
        missing = aligned.loc[aligned["benchmark_return"].isna(), "return_period"].astype(str).tolist()
        raise RuntimeError(f"Missing SPY benchmark returns for months: {missing[:5]}")
    return aligned.sort_values("return_period").reset_index(drop=True)


def month_split_frame(df: pd.DataFrame) -> pd.DataFrame:
    months = df.loc[df["split"].isin(SPLITS), ["split", "MthCalDt"]].dropna().drop_duplicates().copy()
    months["split"] = months["split"].astype(str)
    months["MthCalDt"] = pd.to_datetime(months["MthCalDt"], errors="raise")
    return months


def fetch_spy_benchmark_for_months(months: pd.DataFrame) -> pd.DataFrame:
    if months.empty:
        return pd.DataFrame(columns=["split", "month", "return_period", "benchmark_return", "benchmark_name"])
    work = months.drop_duplicates(["split", "MthCalDt"]).copy()
    work["month"] = pd.to_datetime(work["MthCalDt"], errors="raise")
    work["return_period"] = work["month"].dt.to_period("M") + 1
    spy = fetch_spy_returns(work["return_period"])
    out = work[["split", "month", "return_period"]].merge(
        spy,
        on="return_period",
        how="left",
        validate="many_to_one",
    )
    if out["benchmark_return"].isna().any():
        missing = out.loc[out["benchmark_return"].isna(), "return_period"].astype(str).tolist()
        raise RuntimeError(f"Missing aligned SPY benchmark returns for months: {missing[:5]}")
    out["benchmark_name"] = BENCHMARK_TICKER
    return out.sort_values(["split", "month"]).reset_index(drop=True)


def ensure_spy_benchmark(benchmark: pd.DataFrame, df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    required = month_split_frame(df)
    if benchmark.empty:
        return fetch_spy_benchmark_for_months(required)
    covered = benchmark[["split", "month"]].drop_duplicates()
    missing = required.rename(columns={"MthCalDt": "month"}).merge(
        covered,
        on=["split", "month"],
        how="left",
        indicator=True,
    )
    if missing["_merge"].eq("left_only").any():
        existing = benchmark[["split", "month"]].rename(columns={"month": "MthCalDt"})
        combined = pd.concat([existing, required], ignore_index=True).drop_duplicates(["split", "MthCalDt"])
        warnings.append("Extended SPY benchmark from yfinance to cover additional prediction months.")
        return fetch_spy_benchmark_for_months(combined)
    return benchmark


def merge_rank_ic(metrics: pd.DataFrame, rank_ic: pd.DataFrame) -> pd.DataFrame:
    return metrics.merge(
        rank_ic,
        on=["source_file", "score_label", "model_family", "split"],
        how="left",
        validate="many_to_one",
    )


def validation_preselection(
    full_metrics: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    validation = full_metrics.loc[full_metrics["split"].eq("validation")].copy()
    validation["pass_rank_ic"] = validation["mean_rank_ic"].gt(args.min_rank_ic)
    validation["pass_rank_ic_tstat"] = validation["rank_ic_tstat"].gt(args.min_rank_ic_tstat)
    validation["pass_max_drawdown"] = validation["max_drawdown"].gt(args.min_max_dd)
    validation["pass_turnover"] = validation["average_monthly_turnover"].lt(args.max_turnover)
    validation["pass_validation_filters"] = (
        validation["pass_rank_ic"]
        & validation["pass_rank_ic_tstat"]
        & validation["pass_max_drawdown"]
        & validation["pass_turnover"]
    )
    validation["selected_by_validation"] = False
    validation = validation.sort_values(
        ["pass_validation_filters", "net_sharpe", "net_annualized_return", "mean_rank_ic"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    validation["validation_rank"] = np.arange(1, len(validation) + 1)
    return validation


def select_best_specs(preselection: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selected = preselection.loc[preselection["pass_validation_filters"]].head(top_n).copy()
    selected["selected_by_validation"] = True
    selected["validation_rank"] = np.arange(1, len(selected) + 1)
    return selected


def test_rows_for_selected(full_metrics: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    cols = ["spec_id", "validation_rank", "net_sharpe"]
    test = full_metrics.loc[full_metrics["split"].eq("test")].merge(
        selected[cols].rename(columns={"net_sharpe": "validation_net_sharpe"}),
        on="spec_id",
        how="inner",
    )
    test["selected_by_validation"] = True
    return test.sort_values("validation_rank")


def select_representative_specs(
    preselection: pd.DataFrame,
    families: Iterable[str],
    rank_prefix: str,
) -> pd.DataFrame:
    picks = []
    for family in families:
        fam = preselection.loc[
            preselection["model_family"].eq(family) & preselection["pass_validation_filters"]
        ]
        if fam.empty:
            fam = preselection.loc[preselection["model_family"].eq(family)]
        if fam.empty:
            continue
        pick = fam.sort_values(["net_sharpe", "net_annualized_return"], ascending=False).head(1).copy()
        pick["comparison_role"] = rank_prefix
        picks.append(pick)
    if not picks:
        return pd.DataFrame()
    return pd.concat(picks, ignore_index=True)


def compact_model_comparison(
    full_metrics: pd.DataFrame,
    preselection: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    baseline_picks = select_representative_specs(preselection, BASELINE_FAMILIES, "baseline")
    model_picks = select_representative_specs(preselection, COMPARISON_MODEL_FAMILIES[2:], "model_representative")
    top_picks = selected.copy()
    if not top_picks.empty:
        top_picks["comparison_role"] = "top_validation"
    pieces = [part for part in [top_picks, baseline_picks, model_picks] if not part.empty]
    if not pieces:
        return pd.DataFrame()
    picked = pd.concat(pieces, ignore_index=True)
    picked = picked.drop_duplicates("spec_id", keep="first")
    selected_specs = picked[["spec_id", "comparison_role", "validation_rank"]].copy()
    out = full_metrics.loc[full_metrics["spec_id"].isin(selected_specs["spec_id"])].merge(
        selected_specs,
        on="spec_id",
        how="left",
        validate="many_to_one",
    )
    family_order = {family: idx for idx, family in enumerate(COMPARISON_MODEL_FAMILIES)}
    out["_family_order"] = out["model_family"].map(family_order).fillna(len(family_order))
    out["_role_order"] = out["comparison_role"].map(
        {"top_validation": 0, "baseline": 1, "model_representative": 2}
    ).fillna(3)
    return (
        out.sort_values(["_role_order", "validation_rank", "_family_order", "model_family", "split"])
        .drop(columns=["_family_order", "_role_order"])
        .reset_index(drop=True)
    )


def top_specs_spy_alpha_beta(full_metrics: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    rank_cols = ["spec_id", "validation_rank"]
    cols = [
        "spec_id",
        "validation_rank",
        "split",
        "model_family",
        "score_label",
        "rule",
        "gate_type",
        "q",
        "threshold",
        "z_threshold",
        "net_sharpe",
        "net_annualized_return",
        "annualized_alpha",
        "beta",
        "alpha_tstat",
        "alpha_beta_months",
    ]
    out = full_metrics.loc[full_metrics["spec_id"].isin(selected["spec_id"])].merge(
        selected[rank_cols],
        on="spec_id",
        how="left",
        validate="many_to_one",
    )
    return out[cols].sort_values(["validation_rank", "split"]).reset_index(drop=True)


def cumulative_wealth(returns: pd.Series, months: pd.Series) -> pd.Series:
    index = pd.to_datetime(months)
    values = (1.0 + returns.fillna(0.0).to_numpy(dtype=float)).cumprod()
    if len(index) == 0:
        return pd.Series(dtype=float)
    start = index.iloc[0] - pd.offsets.MonthBegin(1)
    return pd.concat([pd.Series([1.0], index=[start]), pd.Series(values, index=index)])


def drawdown_series(returns: pd.Series, months: pd.Series) -> pd.Series:
    wealth = cumulative_wealth(returns, months)
    if wealth.empty:
        return wealth
    return wealth / wealth.cummax() - 1.0


def figure_strategy_label(row: pd.Series) -> str:
    family = str(row.get("model_family", "Strategy"))
    rule = str(row.get("rule", ""))
    q = row.get("q")
    gate = str(row.get("gate_type", "none"))
    threshold = row.get("threshold")
    score = str(row.get("score_label", "")).split("__")[-1]
    parts = [family]
    if family == "TTT" and pd.notna(row.get("validation_rank")):
        parts.append(f"rank {int(row['validation_rank'])}")
    if family in {"Naive Momentum", "XGBoost", "MLP", "FT-Transformer"} and score:
        parts.append(score.replace("_", " ")[:28])
    if rule:
        parts.append(rule.replace("_", " "))
    if pd.notna(q):
        parts.append(f"q={float(q):.0%}")
    if gate == "absolute" and pd.notna(threshold):
        parts.append(f"thr={float(threshold):g}")
    elif gate == "zscore" and pd.notna(row.get("z_threshold")):
        parts.append(f"z={float(row['z_threshold']):g}")
    return " | ".join(parts)


def select_figure_specs(
    full_metrics: pd.DataFrame,
    selected: pd.DataFrame,
    comparison: pd.DataFrame,
    ttt_top_n: int = 5,
) -> pd.DataFrame:
    picks = []
    seen = set()

    def add_rows(rows: pd.DataFrame, role: str) -> None:
        for _, row in rows.iterrows():
            spec_id = row["spec_id"]
            if spec_id in seen:
                continue
            picked = row.copy()
            picked["figure_role"] = role
            picked["figure_order"] = len(picks) + 1
            picks.append(picked)
            seen.add(spec_id)

    add_rows(selected.head(ttt_top_n), "top_validation_ttt")

    comp_val = comparison.loc[comparison["split"].eq("validation")].copy() if not comparison.empty else pd.DataFrame()
    val = full_metrics.loc[full_metrics["split"].eq("validation")].copy()
    for family in ["Naive Momentum", "XGBoost", "MLP", "FT-Transformer"]:
        family_rows = comp_val.loc[comp_val["model_family"].eq(family)]
        if family_rows.empty:
            family_rows = val.loc[val["model_family"].eq(family)].sort_values(
                ["net_sharpe", "net_annualized_return"],
                ascending=False,
            )
        add_rows(family_rows.head(1), f"best_{safe_slug(family)}")

    if not picks:
        return pd.DataFrame()
    picked = pd.DataFrame(picks)
    figure_cols = ["spec_id", "figure_role", "figure_order"]
    if "validation_rank" in picked.columns:
        figure_cols.append("validation_rank")
    out = full_metrics.loc[full_metrics["spec_id"].isin(picked["spec_id"])].merge(
        picked[figure_cols].drop_duplicates("spec_id"),
        on="spec_id",
        how="left",
        validate="many_to_one",
    )
    return out.sort_values(["figure_order", "split"]).reset_index(drop=True)


def selected_strategy_display_name(row: pd.Series) -> str:
    rank = int(row["validation_rank"]) if pd.notna(row.get("validation_rank")) else None
    if rank == 1:
        return "Main selected TTT"
    if rank == 3:
        return "Risk-balanced TTT"
    return figure_strategy_label(row)


def period_max_drawdown(returns: pd.Series) -> float:
    clean = returns.fillna(0.0)
    if clean.empty:
        return np.nan
    wealth = pd.concat([pd.Series([1.0]), (1.0 + clean).cumprod().reset_index(drop=True)], ignore_index=True)
    return float((wealth / wealth.cummax() - 1.0).min())


def selected_annual_return_mdd(monthly: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    target = selected.loc[selected["validation_rank"].isin([1, 3])].copy()
    if target.empty:
        return pd.DataFrame()
    target["strategy_name"] = target.apply(selected_strategy_display_name, axis=1)
    target_meta_cols = [
        "spec_id",
        "validation_rank",
        "strategy_name",
        "model_family",
        "score_label",
        "rule",
        "q",
        "gate_type",
        "threshold",
        "z_threshold",
    ]
    rows = []
    for _, spec in target[target_meta_cols].iterrows():
        spec_monthly = monthly.loc[monthly["spec_id"].eq(spec["spec_id"])].copy()
        if spec_monthly.empty:
            continue
        spec_monthly["month"] = pd.to_datetime(spec_monthly["month"])
        spec_monthly["year"] = spec_monthly["month"].dt.year
        for (split, year), part in spec_monthly.groupby(["split", "year"], sort=True):
            ordered = part.sort_values("month")
            returns = ordered["net_return"]
            row = spec.to_dict()
            row.update(
                {
                    "split": split,
                    "year": int(year),
                    "months": int(returns.notna().sum()),
                    "annual_net_return": float((1.0 + returns.fillna(0.0)).prod() - 1.0),
                    "year_max_drawdown": period_max_drawdown(returns),
                }
            )
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    split_order = {"validation": 0, "test": 1}
    out["_split_order"] = out["split"].map(split_order).fillna(9)
    return (
        out.sort_values(["_split_order", "year", "validation_rank"])
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )


def write_selected_annual_heatmap(output_dir: Path, monthly: pd.DataFrame, selected: pd.DataFrame) -> list[Path]:
    annual = selected_annual_return_mdd(monthly, selected)
    if annual.empty:
        return []

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = fig_dir / "selected_strategies_annual_return_mdd_heatmap_25bps.png"

    metric_rows = []
    for (split, year), part in annual.groupby(["split", "year"], sort=False):
        row = {"period": f"{split.title()} {int(year)}", "split": split, "year": int(year)}
        for _, spec_row in part.sort_values("validation_rank").iterrows():
            prefix = "Main" if int(spec_row["validation_rank"]) == 1 else "Risk-balanced"
            row[f"{prefix} return"] = spec_row["annual_net_return"]
            row[f"{prefix} MDD"] = spec_row["year_max_drawdown"]
        metric_rows.append(row)
    heat = pd.DataFrame(metric_rows)
    columns = ["Main return", "Main MDD", "Risk-balanced return", "Risk-balanced MDD"]
    heat = heat[["period", "split", "year"] + columns]
    average_rows = []
    for split, part in heat.groupby("split", sort=False):
        avg = {"period": f"{str(split).title()} AVG", "split": split, "year": 9999, "is_average": True}
        for col in columns:
            avg[col] = part[col].mean()
        average_rows.append(avg)
    heat["is_average"] = False
    with_avg = []
    for split, part in heat.groupby("split", sort=False):
        with_avg.append(part)
        avg = pd.DataFrame([row for row in average_rows if row["split"] == split])
        if not avg.empty:
            with_avg.append(avg)
    heat = pd.concat(with_avg, ignore_index=True)

    table_path = output_dir / "portfolio_selected_annual_return_mdd_25bps.csv"
    heat.to_csv(table_path, index=False)

    n_rows = len(heat)
    n_cols = len(columns)
    fig_height = max(5.2, 0.36 * n_rows + 1.25)
    fig, ax = plt.subplots(figsize=(10.6, fig_height))
    ax.set_xlim(0, n_cols + 1)
    ax.set_ylim(0, n_rows + 1)
    ax.axis("off")

    header_bg = "#E6E8E3"
    edge = "#333333"
    val_bg = "#F6FAFF"
    test_bg = "#FFF8F0"
    avg_bg = "#ECE7DA"
    ax.add_patch(plt.Rectangle((0, n_rows), 1, 1, facecolor=header_bg, edgecolor=edge, linewidth=1.4))
    ax.text(0.5, n_rows + 0.5, "Period", ha="center", va="center", weight="bold")
    for col_idx, col in enumerate(columns, start=1):
        ax.add_patch(plt.Rectangle((col_idx, n_rows), 1, 1, facecolor=header_bg, edgecolor=edge, linewidth=1.4))
        ax.text(col_idx + 0.5, n_rows + 0.5, col, ha="center", va="center", weight="bold")

    regular_heat = heat.loc[~heat["is_average"]]
    return_values = regular_heat[[col for col in columns if "return" in col]].to_numpy(dtype=float).ravel()
    mdd_values = regular_heat[[col for col in columns if "MDD" in col]].to_numpy(dtype=float).ravel()
    return_abs = max(abs(np.nanmin(return_values)), abs(np.nanmax(return_values)), 0.01)
    mdd_min = min(float(np.nanmin(mdd_values)), -0.01)

    def return_color(value: float) -> tuple[float, float, float, float]:
        scaled = 0.5 + 0.5 * np.clip(value / return_abs, -1.0, 1.0)
        return plt.cm.RdYlGn(float(scaled))

    def mdd_color(value: float) -> tuple[float, float, float, float]:
        intensity = np.clip(value / mdd_min, 0.0, 1.0)
        return plt.cm.Reds(float(0.10 + 0.75 * intensity))

    previous_split = None
    for row_idx, row in heat.iterrows():
        y = n_rows - row_idx - 1
        is_average = bool(row["is_average"])
        split_bg = avg_bg if is_average else (val_bg if row["split"] == "validation" else test_bg)
        linewidth = 2.2 if previous_split is not None and row["split"] != previous_split else (1.4 if is_average else 0.75)
        ax.add_patch(plt.Rectangle((0, y), 1, 1, facecolor=split_bg, edgecolor=edge, linewidth=linewidth))
        ax.text(0.5, y + 0.5, row["period"], ha="center", va="center", weight="bold" if is_average else "normal")
        for col_idx, col in enumerate(columns, start=1):
            value = float(row[col])
            color = avg_bg if is_average else (return_color(value) if "return" in col else mdd_color(value))
            ax.add_patch(plt.Rectangle((col_idx, y), 1, 1, facecolor=color, edgecolor=edge, linewidth=linewidth))
            ax.text(col_idx + 0.5, y + 0.5, f"{value:.1%}", ha="center", va="center", weight="bold")
        previous_split = row["split"]

    ax.text(
        0,
        n_rows + 1.12,
        "Annual Net Return and Within-Year Max Drawdown at 25 bps",
        ha="left",
        va="bottom",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [table_path, heatmap_path]


def write_figures(
    output_dir: Path,
    full_metrics: pd.DataFrame,
    selected: pd.DataFrame,
    comparison: pd.DataFrame,
    monthly: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> list[str]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    val = full_metrics.loc[full_metrics["split"].eq("validation")].copy()
    plot_specs = select_figure_specs(full_metrics, selected, comparison)
    plot_spec_ids = plot_specs["spec_id"].drop_duplicates().tolist() if not plot_specs.empty else selected["spec_id"].head(5).tolist()
    plot_validation = val.loc[val["spec_id"].isin(plot_spec_ids)].copy()
    plt.figure(figsize=(10, 6))
    families = sorted(val["model_family"].dropna().unique())
    for family in families:
        grp = val.loc[val["model_family"].eq(family)]
        plt.scatter(grp["max_drawdown"], grp["net_sharpe"], s=25, alpha=0.45, label=family)
    if not plot_validation.empty:
        plt.scatter(
            plot_validation["max_drawdown"],
            plot_validation["net_sharpe"],
            s=70,
            facecolors="none",
            edgecolors="black",
            linewidths=1.3,
            label="plotted specs",
        )
    plt.xlabel("Validation max drawdown")
    plt.ylabel("Validation net Sharpe")
    plt.title("Validation net Sharpe vs max drawdown at 25 bps")
    plt.legend(fontsize=8)
    path = fig_dir / "validation_net_sharpe_vs_max_dd_25bps.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    paths.append(str(path))

    test_monthly = monthly.loc[monthly["split"].eq("test") & monthly["spec_id"].isin(plot_spec_ids)].copy()

    plt.figure(figsize=(12, 6))
    for spec_id, grp in test_monthly.groupby("spec_id", sort=False):
        label = figure_strategy_label(grp.iloc[0])
        wealth = cumulative_wealth(grp.sort_values("month")["net_return"], grp.sort_values("month")["month"])
        plt.plot(wealth.index, wealth.values, linewidth=1.6, label=label[:95])
    bench_test = benchmark.loc[benchmark["split"].eq("test")].sort_values("month")
    if not bench_test.empty:
        wealth = cumulative_wealth(bench_test["benchmark_return"], bench_test["month"])
        plt.plot(wealth.index, wealth.values, color="black", linewidth=2.0, linestyle="--", label=bench_test["benchmark_name"].iloc[0])
    plt.axhline(1.0, color="black", linewidth=0.8, alpha=0.4)
    plt.title("Test cumulative wealth of selected raw strategies")
    plt.ylabel("Cumulative wealth")
    plt.legend(fontsize=7)
    path = fig_dir / "test_cumulative_wealth_best_specs_25bps.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    paths.append(str(path))

    plt.figure(figsize=(12, 6))
    for spec_id, grp in test_monthly.groupby("spec_id", sort=False):
        ordered = grp.sort_values("month")
        label = figure_strategy_label(ordered.iloc[0])
        dd = drawdown_series(ordered["net_return"], ordered["month"])
        plt.plot(dd.index, dd.values, linewidth=1.4, label=label[:95])
    if not bench_test.empty:
        dd = drawdown_series(bench_test["benchmark_return"], bench_test["month"])
        plt.plot(dd.index, dd.values, color="black", linewidth=2.0, linestyle="--", label=bench_test["benchmark_name"].iloc[0])
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    plt.title("Test drawdowns of selected raw strategies")
    plt.ylabel("Drawdown")
    plt.legend(fontsize=7)
    path = fig_dir / "test_drawdown_best_specs_25bps.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    paths.append(str(path))

    bars = plot_specs.loc[plot_specs["split"].eq("test")].copy() if not plot_specs.empty else pd.DataFrame()
    if not bars.empty:
        bars = bars.sort_values("figure_order")
        bars["plot_label"] = bars.apply(figure_strategy_label, axis=1)
        if not bench_test.empty:
            benchmark_bar = pd.DataFrame(
                [
                    {
                        "plot_label": bench_test["benchmark_name"].iloc[0],
                        "net_annualized_return": annualized_return(bench_test["benchmark_return"]),
                        "net_sharpe": sharpe_ratio(bench_test["benchmark_return"]),
                        "max_drawdown": max_drawdown(bench_test["benchmark_return"]),
                    }
                ]
            )
            bars = pd.concat([bars, benchmark_bar], ignore_index=True, sort=False)
        labels = bars["plot_label"]
        metrics = [
            ("net_annualized_return", "Net annualized return"),
            ("net_sharpe", "Net Sharpe"),
            ("max_drawdown", "Max drawdown"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for ax, (column, title) in zip(axes, metrics):
            ax.bar(labels, bars[column])
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = fig_dir / "selected_model_test_metric_bars_25bps.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def write_report(
    output_dir: Path,
    loaded_files: list[str],
    score_count: int,
    full_metrics: pd.DataFrame,
    preselection: pd.DataFrame,
    best_specs: pd.DataFrame,
    best_test: pd.DataFrame,
    comparison: pd.DataFrame,
    top_alpha_beta: pd.DataFrame,
    warnings: list[str],
) -> Path:
    path = output_dir / "portfolio_analysis_report_25bps.md"
    top10 = best_specs.head(10)
    def table_block(df: pd.DataFrame) -> str:
        if df.empty:
            return "No rows."
        return "```text\n" + df.to_string(index=False, max_cols=18, max_colwidth=80) + "\n```"

    lines = [
        "# Portfolio Analysis at 25 bps",
        "",
        "This report was generated by `scripts/backtesting/run_portfolio_analysis.py`.",
        "It evaluates raw model-score portfolios only; no risk overlays are included.",
        "",
        "## Run Summary",
        "",
        f"- Prediction files loaded: {len(loaded_files)}",
        f"- Score columns found: {score_count}",
        f"- Total split-level spec rows evaluated: {len(full_metrics):,}",
        f"- Unique specs evaluated: {full_metrics['spec_id'].nunique():,}",
        f"- Validation specs passing filters: {int(preselection['pass_validation_filters'].sum()) if not preselection.empty else 0:,}",
        "",
        "## Loaded Prediction Files",
        "",
    ]
    lines.extend([f"- `{name}`" for name in loaded_files])
    lines.extend(["", "## Top 10 Validation Specs", ""])
    lines.append(table_block(top10))
    lines.extend(["", "## Test Performance of Top Validation Specs", ""])
    lines.append(table_block(best_test.head(10)))
    lines.extend(["", "## SPY Alpha/Beta for Top Validation Specs", ""])
    lines.append(table_block(top_alpha_beta.head(20)))
    lines.extend(["", "## Best Model Comparison", ""])
    lines.append(table_block(comparison))
    lines.extend(
        [
            "",
            "## Method Notes and Caveats",
            "",
            "- Scores at month t are used with `target_ret_1m`, which is the next-month return already stored in the prediction rows.",
            "- SPY is fetched from yfinance and used as the market proxy for both benchmark plots and alpha/beta regressions.",
            "- The SPY benchmark is aligned so each formation month t receives SPY's return in calendar month t+1.",
            "- Naive momentum and XGBoost are retained as explicit baseline representatives in the comparison outputs.",
            "- Turnover uses drift-adjusted previous weights whenever the previous portfolio gross return is finite and greater than -100%.",
            "- The notebook absolute-threshold grid was extended to include `0.005` by default, as requested by the script CLI.",
            "- Z-score threshold strategies are included as raw threshold strategies only. Risk overlays are intentionally excluded.",
            "- The annual return/MDD heatmap compares validation-rank 1 and validation-rank 3 TTT specs by calendar year, with validation and test periods shown separately.",
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- None"])
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings_list: list[str] = []
    strategy_grid = build_strategy_grid(args)
    prediction_files = discover_prediction_files(args.prediction_dir, warnings_list)
    benchmark = pd.DataFrame(columns=["split", "month", "return_period", "benchmark_return", "benchmark_name"])

    all_monthly = []
    all_metrics = []
    all_rank_ic = []
    loaded_files = []
    score_count = 0
    evaluated_specs = 0

    for path in prediction_files:
        log(f"Loading {path.name}", args.debug)
        try:
            df = normalize_prediction_frame(pd.read_parquet(path), path)
        except Exception as exc:
            warnings_list.append(f"Skipped {path.name}: {exc}")
            continue
        scores = discover_scores(df, path)
        if not scores:
            warnings_list.append(f"No usable score columns found in {path.name}")
            continue
        loaded_files.append(path.name)
        score_count += len(scores)
        benchmark = ensure_spy_benchmark(benchmark, df, warnings_list)
        log(f"  usable scores: {len(scores)}", args.debug)
        for score in scores:
            valid_scores = df.loc[df["split"].isin(SPLITS), score.score_column]
            if valid_scores.isna().all():
                warnings_list.append(f"Skipped {score.score_label}: no validation/test scores.")
                continue
            rank_ic = compute_rank_ic(df, score)
            if not rank_ic.empty:
                all_rank_ic.append(rank_ic)
            strategies = [
                strategy
                for strategy in strategy_grid
                if strategy.gate_type != "sign" or score.sign_meaningful
            ]
            if args.max_specs is not None:
                remaining = args.max_specs - evaluated_specs
                if remaining <= 0:
                    break
                strategies = strategies[:remaining]
            monthly = build_portfolio_returns_for_score(df, score, strategies)
            if not monthly.empty:
                metrics = summarize_spec(monthly, benchmark)
                all_monthly.append(monthly)
                all_metrics.append(metrics)
                evaluated_specs += monthly["spec_id"].nunique()
            if args.max_specs is not None and evaluated_specs >= args.max_specs:
                break
        if args.max_specs is not None and evaluated_specs >= args.max_specs:
            warnings_list.append(f"Stopped after --max-specs={args.max_specs}.")
            break

    if not all_metrics:
        raise RuntimeError("No portfolio specifications were evaluated.")

    monthly_returns = pd.concat(all_monthly, ignore_index=True)
    full_metrics = pd.concat(all_metrics, ignore_index=True)
    rank_ic_summary = pd.concat(all_rank_ic, ignore_index=True) if all_rank_ic else pd.DataFrame()
    if not rank_ic_summary.empty:
        full_metrics = merge_rank_ic(full_metrics, rank_ic_summary)
    else:
        for col in ["mean_rank_ic", "rank_ic_tstat", "rank_ic_hit_rate", "rank_ic_months"]:
            full_metrics[col] = np.nan

    preselection = validation_preselection(full_metrics, args)
    best_specs = select_best_specs(preselection, args.top_n)
    best_test = test_rows_for_selected(full_metrics, best_specs)
    comparison = compact_model_comparison(full_metrics, preselection, best_specs)
    top_alpha_beta = top_specs_spy_alpha_beta(full_metrics, best_specs)

    if "split" in benchmark.columns and benchmark["split"].ne("").any():
        monthly_with_bench = monthly_returns.merge(
            benchmark[["split", "month", "benchmark_return", "benchmark_name"]],
            on=["split", "month"],
            how="left",
        )
    else:
        monthly_with_bench = monthly_returns.merge(
            benchmark[["month", "benchmark_return", "benchmark_name"]],
            on="month",
            how="left",
        )

    benchmark.to_csv(output_dir / "spy_benchmark_returns.csv", index=False)
    full_metrics.to_csv(output_dir / "portfolio_full_grid_metrics_25bps.csv", index=False)
    preselection.to_csv(output_dir / "portfolio_validation_preselection_25bps.csv", index=False)
    best_specs.to_csv(output_dir / "portfolio_best_specs_validation_25bps.csv", index=False)
    best_test.to_csv(output_dir / "portfolio_best_specs_test_25bps.csv", index=False)
    top_alpha_beta.to_csv(output_dir / "portfolio_top_specs_spy_alpha_beta_25bps.csv", index=False)
    comparison.to_csv(output_dir / "portfolio_selected_model_comparison_25bps.csv", index=False)
    monthly_with_bench.to_csv(output_dir / "portfolio_monthly_returns_25bps.csv", index=False)

    figure_paths = write_figures(output_dir, full_metrics, best_specs, comparison, monthly_returns, benchmark)
    annual_heatmap_paths = write_selected_annual_heatmap(output_dir, monthly_returns, best_specs)
    report_path = write_report(
        output_dir,
        loaded_files,
        score_count,
        full_metrics,
        preselection,
        best_specs,
        best_test,
        comparison,
        top_alpha_beta,
        warnings_list,
    )

    print("Portfolio analysis completed.")
    print(f"Prediction files loaded: {len(loaded_files)}")
    print(f"Score columns found: {score_count}")
    print(f"Unique specs evaluated: {full_metrics['spec_id'].nunique()}")
    print(f"Validation specs passing filters: {int(preselection['pass_validation_filters'].sum())}")
    print("Top 5 validation specs:")
    cols = [
        "validation_rank",
        "model_family",
        "score_label",
        "rule",
        "gate_type",
        "threshold",
        "z_threshold",
        "net_sharpe",
        "net_annualized_return",
        "max_drawdown",
        "average_monthly_turnover",
    ]
    print(best_specs[cols].head(5).to_string(index=False) if not best_specs.empty else "None")
    print("Files created:")
    for path in [
        output_dir / "spy_benchmark_returns.csv",
        output_dir / "portfolio_full_grid_metrics_25bps.csv",
        output_dir / "portfolio_validation_preselection_25bps.csv",
        output_dir / "portfolio_best_specs_validation_25bps.csv",
        output_dir / "portfolio_best_specs_test_25bps.csv",
        output_dir / "portfolio_top_specs_spy_alpha_beta_25bps.csv",
        output_dir / "portfolio_selected_model_comparison_25bps.csv",
        output_dir / "portfolio_monthly_returns_25bps.csv",
        *annual_heatmap_paths,
        report_path,
        *[Path(p) for p in figure_paths],
    ]:
        print(f" - {path}")
    if warnings_list:
        print("Warnings:")
        for warning in warnings_list:
            print(f" - {warning}")


if __name__ == "__main__":
    main()
