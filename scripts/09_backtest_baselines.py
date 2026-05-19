#!/usr/bin/env python3
"""Backtest existing baseline prediction files against SPY."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PREFERRED_PREDICTIONS = Path("outputs/predictions/baseline_predictions_with_boosting.parquet")
FALLBACK_PREDICTIONS = Path("outputs/predictions/baseline_predictions.parquet")
DEFAULT_OUTPUT_DIR = Path("outputs/backtests")
DEFAULT_FIGURE_DIR = DEFAULT_OUTPUT_DIR / "figures"
BENCHMARK_TICKER = "SPY"
COST_BPS_LEVELS = [0, 5, 10, 25, 50, 100]
LONG_SHORT_PORTFOLIOS = {"long_short_top_20": 0.20, "long_short_top_10": 0.10}
LONG_ONLY_PORTFOLIOS = {"long_only_top_20": 0.20, "long_only_top_10": 0.10}
SPLITS = ["validation", "test"]
FORBIDDEN_SCORE_COLUMNS = {"target_ret_1m", "target_quintile", "top_bottom_label"}
MODEL_COLUMNS = {
    "gb_classifier": ("prediction_gb_classifier_score", "XGBoost classifier"),
    "logistic_classifier": ("prediction_logistic_classifier_score", "Logistic classifier"),
    "naive_momentum": ("prediction_naive_momentum", "Naive momentum"),
    "gradient_boosting_reg": ("prediction_gradient_boosting_reg", "XGBoost regressor"),
}
OUTPUT_FILES = {
    "monthly_returns": DEFAULT_OUTPUT_DIR / "backtest_monthly_returns.csv",
    "performance_summary": DEFAULT_OUTPUT_DIR / "backtest_performance_summary.csv",
    "break_even": DEFAULT_OUTPUT_DIR / "backtest_break_even_costs.csv",
    "spy_returns": DEFAULT_OUTPUT_DIR / "spy_benchmark_returns.csv",
    "readme": DEFAULT_OUTPUT_DIR / "backtest_readme.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest existing baseline prediction files.")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def resolve_prediction_path(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Prediction file not found: {explicit}")
        return explicit
    if PREFERRED_PREDICTIONS.exists():
        return PREFERRED_PREDICTIONS
    if FALLBACK_PREDICTIONS.exists():
        return FALLBACK_PREDICTIONS
    raise FileNotFoundError(
        "No prediction parquet found. Expected one of "
        f"{PREFERRED_PREDICTIONS} or {FALLBACK_PREDICTIONS}."
    )


def import_yfinance() -> Any:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit(
            "yfinance is required for the SPY benchmark. Install it with:\n"
            "  ./.venv/bin/pip install yfinance"
        ) from exc
    return yf


def available_models(columns: list[str]) -> dict[str, tuple[str, str]]:
    selected = {
        model: (column, label)
        for model, (column, label) in MODEL_COLUMNS.items()
        if column in columns
    }
    if not selected:
        raise ValueError(
            "No supported prediction columns found. Expected any of: "
            + ", ".join(column for column, _ in MODEL_COLUMNS.values())
        )
    return selected


def load_predictions(path: Path) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    df = pd.read_parquet(path)
    if "split" not in df.columns:
        raise ValueError("Prediction file must contain a split column.")
    if df.duplicated(["PERMNO", "MthCalDt"]).any():
        raise ValueError("Prediction file contains duplicate PERMNO-month rows.")
    models = available_models(df.columns.tolist())
    selected_columns = ["PERMNO", "MthCalDt", "split", "target_ret_1m"] + [
        column for column, _ in models.values()
    ]
    df = df[selected_columns].copy()
    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"], errors="raise")
    df["split"] = df["split"].astype("string")
    df["target_ret_1m"] = pd.to_numeric(df["target_ret_1m"], errors="coerce")
    if df.loc[df["split"].isin(SPLITS), "target_ret_1m"].isna().any():
        raise ValueError("target_ret_1m is missing for rows used in validation/test backtests.")
    for column, _ in models.values():
        if column in FORBIDDEN_SCORE_COLUMNS:
            raise ValueError(f"Forbidden score column selected: {column}")
        df[column] = pd.to_numeric(df[column], errors="coerce")
    date_ranges = {
        split: df.loc[df["split"].eq(split), "MthCalDt"].sort_values()
        for split in SPLITS
    }
    if any(values.empty for values in date_ranges.values()):
        raise ValueError("Validation/test rows are required for backtesting.")
    if not date_ranges["validation"].max() < date_ranges["test"].min():
        raise ValueError("Validation period must be strictly before test period.")
    df = df.loc[df["split"].isin(SPLITS)].copy()
    df["return_month"] = df["MthCalDt"] + pd.offsets.MonthEnd(1)
    return df, models


def select_extreme_bucket(part: pd.DataFrame, score_column: str, fraction: float, ascending: bool) -> pd.DataFrame:
    count = max(1, int(math.floor(len(part) * fraction)))
    ordered = part.sort_values([score_column, "PERMNO"], ascending=[ascending, True], kind="mergesort")
    return ordered.head(count).copy()


def build_target_weights(permnos: pd.Series, weight: float) -> dict[int, float]:
    return {int(permno): float(weight) for permno in permnos.tolist()}


def portfolio_turnover(target_weights: dict[int, float], prior_weights: dict[int, float]) -> float:
    keys = set(target_weights).union(prior_weights)
    return float(sum(abs(target_weights.get(key, 0.0) - prior_weights.get(key, 0.0)) for key in keys))


def drift_weights(
    target_weights: dict[int, float],
    realized_returns: dict[int, float],
) -> dict[int, float]:
    if not target_weights:
        return {}
    portfolio_return = float(sum(weight * realized_returns.get(permno, 0.0) for permno, weight in target_weights.items()))
    denominator = 1.0 + portfolio_return
    if denominator <= 0:
        return {}
    return {
        permno: float(weight * (1.0 + realized_returns.get(permno, 0.0)) / denominator)
        for permno, weight in target_weights.items()
    }


def backtest_model(
    df: pd.DataFrame,
    model: str,
    model_label: str,
    score_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_df = df.loc[df["split"].eq(split), ["PERMNO", "MthCalDt", "return_month", "target_ret_1m", score_column]].dropna()
        if split_df.empty:
            continue
        for portfolio_type, fraction in {**LONG_ONLY_PORTFOLIOS, **LONG_SHORT_PORTFOLIOS}.items():
            previous_target_weights: dict[int, float] = {}
            previous_realized_returns: dict[int, float] = {}
            for month, part in split_df.groupby("MthCalDt", sort=True, observed=True):
                target_return_month = pd.Timestamp(month) + pd.offsets.MonthEnd(1)
                longs = select_extreme_bucket(part, score_column, fraction, ascending=False)
                longs = longs.assign(weight=1.0 / len(longs))
                if portfolio_type.startswith("long_only"):
                    shorts = part.iloc[0:0].copy()
                    target_weights = build_target_weights(longs["PERMNO"], 1.0 / len(longs))
                else:
                    shorts = select_extreme_bucket(part, score_column, fraction, ascending=True)
                    shared = set(longs["PERMNO"]).intersection(set(shorts["PERMNO"]))
                    if shared:
                        shorts = shorts.loc[~shorts["PERMNO"].isin(shared)].copy()
                    if shorts.empty:
                        continue
                    shorts = shorts.assign(weight=-1.0 / len(shorts))
                    target_weights = build_target_weights(longs["PERMNO"], 1.0 / len(longs))
                    target_weights.update(build_target_weights(shorts["PERMNO"], -1.0 / len(shorts)))

                prior_weights = drift_weights(previous_target_weights, previous_realized_returns)
                turnover = portfolio_turnover(target_weights, prior_weights)

                long_gross = float(longs["target_ret_1m"].mean())
                short_gross = float(shorts["target_ret_1m"].mean()) if not shorts.empty else 0.0
                gross_return = long_gross - short_gross

                realized_returns = {
                    int(row["PERMNO"]): float(row["target_ret_1m"])
                    for row in pd.concat([longs[["PERMNO", "target_ret_1m"]], shorts[["PERMNO", "target_ret_1m"]]], ignore_index=True).to_dict(orient="records")
                }

                rows.append(
                    {
                        "model": model,
                        "model_label": model_label,
                        "split": split,
                        "portfolio_type": portfolio_type,
                        "formation_month": pd.Timestamp(month).date().isoformat(),
                        "return_month": pd.Timestamp(target_return_month).date().isoformat(),
                        "score_column": score_column,
                        "gross_return": gross_return,
                        "turnover": turnover,
                        "long_count": int(len(longs)),
                        "short_count": int(len(shorts)),
                        "uses_prediction_scores_only": True,
                    }
                )
                previous_target_weights = target_weights
                previous_realized_returns = realized_returns
    return pd.DataFrame(rows)


def fetch_spy_monthly_returns(months: pd.Series) -> pd.DataFrame:
    yf = import_yfinance()
    month_index = pd.to_datetime(months, errors="raise").sort_values().drop_duplicates()
    start = (month_index.min() - pd.offsets.MonthEnd(2)).date().isoformat()
    end = (month_index.max() + pd.offsets.MonthEnd(1) + pd.Timedelta(days=5)).date().isoformat()
    benchmark = yf.download(
        BENCHMARK_TICKER,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )
    if benchmark.empty:
        raise RuntimeError("yfinance returned no SPY data.")
    if isinstance(benchmark.columns, pd.MultiIndex):
        if ("Adj Close", BENCHMARK_TICKER) in benchmark.columns:
            adj_close = benchmark[("Adj Close", BENCHMARK_TICKER)]
        elif ("Close", BENCHMARK_TICKER) in benchmark.columns:
            adj_close = benchmark[("Close", BENCHMARK_TICKER)]
        else:
            adj_close = benchmark.xs(BENCHMARK_TICKER, axis=1, level=-1).iloc[:, 0]
    else:
        if "Adj Close" in benchmark.columns:
            adj_close = benchmark["Adj Close"]
        elif "Close" in benchmark.columns:
            adj_close = benchmark["Close"]
        else:
            raise RuntimeError("Unable to locate SPY adjusted close in yfinance output.")
    monthly = adj_close.resample("ME").last().pct_change().dropna().rename("spy_return").to_frame()
    monthly = monthly.reset_index().rename(columns={"Date": "return_month"})
    monthly["return_month"] = pd.to_datetime(monthly["return_month"], errors="raise")
    benchmark_months = pd.DataFrame({"return_month": month_index})
    aligned = benchmark_months.merge(monthly, on="return_month", how="left", validate="one_to_one")
    if aligned["spy_return"].isna().any():
        missing = aligned.loc[aligned["spy_return"].isna(), "return_month"].dt.date.tolist()
        raise RuntimeError(f"Missing SPY benchmark returns for months: {missing[:5]}")
    return aligned.sort_values("return_month").reset_index(drop=True)


def expand_costs(monthly_returns: pd.DataFrame, spy_returns: pd.DataFrame) -> pd.DataFrame:
    expanded_frames = []
    spy = spy_returns.copy()
    spy["return_month"] = pd.to_datetime(spy["return_month"], errors="raise")
    for cost_bps in COST_BPS_LEVELS:
        cost_rate = cost_bps / 10_000.0
        frame = monthly_returns.copy()
        frame["cost_bps"] = cost_bps
        frame["cost_rate"] = cost_rate
        frame["net_return"] = frame["gross_return"] - cost_rate * frame["turnover"]
        frame["return_month"] = pd.to_datetime(frame["return_month"], errors="raise")
        frame = frame.merge(spy, on="return_month", how="left", validate="many_to_one")
        if frame["spy_return"].isna().any():
            raise ValueError("SPY benchmark alignment produced missing values.")
        expanded_frames.append(frame)
    out = pd.concat(expanded_frames, ignore_index=True)
    out["formation_month"] = pd.to_datetime(out["formation_month"], errors="raise").dt.date.astype(str)
    out["return_month"] = pd.to_datetime(out["return_month"], errors="raise").dt.date.astype(str)
    return out


def cumulative_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def annualized_return(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    return float((1.0 + returns).prod() ** (12.0 / len(returns)) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * math.sqrt(12.0))


def sharpe_ratio(returns: pd.Series) -> float:
    if len(returns) < 2:
        return np.nan
    volatility = returns.std(ddof=1)
    if volatility <= 0 or not np.isfinite(volatility):
        return np.nan
    return float(returns.mean() / volatility * math.sqrt(12.0))


def sortino_ratio(returns: pd.Series) -> float:
    downside = np.minimum(returns.to_numpy(dtype="float64"), 0.0)
    downside_deviation = math.sqrt(float(np.mean(np.square(downside))))
    if downside_deviation <= 0 or not np.isfinite(downside_deviation):
        return np.nan
    return float(returns.mean() / downside_deviation * math.sqrt(12.0))


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns).cumprod()
    peaks = wealth.cummax()
    return wealth / peaks - 1.0


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    return float(drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series) -> float:
    ann_return = annualized_return(returns)
    mdd = max_drawdown(returns)
    if not np.isfinite(ann_return) or not np.isfinite(mdd) or mdd >= 0:
        return np.nan
    return float(ann_return / abs(mdd))


def monthly_return_tstat(returns: pd.Series) -> float:
    if len(returns) < 2:
        return np.nan
    std = returns.std(ddof=1)
    if std <= 0 or not np.isfinite(std):
        return np.nan
    return float(returns.mean() / (std / math.sqrt(len(returns))))


def beta_alpha_corr(returns: pd.Series, benchmark: pd.Series) -> tuple[float, float, float]:
    frame = pd.DataFrame({"strategy": returns, "benchmark": benchmark}).dropna()
    if len(frame) < 2:
        return np.nan, np.nan, np.nan
    bench_var = float(frame["benchmark"].var(ddof=1))
    correlation = float(frame["strategy"].corr(frame["benchmark"]))
    if bench_var <= 0 or not np.isfinite(bench_var):
        return np.nan, np.nan, correlation
    covariance = float(frame["strategy"].cov(frame["benchmark"]))
    beta = covariance / bench_var
    monthly_alpha = float(frame["strategy"].mean() - beta * frame["benchmark"].mean())
    return beta, 12.0 * monthly_alpha, correlation


def summarize_backtests(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in monthly_returns.groupby(
        ["model", "model_label", "split", "portfolio_type", "cost_bps"],
        sort=True,
        observed=True,
    ):
        returns = part["net_return"].astype("float64")
        benchmark = part["spy_return"].astype("float64")
        beta, alpha, corr = beta_alpha_corr(returns, benchmark)
        rows.append(
            {
                "model": keys[0],
                "model_label": keys[1],
                "split": keys[2],
                "portfolio_type": keys[3],
                "cost_bps": int(keys[4]),
                "months": int(len(part)),
                "cumulative_return": cumulative_return(returns),
                "annualized_return": annualized_return(returns),
                "annualized_volatility": annualized_volatility(returns),
                "sharpe_ratio": sharpe_ratio(returns),
                "sortino_ratio": sortino_ratio(returns),
                "max_drawdown": max_drawdown(returns),
                "calmar_ratio": calmar_ratio(returns),
                "mean_monthly_return": float(returns.mean()),
                "monthly_return_tstat": monthly_return_tstat(returns),
                "hit_rate": float((returns > 0).mean()),
                "skewness": float(returns.skew()) if len(returns) >= 3 else np.nan,
                "excess_kurtosis": float(returns.kurt()) if len(returns) >= 4 else np.nan,
                "average_turnover": float(part["turnover"].mean()),
                "average_number_of_long_stocks": float(part["long_count"].mean()),
                "average_number_of_short_stocks": float(part["short_count"].mean()),
                "beta_to_spy": beta,
                "annualized_alpha_vs_spy": alpha,
                "correlation_with_spy": corr,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["split", "portfolio_type", "cost_bps", "model_label"]).reset_index(drop=True)


def summarize_spy_benchmark(benchmark: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, part in benchmark.groupby("split", sort=True, observed=True):
        returns = part["spy_return"].astype("float64")
        rows.append(
            {
                "benchmark": BENCHMARK_TICKER,
                "split": split,
                "months": int(len(part)),
                "cumulative_return": cumulative_return(returns),
                "annualized_return": annualized_return(returns),
                "annualized_volatility": annualized_volatility(returns),
                "sharpe_ratio": sharpe_ratio(returns),
                "sortino_ratio": sortino_ratio(returns),
                "max_drawdown": max_drawdown(returns),
                "calmar_ratio": calmar_ratio(returns),
                "mean_monthly_return": float(returns.mean()),
                "monthly_return_tstat": monthly_return_tstat(returns),
                "hit_rate": float((returns > 0).mean()),
                "skewness": float(returns.skew()) if len(returns) >= 3 else np.nan,
                "excess_kurtosis": float(returns.kurt()) if len(returns) >= 4 else np.nan,
                "beta_to_spy": 1.0,
                "annualized_alpha_vs_spy": 0.0,
                "correlation_with_spy": 1.0,
            }
        )
    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)


def break_even_annualized_bps(gross_returns: pd.Series, turnover: pd.Series) -> float:
    avg_turnover = float(turnover.mean())
    avg_gross = float(gross_returns.mean())
    if avg_turnover <= 0:
        return np.nan
    if avg_gross <= 0:
        return 0.0
    return 10_000.0 * avg_gross / avg_turnover


def cumulative_break_even_function(cost_rate: float, gross_returns: np.ndarray, turnover: np.ndarray) -> float:
    net = gross_returns - cost_rate * turnover
    if np.any(1.0 + net <= 0):
        return -1.0
    return float(np.prod(1.0 + net) - 1.0)


def break_even_cumulative_bps(gross_returns: pd.Series, turnover: pd.Series) -> float:
    gross = gross_returns.to_numpy(dtype="float64")
    turn = turnover.to_numpy(dtype="float64")
    base = cumulative_break_even_function(0.0, gross, turn)
    if base <= 0:
        return 0.0
    low = 0.0
    high = 0.0005
    value = cumulative_break_even_function(high, gross, turn)
    while value > 0 and high < 5.0:
        low = high
        high *= 2.0
        value = cumulative_break_even_function(high, gross, turn)
    if value > 0:
        return np.nan
    for _ in range(80):
        mid = 0.5 * (low + high)
        value = cumulative_break_even_function(mid, gross, turn)
        if abs(value) < 1e-10:
            return mid * 10_000.0
        if value > 0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high) * 10_000.0


def build_break_even_table(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    gross = monthly_returns.loc[monthly_returns["cost_bps"].eq(0)].copy()
    rows = []
    for keys, part in gross.groupby(["model", "model_label", "split", "portfolio_type"], sort=True, observed=True):
        rows.append(
            {
                "model": keys[0],
                "model_label": keys[1],
                "split": keys[2],
                "portfolio_type": keys[3],
                "break_even_annualized_return_bps": break_even_annualized_bps(part["gross_return"], part["turnover"]),
                "break_even_cumulative_return_bps": break_even_cumulative_bps(part["gross_return"], part["turnover"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "portfolio_type", "model_label"]).reset_index(drop=True)


def attach_split_to_benchmark(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    spy = (
        monthly_returns[["split", "return_month", "spy_return"]]
        .drop_duplicates()
        .sort_values(["split", "return_month"])
        .reset_index(drop=True)
    )
    spy["return_month"] = pd.to_datetime(spy["return_month"], errors="raise")
    spy["cumulative_wealth"] = spy.groupby("split", observed=True)["spy_return"].transform(lambda s: (1.0 + s).cumprod())
    spy["return_month"] = spy["return_month"].dt.date.astype(str)
    return spy


def plot_cumulative_wealth(
    summary_source: pd.DataFrame,
    split: str,
    portfolio_type: str,
    cost_bps: int,
    path: Path,
    title: str,
) -> None:
    part = summary_source.loc[
        summary_source["split"].eq(split)
        & summary_source["portfolio_type"].eq(portfolio_type)
        & summary_source["cost_bps"].eq(cost_bps)
    ].copy()
    if part.empty:
        return
    part["return_month"] = pd.to_datetime(part["return_month"], errors="raise")
    fig, ax = plt.subplots(figsize=(12, 6))
    for model_label, model_part in part.groupby("model_label", sort=True, observed=True):
        wealth = (1.0 + model_part.sort_values("return_month")["net_return"]).cumprod()
        ax.plot(model_part.sort_values("return_month")["return_month"], wealth, linewidth=1.5, label=model_label)
    spy = (
        part[["return_month", "spy_return"]]
        .drop_duplicates()
        .sort_values("return_month")
        .copy()
    )
    spy_wealth = (1.0 + spy["spy_return"]).cumprod()
    ax.plot(spy["return_month"], spy_wealth, linewidth=1.8, color="black", linestyle="--", label=BENCHMARK_TICKER)
    ax.set_title(title)
    ax.set_xlabel("Return month")
    ax.set_ylabel("Cumulative wealth")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_drawdowns(monthly_returns: pd.DataFrame, path: Path) -> None:
    data = monthly_returns.loc[
        monthly_returns["split"].eq("test")
        & monthly_returns["cost_bps"].eq(0)
        & monthly_returns["portfolio_type"].isin(["long_short_top_20", "long_only_top_20"])
    ].copy()
    if data.empty:
        return
    data["return_month"] = pd.to_datetime(data["return_month"], errors="raise")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, portfolio_type in zip(axes, ["long_short_top_20", "long_only_top_20"]):
        part = data.loc[data["portfolio_type"].eq(portfolio_type)]
        for model_label, model_part in part.groupby("model_label", sort=True, observed=True):
            series = model_part.sort_values("return_month")
            ax.plot(
                series["return_month"],
                drawdown_series(series["net_return"].astype("float64")),
                linewidth=1.3,
                label=model_label,
            )
        ax.set_title(f"Test drawdowns: {portfolio_type.replace('_', ' ')}")
        ax.set_ylabel("Drawdown")
        ax.legend(frameon=False, fontsize=8, ncol=2)
    axes[-1].set_xlabel("Return month")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sharpe_bars(summary: pd.DataFrame, path: Path) -> None:
    data = summary.loc[
        summary["split"].eq("test")
        & summary["cost_bps"].eq(0)
        & summary["portfolio_type"].isin(["long_short_top_20", "long_only_top_20"])
    ].copy()
    if data.empty:
        return
    order = ["long_short_top_20", "long_only_top_20"]
    pivot = (
        data.pivot(index="model_label", columns="portfolio_type", values="sharpe_ratio")
        .reindex(columns=order)
        .sort_index()
    )
    x = np.arange(len(pivot.index))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, column in enumerate(order):
        values = pivot[column].to_numpy(dtype="float64")
        ax.bar(x + (idx - 0.5) * width, values, width=width, label=column.replace("_", " "))
    ax.set_xticks(x, pivot.index, rotation=20, ha="right")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Test Sharpe ratios for main strategies")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_break_even_bars(break_even: pd.DataFrame, path: Path) -> None:
    data = break_even.loc[
        break_even["split"].eq("test")
        & break_even["portfolio_type"].isin(["long_short_top_20", "long_only_top_20"])
    ].copy()
    if data.empty:
        return
    order = ["long_short_top_20", "long_only_top_20"]
    pivot = (
        data.pivot(index="model_label", columns="portfolio_type", values="break_even_annualized_return_bps")
        .reindex(columns=order)
        .sort_index()
    )
    x = np.arange(len(pivot.index))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, column in enumerate(order):
        values = pivot[column].to_numpy(dtype="float64")
        ax.bar(x + (idx - 0.5) * width, values, width=width, label=column.replace("_", " "))
    ax.set_xticks(x, pivot.index, rotation=20, ha="right")
    ax.set_ylabel("Break-even one-way cost (bps)")
    ax.set_title("Break-even trading cost by model")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readme(
    path: Path,
    prediction_path: Path,
    models: dict[str, tuple[str, str]],
    monthly_returns: pd.DataFrame,
) -> None:
    test_rows = monthly_returns.loc[monthly_returns["split"].eq("test")]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Baseline Backtests\n\n")
        handle.write(f"- Prediction source: `{prediction_path}`\n")
        handle.write(f"- Models included: {', '.join(label for _, label in models.values())}\n")
        handle.write("- Split coverage: validation and test only\n")
        handle.write("- Formation rule: portfolios use prediction scores at month t and earn `target_ret_1m`\n")
        handle.write("- Portfolio construction: equal-weight top 20% long-only and top/bottom 20% long-short; top/bottom 10% variants also included\n")
        handle.write("- Transaction cost model: `net_return = gross_return - cost_rate * turnover`\n")
        handle.write("- Turnover definition: total traded notional from changes in signed portfolio weights, including initial entry\n")
        handle.write(f"- Test monthly observations: {len(test_rows):,}\n")
        handle.write("\n## Outputs\n\n")
        for name, output_path in OUTPUT_FILES.items():
            handle.write(f"- `{name}`: `{output_path}`\n")
        handle.write("\n## Validation Checks\n\n")
        handle.write("- Duplicate PERMNO-month rows are rejected.\n")
        handle.write("- Missing `target_ret_1m` in validation/test rows is rejected.\n")
        handle.write("- Validation must finish before test starts.\n")
        handle.write("- Score columns are restricted to existing prediction columns and cannot be target labels.\n")
        handle.write("- Portfolio construction reads only prediction scores, identifiers, split, and `target_ret_1m`.\n")


def print_table(title: str, frame: pd.DataFrame, columns: list[str]) -> None:
    print(f"\n{title}", flush=True)
    if frame.empty:
        print("(empty)", flush=True)
        return
    print(frame[columns].to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figure_dir = args.figure_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = resolve_prediction_path(args.predictions)
    log(f"Using prediction file: {prediction_path}")
    predictions, models = load_predictions(prediction_path)
    log(f"Models backtested: {', '.join(label for _, label in models.values())}")

    log("Building gross monthly strategy returns...")
    monthly_frames = []
    for model, (column, label) in models.items():
        monthly_frames.append(backtest_model(predictions, model, label, column))
    gross_monthly = pd.concat(monthly_frames, ignore_index=True)
    if gross_monthly.empty:
        raise ValueError("No monthly strategy returns were generated.")

    log("Fetching SPY benchmark from yfinance...")
    spy_returns = fetch_spy_monthly_returns(pd.to_datetime(gross_monthly["return_month"], errors="raise"))

    log("Applying trading cost scenarios...")
    monthly_returns = expand_costs(gross_monthly, spy_returns)
    performance_summary = summarize_backtests(monthly_returns)
    break_even = build_break_even_table(monthly_returns)
    spy_monthly = attach_split_to_benchmark(monthly_returns)
    spy_summary = summarize_spy_benchmark(spy_monthly)

    log("Writing backtest outputs...")
    monthly_returns.to_csv(output_dir / OUTPUT_FILES["monthly_returns"].name, index=False)
    performance_summary.to_csv(output_dir / OUTPUT_FILES["performance_summary"].name, index=False)
    break_even.to_csv(output_dir / OUTPUT_FILES["break_even"].name, index=False)
    spy_monthly.to_csv(output_dir / OUTPUT_FILES["spy_returns"].name, index=False)
    write_readme(output_dir / OUTPUT_FILES["readme"].name, prediction_path, models, monthly_returns)

    log("Writing figures...")
    plot_cumulative_wealth(
        monthly_returns,
        "test",
        "long_short_top_20",
        0,
        figure_dir / "test_cumulative_wealth_long_short_gross_vs_spy.png",
        "Test cumulative wealth: gross long-short top 20% vs SPY",
    )
    plot_cumulative_wealth(
        monthly_returns,
        "test",
        "long_only_top_20",
        0,
        figure_dir / "test_cumulative_wealth_long_only_gross_vs_spy.png",
        "Test cumulative wealth: gross long-only top 20% vs SPY",
    )
    plot_cumulative_wealth(
        monthly_returns,
        "test",
        "long_short_top_20",
        10,
        figure_dir / "test_cumulative_wealth_10bps_costs.png",
        "Test cumulative wealth: long-short top 20% with 10 bps one-way cost vs SPY",
    )
    plot_cumulative_wealth(
        monthly_returns,
        "test",
        "long_short_top_20",
        50,
        figure_dir / "test_cumulative_wealth_50bps_costs.png",
        "Test cumulative wealth: long-short top 20% with 50 bps one-way cost vs SPY",
    )
    plot_drawdowns(monthly_returns, figure_dir / "test_drawdowns_main_strategies.png")
    plot_sharpe_bars(performance_summary, figure_dir / "test_sharpe_ratio_bar_chart.png")
    plot_break_even_bars(break_even, figure_dir / "test_break_even_trading_cost_bar_chart.png")

    test_long_short = performance_summary.loc[
        performance_summary["split"].eq("test")
        & performance_summary["cost_bps"].eq(0)
        & performance_summary["portfolio_type"].eq("long_short_top_20")
    ].sort_values("annualized_return", ascending=False)
    test_long_only = performance_summary.loc[
        performance_summary["split"].eq("test")
        & performance_summary["cost_bps"].eq(0)
        & performance_summary["portfolio_type"].eq("long_only_top_20")
    ].sort_values("annualized_return", ascending=False)
    test_break_even = break_even.loc[
        break_even["split"].eq("test")
        & break_even["portfolio_type"].isin(["long_short_top_20", "long_only_top_20"])
    ].sort_values(["portfolio_type", "break_even_annualized_return_bps"], ascending=[True, False])
    spy_test = spy_summary.loc[spy_summary["split"].eq("test")].copy()

    print_table(
        "Test gross long-short performance",
        test_long_short,
        [
            "model_label",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "average_turnover",
            "beta_to_spy",
            "annualized_alpha_vs_spy",
        ],
    )
    print_table(
        "Test gross long-only performance",
        test_long_only,
        [
            "model_label",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "average_turnover",
            "beta_to_spy",
            "annualized_alpha_vs_spy",
        ],
    )
    print_table(
        "SPY benchmark performance",
        spy_test,
        [
            "benchmark",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "monthly_return_tstat",
        ],
    )
    print_table(
        "Break-even trading cost table",
        test_break_even,
        [
            "model_label",
            "portfolio_type",
            "break_even_annualized_return_bps",
            "break_even_cumulative_return_bps",
        ],
    )

    print("\nSaved output paths", flush=True)
    for output_path in [
        output_dir / OUTPUT_FILES["monthly_returns"].name,
        output_dir / OUTPUT_FILES["performance_summary"].name,
        output_dir / OUTPUT_FILES["break_even"].name,
        output_dir / OUTPUT_FILES["spy_returns"].name,
        output_dir / OUTPUT_FILES["readme"].name,
        figure_dir,
    ]:
        print(f"- {output_path}", flush=True)


if __name__ == "__main__":
    main()
