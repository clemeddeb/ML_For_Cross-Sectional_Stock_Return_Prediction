#!/usr/bin/env python3
"""Run drawdown-reduction overlays on selected portfolio backtest returns.

This script is intentionally separate from run_portfolio_analysis.py. The raw
portfolio script remains a no-overlay benchmark; this script applies the
notebook 13 drawdown speed/phase controller and an optional volatility cap to
already-computed monthly strategy returns.

The monthly output from run_portfolio_analysis.py does not persist security
weights. As a result, this script uses a transparent monthly turnover
approximation for scaled overlays instead of the notebook's exact
security-weight transaction-cost recomputation.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Path("/tmp/matplotlib-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

plt.style.use("default")
plt.rcParams.update(
    {
        "figure.figsize": (11, 5.5),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


OVERLAY_METHODS = ["dd_speed_phase_control", "dd_speed_phase_vol_target"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run drawdown overlay analysis on selected portfolio specs.")
    parser.add_argument("--monthly-returns", type=Path, default=Path("outputs/backtests/portfolio_monthly_returns_25bps.csv"))
    parser.add_argument("--selected-specs", type=Path, default=Path("outputs/backtests/portfolio_best_specs_validation_25bps.csv"))
    parser.add_argument("--preselection", type=Path, default=Path("outputs/backtests/portfolio_validation_preselection_25bps.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/backtests"))
    parser.add_argument("--cost-bps", type=float, default=25.0)
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument(
        "--selection-mode",
        choices=["validation_candidates", "selected"],
        default="validation_candidates",
        help="Use notebook-style validation candidates or only the selected top specs.",
    )
    parser.add_argument("--all-candidates", action="store_true", help="Evaluate all validation candidates, ignoring --top-n.")
    parser.add_argument("--spec-id", action="append", default=None, help="Specific spec_id to evaluate; can be repeated.")
    parser.add_argument("--reentry-2m-threshold", type=float, default=0.02)
    parser.add_argument("--vol-lookback-months", type=int, default=4)
    parser.add_argument("--vol-min-obs", type=int, default=4)
    parser.add_argument("--normal-vol-lookback-months", type=int, default=36)
    parser.add_argument("--min-normal-vol", type=float, default=0.03)
    parser.add_argument("--max-normal-vol", type=float, default=1.00)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def log(message: str, debug: bool) -> None:
    if debug:
        print(message, flush=True)


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


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    if wealth.empty:
        return wealth
    return wealth / wealth.cummax() - 1.0


def cumulative_wealth_with_start(returns: pd.Series, months: pd.Series) -> pd.Series:
    month_index = pd.to_datetime(months)
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    if len(month_index) == 0:
        return pd.Series(dtype="float64")
    start_month = month_index.iloc[0] - pd.offsets.MonthBegin(1)
    return pd.concat([pd.Series([1.0], index=[start_month]), pd.Series(wealth.to_numpy(), index=month_index)])


def performance_metrics(returns: pd.Series) -> dict[str, float]:
    dd = drawdown_series(returns)
    return {
        "months": int(returns.dropna().shape[0]),
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "average_drawdown": float(dd.mean()) if not dd.empty else np.nan,
        "max_drawdown": float(dd.min()) if not dd.empty else np.nan,
        "terminal_wealth": float((1.0 + returns.fillna(0.0)).prod()) if len(returns) else np.nan,
    }


def trailing_cumulative_return(realized_returns: pd.Series, months: int) -> pd.Series:
    return realized_returns.rolling(months, min_periods=months).apply(
        lambda x: float(np.prod(1.0 + x) - 1.0),
        raw=True,
    )


def apply_fast_reentry(lambda_t: float, base_1m: float, base_2m: float, reentry_2m_threshold: float) -> tuple[float, str]:
    note = ""
    if pd.notna(base_1m) and base_1m > 0.0:
        lambda_t = max(lambda_t, 0.75)
        note = "reentry_1m_positive"
    if pd.notna(base_2m) and base_2m > reentry_2m_threshold:
        lambda_t = 1.0
        note = "reentry_2m_gt_2pct"
    return float(np.clip(lambda_t, 0.0, 1.0)), note


def phase_multiplier(risk_band: str) -> float:
    band = str(risk_band)
    if "reentry_2m_gt_2pct" in band:
        return 1.50
    if "reentry_1m_positive" in band:
        return 1.25
    if band.startswith("new_high"):
        return np.inf
    if band.startswith("shallow_drawdown"):
        return 1.10
    if band.startswith("recovery"):
        return 1.25
    if band.startswith("stale_recovery"):
        return 1.05
    if band.startswith("deep_drawdown_not_recovering"):
        return 0.90
    if band.startswith("shock_drawdown") or band.startswith("grinding_drawdown"):
        return 0.75
    return 1.00


def compute_dd_lambda(
    dd_before: float,
    previous_dd: float,
    dd_change_3m: float,
    dd_age: int,
    recovery_age: int,
    wealth: float,
    trough: float,
    base_1m: float,
    base_2m: float,
    reentry_2m_threshold: float,
) -> tuple[float, str, float]:
    dd_change_1m = dd_before - previous_dd
    in_drawdown = dd_before < 0.0
    above_trough = wealth > trough * 1.01
    improving = dd_change_1m > 0.0
    recovering = in_drawdown and above_trough and improving
    shock_drawdown = in_drawdown and (dd_change_1m <= -0.10 or (pd.notna(base_1m) and base_1m <= -0.10))
    grinding_drawdown = in_drawdown and dd_age >= 3 and (
        (pd.notna(dd_change_3m) and dd_change_3m <= -0.05)
        or (not improving and pd.notna(base_2m) and base_2m < 0.0)
    )
    stale_recovery = recovering and recovery_age >= 3 and dd_before < -0.05 and not (
        pd.notna(base_2m) and base_2m > reentry_2m_threshold
    )

    if not in_drawdown:
        lambda_t = 1.0
        state = "new_high"
    elif shock_drawdown:
        lambda_t = 0.50
        state = "shock_drawdown"
    elif grinding_drawdown:
        lambda_t = 0.50
        state = "grinding_drawdown"
    elif stale_recovery:
        lambda_t = 0.75
        state = "stale_recovery"
    elif recovering:
        lambda_t = 0.75
        state = "recovery"
    elif dd_before <= -0.25:
        lambda_t = 0.65
        state = "deep_drawdown_not_recovering"
    else:
        lambda_t = 0.85
        state = "shallow_drawdown"

    lambda_t, note = apply_fast_reentry(lambda_t, base_1m, base_2m, reentry_2m_threshold)
    if note:
        state = f"{state}_{note}"
    return float(np.clip(lambda_t, 0.0, 1.0)), state, dd_change_1m


def boolean_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def select_specs(
    monthly: pd.DataFrame,
    selected: pd.DataFrame,
    preselection: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if args.spec_id:
        spec_ids = list(dict.fromkeys(args.spec_id))
        missing = sorted(set(spec_ids) - set(monthly["spec_id"]))
        if missing:
            raise ValueError(f"Requested spec_id values are not in monthly returns: {missing}")
        rows = monthly.loc[monthly["spec_id"].isin(spec_ids), ["spec_id", "model_family", "score_label", "rule", "q", "gate_type", "threshold", "z_threshold"]]
        return rows.drop_duplicates("spec_id").reset_index(drop=True)
    if args.selection_mode == "validation_candidates" and not preselection.empty:
        candidates = preselection.copy()
        if "pass_validation_filters" in candidates.columns:
            candidates = candidates.loc[boolean_mask(candidates["pass_validation_filters"])].copy()
        sort_cols = [col for col in ["validation_rank", "net_sharpe", "net_annualized_return"] if col in candidates.columns]
        if sort_cols:
            ascending = [True if col == "validation_rank" else False for col in sort_cols]
            candidates = candidates.sort_values(sort_cols, ascending=ascending)
        if not args.all_candidates:
            candidates = candidates.head(args.top_n)
        return candidates.drop_duplicates("spec_id").reset_index(drop=True)
    if selected.empty:
        rows = monthly[["spec_id", "model_family", "score_label", "rule", "q", "gate_type", "threshold", "z_threshold"]]
        return rows.drop_duplicates("spec_id").head(args.top_n).reset_index(drop=True)
    return (selected if args.all_candidates else selected.head(args.top_n)).copy()


def apply_overlay(part: pd.DataFrame, method: str, args: argparse.Namespace) -> pd.DataFrame:
    part = part.sort_values("month").reset_index(drop=True).copy()
    base_net = part["net_return"].fillna(0.0)
    base_gross = part["gross_return"].fillna(0.0)
    realized_base_1m = base_net.shift(1)
    realized_base_2m = trailing_cumulative_return(base_net.shift(1), 2)
    realized_available = base_net.shift(1)
    realized_vol = realized_available.rolling(args.vol_lookback_months, min_periods=args.vol_min_obs).std() * math.sqrt(12.0)
    normal_vol = realized_vol.rolling(args.normal_vol_lookback_months, min_periods=args.vol_min_obs).median()
    normal_vol = normal_vol.combine_first(realized_vol.expanding(min_periods=args.vol_min_obs).median())
    normal_vol = normal_vol.clip(lower=args.min_normal_vol, upper=args.max_normal_vol)

    rows = []
    wealth = 1.0
    peak = 1.0
    trough = 1.0
    dd_age = 0
    recovery_age = 0
    dd_before_history: list[float] = []
    previous_lambda: float | None = None
    previous_gross_exposure = 0.0
    cost_rate = args.cost_bps / 10_000.0

    for idx, row in part.iterrows():
        dd_before = wealth / peak - 1.0
        previous_dd = dd_before_history[-1] if dd_before_history else 0.0
        dd_change_3m = dd_before - dd_before_history[-3] if len(dd_before_history) >= 3 else np.nan
        lambda_dd, state, dd_change_1m = compute_dd_lambda(
            dd_before=dd_before,
            previous_dd=previous_dd,
            dd_change_3m=dd_change_3m,
            dd_age=dd_age,
            recovery_age=recovery_age,
            wealth=wealth,
            trough=trough,
            base_1m=realized_base_1m.iloc[idx],
            base_2m=realized_base_2m.iloc[idx],
            reentry_2m_threshold=args.reentry_2m_threshold,
        )
        target_vol = normal_vol.iloc[idx] * phase_multiplier(state) if pd.notna(normal_vol.iloc[idx]) else np.nan
        if not np.isfinite(target_vol):
            target_vol = np.nan
        if method == "dd_speed_phase_control":
            vol_lambda = 1.0
            lambda_t = lambda_dd
        elif method == "dd_speed_phase_vol_target":
            if pd.notna(target_vol) and pd.notna(realized_vol.iloc[idx]) and realized_vol.iloc[idx] > 0:
                vol_lambda = float(min(target_vol / realized_vol.iloc[idx], 1.0))
            else:
                vol_lambda = 1.0
            lambda_t = float(np.clip(min(lambda_dd, vol_lambda, 1.0), 0.0, 1.0))
        else:
            raise ValueError(f"Unknown overlay method: {method}")

        gross_exposure = float(row.get("gross_exposure", 1.0))
        base_turnover = float(row.get("turnover", np.nan))
        if not np.isfinite(base_turnover):
            base_turnover = gross_exposure if previous_lambda is None else 0.0
        if previous_lambda is None:
            overlay_turnover = lambda_t * gross_exposure
        else:
            overlay_turnover = lambda_t * base_turnover + abs(lambda_t - previous_lambda) * previous_gross_exposure
        overlay_gross_return = lambda_t * float(base_gross.iloc[idx])
        overlay_trading_cost = cost_rate * overlay_turnover
        overlay_net_return = overlay_gross_return - overlay_trading_cost

        wealth *= 1.0 + overlay_net_return
        if wealth >= peak:
            peak = wealth
            trough = wealth
            dd_age = 0
            recovery_age = 0
        else:
            dd_age += 1
            if wealth < trough:
                trough = wealth
                recovery_age = 0
            elif wealth > trough * 1.01:
                recovery_age += 1
        dd_before_history.append(dd_before)

        out = row.to_dict()
        out.update(
            {
                "overlay_method": method,
                "overlay_label": "DD speed/phase" if method == "dd_speed_phase_control" else "DD speed/phase + vol target",
                "lambda_dd": lambda_dd,
                "lambda_t": lambda_t,
                "vol_lambda": vol_lambda,
                "risk_band": state,
                "control_signal": dd_change_1m,
                "realized_vol_4m_no_leak": realized_vol.iloc[idx],
                "target_vol": target_vol,
                "vol_cap_binding": bool(method == "dd_speed_phase_vol_target" and vol_lambda < lambda_dd),
                "overlay_gross_return": overlay_gross_return,
                "overlay_turnover": overlay_turnover,
                "overlay_trading_cost": overlay_trading_cost,
                "overlay_net_return": overlay_net_return,
                "overlay_gross_exposure": lambda_t * gross_exposure,
                "overlay_net_exposure": lambda_t * float(row.get("net_exposure", 0.0)),
                "overlay_drawdown_after_return": wealth / peak - 1.0,
                "turnover_approximation": "monthly_scaled_base_turnover_plus_lambda_change",
            }
        )
        rows.append(out)
        previous_lambda = lambda_t
        previous_gross_exposure = gross_exposure

    return pd.DataFrame(rows)


def summarize_overlay(spec: pd.Series, split: str, method: str, overlay_monthly: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {
        "spec_id": spec["spec_id"],
        "split": split,
        "model_family": spec.get("model_family"),
        "score_label": spec.get("score_label"),
        "rule": spec.get("rule"),
        "q": spec.get("q"),
        "gate_type": spec.get("gate_type"),
        "threshold": spec.get("threshold"),
        "z_threshold": spec.get("z_threshold"),
        "validation_rank": spec.get("validation_rank"),
        "overlay_method": method,
        "method_label": overlay_monthly["overlay_label"].iloc[0],
        "overlay_average_lambda": overlay_monthly["lambda_t"].mean(),
        "overlay_min_lambda": overlay_monthly["lambda_t"].min(),
        "overlay_months_below_full_exposure": int(overlay_monthly["lambda_t"].lt(1.0).sum()),
        "overlay_months_vol_cap_binding": int(overlay_monthly["vol_cap_binding"].sum()),
        "overlay_average_turnover": overlay_monthly["overlay_turnover"].mean(),
        "overlay_average_trading_cost": overlay_monthly["overlay_trading_cost"].mean(),
    }
    for key, value in performance_metrics(overlay_monthly["net_return"]).items():
        row[f"base_{key}"] = value
    for key, value in performance_metrics(overlay_monthly["overlay_net_return"]).items():
        row[f"overlay_{key}"] = value
    for key in ["annualized_return", "annualized_volatility", "sharpe", "average_drawdown", "max_drawdown", "terminal_wealth"]:
        row[f"delta_{key}"] = row[f"overlay_{key}"] - row[f"base_{key}"]
    return row


def aggregate_change_table(summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in summary.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(
            {
                "n_specs": int(part["spec_id"].nunique()),
                "mean_delta_return": part["delta_annualized_return"].mean(),
                "median_delta_return": part["delta_annualized_return"].median(),
                "mean_delta_volatility": part["delta_annualized_volatility"].mean(),
                "median_delta_volatility": part["delta_annualized_volatility"].median(),
                "mean_delta_sharpe": part["delta_sharpe"].mean(),
                "median_delta_sharpe": part["delta_sharpe"].median(),
                "mean_delta_average_drawdown": part["delta_average_drawdown"].mean(),
                "median_delta_average_drawdown": part["delta_average_drawdown"].median(),
                "mean_delta_max_drawdown": part["delta_max_drawdown"].mean(),
                "median_delta_max_drawdown": part["delta_max_drawdown"].median(),
                "mean_delta_terminal_wealth": part["delta_terminal_wealth"].mean(),
                "median_delta_terminal_wealth": part["delta_terminal_wealth"].median(),
                "mean_overlay_lambda": part["overlay_average_lambda"].mean(),
                "mean_months_below_full_exposure": part["overlay_months_below_full_exposure"].mean(),
                "share_return_improved": part["delta_annualized_return"].gt(0).mean(),
                "share_volatility_reduced": part["delta_annualized_volatility"].lt(0).mean(),
                "share_sharpe_improved": part["delta_sharpe"].gt(0).mean(),
                "share_average_drawdown_improved": part["delta_average_drawdown"].gt(0).mean(),
                "share_max_drawdown_improved": part["delta_max_drawdown"].gt(0).mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def strategy_label(row: pd.Series) -> str:
    score = str(row.get("score_label", "")).split("__")[-1].replace("_", " ")
    rule = str(row.get("rule", "")).replace("_", " ")
    q = row.get("q")
    threshold = row.get("threshold")
    parts = [str(row.get("model_family", "Strategy")), score[:28], rule]
    if pd.notna(q):
        parts.append(f"q={float(q):.0%}")
    if str(row.get("gate_type")) == "absolute" and pd.notna(threshold):
        parts.append(f"thr={float(threshold):g}")
    return " | ".join([p for p in parts if p])


def compact_bar_label(row: pd.Series) -> str:
    family = str(row.get("model_family", "Strategy"))
    q = row.get("q")
    rule = str(row.get("rule", "")).replace("gross_normalized_long_short", "GN L/S").replace("long_short_130_30", "130/30")
    method = "DD" if row.get("overlay_method") == "dd_speed_phase_control" else "DD+vol"
    q_label = f" q={float(q):.0%}" if pd.notna(q) else ""
    return f"{family} {rule}{q_label} {method}"


def write_figures(output_dir: Path, monthly: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    test_summary = summary.loc[summary["split"].eq("test")].copy()
    if test_summary.empty:
        return paths

    sort_cols = [col for col in ["validation_rank", "spec_id", "overlay_method"] if col in test_summary.columns]
    plot_specs = test_summary.sort_values(sort_cols)["spec_id"].drop_duplicates().head(5).tolist()
    plot_monthly = monthly.loc[monthly["split"].eq("test") & monthly["spec_id"].isin(plot_specs)].copy()

    for spec_id, part in plot_monthly.groupby("spec_id", sort=False):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        base = part.loc[part["overlay_method"].eq("dd_speed_phase_control")].sort_values("month")
        if base.empty:
            continue
        label = strategy_label(base.iloc[0])
        base_wealth = cumulative_wealth_with_start(base["net_return"], base["month"])
        axes[0].plot(base_wealth.index, base_wealth.values, color="#222222", linewidth=1.8, label="Base")
        axes[1].plot(pd.to_datetime(base["month"]), drawdown_series(base["net_return"]).values, color="#222222", linewidth=1.5, label="Base")
        for method, color in [("dd_speed_phase_control", "#4C78A8"), ("dd_speed_phase_vol_target", "#F58518")]:
            method_rows = part.loc[part["overlay_method"].eq(method)].sort_values("month")
            if method_rows.empty:
                continue
            method_label = method_rows["overlay_label"].iloc[0]
            wealth = cumulative_wealth_with_start(method_rows["overlay_net_return"], method_rows["month"])
            dd = drawdown_series(method_rows["overlay_net_return"])
            axes[0].plot(wealth.index, wealth.values, color=color, linewidth=1.6, label=method_label)
            axes[1].plot(pd.to_datetime(method_rows["month"]), dd.values, color=color, linewidth=1.5, label=method_label)
        axes[0].axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
        axes[0].set_title("Test cumulative wealth")
        axes[0].set_ylabel("Cumulative wealth")
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axes[1].set_title("Test drawdown")
        axes[1].set_ylabel("Drawdown")
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        axes[1].legend(frameon=False, fontsize=8)
        fig.suptitle(label, y=1.02, fontsize=12)
        fig.tight_layout()
        path = fig_dir / f"drawdown_overlay_test_{safe_slug(spec_id)}_25bps.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    bars = test_summary.loc[test_summary["spec_id"].isin(plot_specs)].copy()
    if not bars.empty:
        bars["label"] = bars.apply(compact_bar_label, axis=1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for ax, (col, title) in zip(
            axes,
            [
                ("delta_max_drawdown", "Change in max DD"),
                ("delta_sharpe", "Change in Sharpe"),
                ("overlay_average_lambda", "Average lambda"),
            ],
        ):
            ax.bar(bars["label"], bars[col])
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=55, labelsize=7)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        path = fig_dir / "drawdown_overlay_test_metric_bars_25bps.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def safe_slug(text: str) -> str:
    out = []
    for char in str(text).lower():
        if char.isalnum():
            out.append(char)
        elif char in {"_", "-", "."}:
            out.append("_")
    return "".join(out).strip("_")


def write_report(
    output_dir: Path,
    selected_specs: pd.DataFrame,
    summary: pd.DataFrame,
    average_changes: pd.DataFrame,
    average_changes_by_family: pd.DataFrame,
    figure_paths: list[Path],
) -> Path:
    path = output_dir / "drawdown_overlay_report_25bps.md"
    top = summary.sort_values(["split", "overlay_method", "delta_max_drawdown"], ascending=[True, True, False]).head(20)
    lines = [
        "# Drawdown Overlay Analysis at 25 bps",
        "",
        "Generated by `scripts/backtesting/run_drawdown_overlay_analysis.py`.",
        "",
        "## Scope",
        "",
        "- Applies the notebook 13 DD speed/phase controller and DD speed/phase plus volatility target overlay.",
        "- Uses selected monthly raw strategy returns from `portfolio_monthly_returns_25bps.csv`.",
        "- Keeps this overlay analysis separate from the raw portfolio-analysis script.",
        "- Transaction costs are approximate because the production monthly returns do not persist security-level weights.",
        "",
        "## Run Summary",
        "",
        f"- Specs evaluated: {selected_specs['spec_id'].nunique():,}",
        f"- Split/method rows: {len(summary):,}",
        f"- Figures written: {len(figure_paths):,}",
        "",
        "## Average Changes",
        "",
        "```text",
        average_changes.to_string(index=False, max_cols=22, max_colwidth=70) if not average_changes.empty else "No rows.",
        "```",
        "",
        "## Average Changes by Family",
        "",
        "```text",
        average_changes_by_family.to_string(index=False, max_cols=22, max_colwidth=70) if not average_changes_by_family.empty else "No rows.",
        "```",
        "",
        "## Top Rows",
        "",
        "```text",
        top.to_string(index=False, max_cols=16, max_colwidth=70) if not top.empty else "No rows.",
        "```",
        "",
        "## Caveat",
        "",
        "The notebook recomputed scaled transaction costs from security-level weights. This script uses `lambda_t * base_turnover + |lambda_t - lambda_{t-1}| * previous_gross_exposure`, so overlay cost estimates are labeled as approximate.",
        "",
        "## Figures",
        "",
    ]
    lines.extend([f"- `{p}`" for p in figure_paths] if figure_paths else ["- None"])
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_csv(args.monthly_returns, parse_dates=["month"])
    selected = pd.read_csv(args.selected_specs) if args.selected_specs.exists() else pd.DataFrame()
    preselection = pd.read_csv(args.preselection) if args.preselection.exists() else pd.DataFrame()
    selected_specs = select_specs(monthly, selected, preselection, args)

    overlay_frames = []
    summary_rows = []
    for idx, spec in selected_specs.iterrows():
        spec_id = spec["spec_id"]
        log(f"Overlay spec {idx + 1}/{len(selected_specs)}: {spec_id}", args.debug)
        for split, part in monthly.loc[monthly["spec_id"].eq(spec_id)].groupby("split", sort=True):
            for method in OVERLAY_METHODS:
                overlay = apply_overlay(part, method, args)
                overlay_frames.append(overlay)
                summary_rows.append(summarize_overlay(spec, split, method, overlay))

    if not overlay_frames:
        raise RuntimeError("No overlay rows were produced.")

    overlay_monthly = pd.concat(overlay_frames, ignore_index=True)
    overlay_summary = pd.DataFrame(summary_rows)
    average_changes = aggregate_change_table(overlay_summary, ["split", "overlay_method", "method_label"])
    average_changes_by_family = aggregate_change_table(
        overlay_summary,
        ["split", "model_family", "overlay_method", "method_label"],
    )
    compact_cols = [
        "spec_id",
        "split",
        "model_family",
        "score_label",
        "rule",
        "q",
        "gate_type",
        "threshold",
        "overlay_method",
        "method_label",
        "base_annualized_return",
        "overlay_annualized_return",
        "delta_annualized_return",
        "base_annualized_volatility",
        "overlay_annualized_volatility",
        "delta_annualized_volatility",
        "base_sharpe",
        "overlay_sharpe",
        "delta_sharpe",
        "base_average_drawdown",
        "overlay_average_drawdown",
        "delta_average_drawdown",
        "base_max_drawdown",
        "overlay_max_drawdown",
        "delta_max_drawdown",
        "overlay_average_lambda",
        "overlay_average_turnover",
    ]
    compact = overlay_summary[compact_cols].sort_values(["split", "spec_id", "overlay_method"]).reset_index(drop=True)

    monthly_path = args.output_dir / "drawdown_overlay_monthly_returns_25bps.csv"
    summary_path = args.output_dir / "drawdown_overlay_strategy_results_25bps.csv"
    compact_path = args.output_dir / "drawdown_overlay_summary_25bps.csv"
    average_path = args.output_dir / "drawdown_overlay_average_changes_25bps.csv"
    average_by_family_path = args.output_dir / "drawdown_overlay_average_changes_by_family_25bps.csv"
    overlay_monthly.to_csv(monthly_path, index=False)
    overlay_summary.to_csv(summary_path, index=False)
    compact.to_csv(compact_path, index=False)
    average_changes.to_csv(average_path, index=False)
    average_changes_by_family.to_csv(average_by_family_path, index=False)
    figure_paths = write_figures(args.output_dir, overlay_monthly, overlay_summary)
    report_path = write_report(args.output_dir, selected_specs, overlay_summary, average_changes, average_changes_by_family, figure_paths)

    print("Drawdown overlay analysis completed.")
    print(f"Specs evaluated: {selected_specs['spec_id'].nunique()}")
    print(f"Overlay rows: {len(overlay_summary)}")
    print("Files created:")
    for path in [monthly_path, summary_path, compact_path, average_path, average_by_family_path, report_path, *figure_paths]:
        print(f"- {path}")
    print("Average changes:")
    print(average_changes.to_string(index=False))
    print("Top test overlay rows by max drawdown improvement:")
    cols = ["model_family", "score_label", "rule", "q", "threshold", "method_label", "delta_max_drawdown", "delta_sharpe"]
    print(
        overlay_summary.loc[overlay_summary["split"].eq("test")]
        .sort_values("delta_max_drawdown", ascending=False)[cols]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
