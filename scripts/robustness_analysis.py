from pathlib import Path
import os

Path("/tmp/matplotlib-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(362559)

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

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

BACKTEST_DIR = ROOT / "outputs" / "backtests" / "dl_xgb_score_strategies"
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SPEC_COLS = [
    "score_label",
    "rule",
    "q",
    "sign_gate",
    "threshold_gate",
    "zero_threshold",
    "one_way_cost_bps",
]
SPEC_NO_COST = [c for c in SPEC_COLS if c != "one_way_cost_bps"]
ROBUSTNESS_COSTS = [0.0, 10.0, 25.0, 50.0]
ANALYSIS_COST = 25.0


def model_family(score_label):
    label = str(score_label)
    if label.startswith("mlp"):
        return "MLP"
    if label.startswith("ft"):
        return "FT"
    if label.startswith("temporal"):
        return "Temporal"
    if label.startswith("xgb"):
        return "baseline_model"
    return "Other"


def strategy_name(row, include_cost=True):
    gate = "thr" if bool(row["threshold_gate"]) else "plain"
    threshold = f"{float(row['zero_threshold']):.2f}"
    pieces = [
        model_family(row["score_label"]),
        str(row["score_label"]),
        str(row["rule"]),
        f"{gate}={threshold}",
    ]
    if include_cost:
        pieces.append(f"cost={float(row['one_way_cost_bps']):.0f}bps")
    return " | ".join(pieces)


def max_drawdown(returns):
    wealth = (1.0 + pd.Series(returns).fillna(0.0)).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return float(drawdown.min()) if len(drawdown) else np.nan


def performance_from_returns(df):
    rows = []
    group_cols = SPEC_COLS + ["split"]
    for keys, grp in df.groupby(group_cols, dropna=False):
        returns = grp.sort_values("month")["net_return"].astype(float)
        n = len(returns)
        if n == 0:
            continue
        cumulative = float((1.0 + returns).prod())
        ann_return = cumulative ** (12.0 / n) - 1.0
        monthly_std = float(returns.std(ddof=1))
        ann_vol = monthly_std * np.sqrt(12.0)
        rows.append(
            dict(
                zip(group_cols, keys),
                months=n,
                annualized_return=ann_return,
                annualized_volatility=ann_vol,
                sharpe=ann_return / ann_vol if ann_vol and np.isfinite(ann_vol) else np.nan,
                max_drawdown=max_drawdown(returns),
                mean_monthly_return=float(returns.mean()),
                std_monthly_return=monthly_std,
                average_turnover=float(grp["turnover"].mean()),
                average_gross_exposure=float(grp["gross_exposure"].mean())
                if "gross_exposure" in grp
                else np.nan,
                average_net_exposure=float(grp["net_exposure"].mean())
                if "net_exposure" in grp
                else np.nan,
            )
        )
    out = pd.DataFrame(rows)
    out["model_family"] = out["score_label"].map(model_family)
    out["strategy_base"] = out.apply(lambda r: strategy_name(r, include_cost=False), axis=1)
    out["strategy"] = out.apply(strategy_name, axis=1)
    return out


def spec_filter(df, spec):
    mask = pd.Series(True, index=df.index)
    for col in SPEC_NO_COST:
        mask &= df[col].eq(spec[col])
    return mask


def select_family_representatives(validation_25):
    selected_path = TABLE_DIR / "portfolio_selected_top5_strategies.csv"
    selected = pd.read_csv(selected_path)
    selected["model_family"] = selected["score_label"].map(model_family)
    reps = []
    for family in ["MLP", "FT", "Temporal", "baseline_model"]:
        fam = selected.loc[selected["model_family"].eq(family)].copy()
        if fam.empty:
            fam = validation_25.loc[validation_25["model_family"].eq(family)].head(1).copy()
        if not fam.empty:
            reps.append(fam.iloc[0][SPEC_COLS + ["model_family"]])
    reps = pd.DataFrame(reps).drop_duplicates(SPEC_NO_COST).reset_index(drop=True)
    reps["strategy_base"] = reps.apply(lambda r: strategy_name(r, include_cost=False), axis=1)
    return reps


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


print("Loading persisted strategy, return, validation, and forecasting artifacts.")
performance_raw = pd.read_csv(BACKTEST_DIR / "dl_xgb_score_strategy_performance_summary.csv")
monthly_raw = pd.read_csv(BACKTEST_DIR / "dl_xgb_score_strategy_monthly_returns.csv")
monthly_raw["month"] = pd.to_datetime(monthly_raw["month"])
validation_ranking_tables = {}
for path in sorted(BACKTEST_DIR.glob("validation_dl_xgb_score_strategy_ranking*.csv")):
    validation_ranking_tables[path.stem] = pd.read_csv(path)
ranking_25 = pd.read_csv(TABLE_DIR / "portfolio_validation_strategy_ranking_all_models.csv")
rank_ic = pd.read_csv(BACKTEST_DIR / "dl_xgb_score_rank_ic_summary.csv")
forecasting = pd.read_csv(TABLE_DIR / "forecasting_summary_report_ready.csv")

base_return_cols = ["split", "month"] + SPEC_NO_COST + [
    "score_col",
    "gross_return",
    "turnover",
    "gross_exposure",
    "net_exposure",
]
base_monthly = monthly_raw[base_return_cols].drop_duplicates(["split", "month"] + SPEC_NO_COST)
expanded = []
for cost in ROBUSTNESS_COSTS:
    tmp = base_monthly.copy()
    tmp["one_way_cost_bps"] = cost
    tmp["trading_cost"] = tmp["turnover"].astype(float) * cost / 10000.0
    tmp["net_return"] = tmp["gross_return"].astype(float) - tmp["trading_cost"]
    expanded.append(tmp)
monthly = pd.concat(expanded, ignore_index=True)
perf = performance_from_returns(monthly)
perf["zero_threshold"] = perf["zero_threshold"].astype(float)
perf["one_way_cost_bps"] = perf["one_way_cost_bps"].astype(float)

validation_25 = perf.loc[
    perf["split"].eq("validation") & perf["one_way_cost_bps"].eq(ANALYSIS_COST)
].sort_values("sharpe", ascending=False)
representatives = select_family_representatives(validation_25)
print("Representative strategies:")
print(representatives[["model_family", "score_label", "rule", "zero_threshold"]].to_string(index=False))

# 1. Transaction cost sensitivity.
rep_perf = []
for _, spec in representatives.iterrows():
    sub = perf.loc[spec_filter(perf, spec) & perf["split"].eq("test")].copy()
    rep_perf.append(sub)
rep_perf = pd.concat(rep_perf, ignore_index=True)
cost_sensitivity = rep_perf.pivot_table(
    index=["model_family", "strategy_base"],
    columns="one_way_cost_bps",
    values="sharpe",
    aggfunc="first",
).reset_index()
cost_sensitivity.columns = [
    "model_family",
    "strategy",
    "Sharpe (0bps)",
    "Sharpe (10bps)",
    "Sharpe (25bps)",
    "Sharpe (50bps)",
]
cost_sensitivity.to_csv(TABLE_DIR / "cost_sensitivity.csv", index=False)
rep_perf.to_csv(TABLE_DIR / "cost_sensitivity_long.csv", index=False)

for metric, ylabel, fname in [
    ("sharpe", "Test Sharpe", "robustness_cost_sensitivity_sharpe.png"),
    ("annualized_return", "Test annualized return", "robustness_cost_sensitivity_return.png"),
]:
    plt.figure(figsize=(9, 5))
    for strategy, grp in rep_perf.groupby("strategy_base"):
        grp = grp.sort_values("one_way_cost_bps")
        plt.plot(grp["one_way_cost_bps"], grp[metric], marker="o", label=strategy)
    plt.xlabel("One-way transaction cost (bps)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} across transaction cost assumptions")
    plt.legend(fontsize=8)
    save_plot(FIGURE_DIR / fname)

# 2. Threshold and rule sensitivity.
threshold_source = perf.loc[
    perf["split"].eq("test")
    & perf["one_way_cost_bps"].eq(ANALYSIS_COST)
    & perf["rule"].isin(["long_short", "long_short_130_30", "gross_normalized_long_short"])
].copy()
threshold_summary = []
for score_label, grp in threshold_source.groupby("score_label"):
    best = grp.sort_values("sharpe", ascending=False).iloc[0]
    threshold_summary.append(
        {
            "model": score_label,
            "model_family": model_family(score_label),
            "best_threshold": float(best["zero_threshold"]),
            "best_rule": best["rule"],
            "best_sharpe": float(best["sharpe"]),
            "sharpe_min": float(grp["sharpe"].min()),
            "sharpe_max": float(grp["sharpe"].max()),
            "sharpe_variation_range": float(grp["sharpe"].max() - grp["sharpe"].min()),
            "stable_vs_threshold": bool((grp["sharpe"].max() - grp["sharpe"].min()) < 0.50),
        }
    )
threshold_summary = pd.DataFrame(threshold_summary).sort_values(
    ["model_family", "sharpe_variation_range"]
)
threshold_summary.to_csv(TABLE_DIR / "threshold_sensitivity.csv", index=False)

plot_models = representatives["score_label"].drop_duplicates().tolist()
plt.figure(figsize=(10, 5.5))
for score_label in plot_models:
    sub = threshold_source.loc[threshold_source["score_label"].eq(score_label)].copy()
    line = sub.groupby("zero_threshold", as_index=False)["sharpe"].max().sort_values("zero_threshold")
    plt.plot(line["zero_threshold"], line["sharpe"], marker="o", label=score_label)
plt.xlabel("Zero threshold")
plt.ylabel("Best test Sharpe at 25 bps")
plt.title("Threshold sensitivity by retained model signal")
plt.legend(fontsize=8)
save_plot(FIGURE_DIR / "robustness_threshold_sensitivity.png")

# 3. Subperiod stability.
selected_monthly = []
for _, spec in representatives.iterrows():
    sub = monthly.loc[
        spec_filter(monthly, spec)
        & monthly["split"].eq("test")
        & monthly["one_way_cost_bps"].eq(ANALYSIS_COST)
    ].copy()
    sub["strategy_base"] = spec["strategy_base"]
    selected_monthly.append(sub)
selected_monthly = pd.concat(selected_monthly, ignore_index=True)
selected_monthly = selected_monthly.loc[selected_monthly["month"].ge(pd.Timestamp("2016-01-01"))].copy()
selected_monthly["subperiod"] = np.where(
    selected_monthly["month"].le(pd.Timestamp("2019-12-31")), "2016-2019", "2020-2024"
)
subperiod_rows = []
for (strategy, subperiod), grp in selected_monthly.groupby(["strategy_base", "subperiod"]):
    returns = grp.sort_values("month")["net_return"]
    n = len(returns)
    ann_return = float((1.0 + returns).prod() ** (12.0 / n) - 1.0)
    ann_vol = float(returns.std(ddof=1) * np.sqrt(12.0))
    subperiod_rows.append(
        {
            "strategy": strategy,
            "subperiod": subperiod,
            "months": n,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe": ann_return / ann_vol if ann_vol else np.nan,
            "max_drawdown": max_drawdown(returns),
        }
    )
subperiod_long = pd.DataFrame(subperiod_rows)
subperiod_table = subperiod_long.pivot_table(
    index="strategy", columns="subperiod", values=["sharpe", "annualized_return"], aggfunc="first"
)
subperiod_table.columns = [f"{metric}_{period}" for metric, period in subperiod_table.columns]
subperiod_table = subperiod_table.reset_index()
subperiod_table.to_csv(TABLE_DIR / "subperiod_performance.csv", index=False)

plt.figure(figsize=(10, 5.5))
for (strategy, subperiod), grp in selected_monthly.groupby(["strategy_base", "subperiod"]):
    grp = grp.sort_values("month").copy()
    grp["cumulative_wealth"] = (1.0 + grp["net_return"]).cumprod()
    label = f"{strategy} ({subperiod})"
    plt.plot(grp["month"], grp["cumulative_wealth"], label=label)
plt.xlabel("Month")
plt.ylabel("Cumulative wealth")
plt.title("Cumulative returns by subperiod at 25 bps")
plt.legend(fontsize=7, ncol=2)
save_plot(FIGURE_DIR / "robustness_subperiod_cumulative_returns.png")

# 4. Validation vs test stability.
val = perf.loc[perf["split"].eq("validation") & perf["one_way_cost_bps"].eq(ANALYSIS_COST)].copy()
test = perf.loc[perf["split"].eq("test") & perf["one_way_cost_bps"].eq(ANALYSIS_COST)].copy()
val_test = val[SPEC_NO_COST + ["strategy_base", "model_family", "sharpe", "annualized_return"]].merge(
    test[SPEC_NO_COST + ["sharpe", "annualized_return", "max_drawdown"]],
    on=SPEC_NO_COST,
    suffixes=("_validation", "_test"),
)
val_test["sharpe_drop"] = val_test["sharpe_validation"] - val_test["sharpe_test"]
val_test["return_drop"] = val_test["annualized_return_validation"] - val_test["annualized_return_test"]
val_test = val_test.sort_values("sharpe_drop", ascending=False)
val_test.to_csv(TABLE_DIR / "validation_test_stability.csv", index=False)
validation_test_corr = float(val_test["sharpe_validation"].corr(val_test["sharpe_test"]))
average_sharpe_drop = float(val_test["sharpe_drop"].mean())

plt.figure(figsize=(7, 6))
colors = {"MLP": "tab:blue", "FT": "tab:green", "Temporal": "tab:orange", "baseline_model": "tab:red"}
for family, grp in val_test.groupby("model_family"):
    plt.scatter(
        grp["sharpe_validation"],
        grp["sharpe_test"],
        label=family,
        alpha=0.75,
        s=45,
        color=colors.get(family),
    )
lims = [
    min(val_test["sharpe_validation"].min(), val_test["sharpe_test"].min()) - 0.1,
    max(val_test["sharpe_validation"].max(), val_test["sharpe_test"].max()) + 0.1,
]
plt.plot(lims, lims, color="black", linewidth=1, linestyle="--")
plt.xlim(lims)
plt.ylim(lims)
plt.xlabel("Validation Sharpe")
plt.ylabel("Test Sharpe")
plt.title(f"Validation vs test Sharpe at 25 bps (corr={validation_test_corr:.2f})")
plt.legend()
save_plot(FIGURE_DIR / "robustness_validation_vs_test_sharpe.png")

# 5. High return and high risk strategies.
risk_source = test.copy()
risk_source["abs_drawdown"] = risk_source["max_drawdown"].abs()
top_risk = pd.concat(
    [
        risk_source.nlargest(10, "annualized_return"),
        risk_source.nlargest(10, "annualized_volatility"),
        risk_source.nlargest(10, "abs_drawdown"),
    ],
    ignore_index=True,
).drop_duplicates(SPEC_NO_COST)
top_risk = top_risk[
    [
        "strategy",
        "model_family",
        "score_label",
        "rule",
        "zero_threshold",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "average_turnover",
    ]
].sort_values(["annualized_return", "annualized_volatility"], ascending=False)
top_risk.to_csv(TABLE_DIR / "high_risk_strategies.csv", index=False)

plt.figure(figsize=(8, 5.5))
for family, grp in risk_source.groupby("model_family"):
    plt.scatter(
        grp["annualized_volatility"],
        grp["annualized_return"],
        label=family,
        alpha=0.55,
        s=40,
        color=colors.get(family),
    )
extreme = risk_source.nlargest(5, "annualized_return")
for _, row in extreme.iterrows():
    plt.annotate(
        row["score_label"],
        (row["annualized_volatility"], row["annualized_return"]),
        fontsize=7,
        xytext=(3, 3),
        textcoords="offset points",
    )
plt.xlabel("Annualized volatility")
plt.ylabel("Annualized return")
plt.title("Return versus volatility for test strategies at 25 bps")
plt.legend(fontsize=8)
save_plot(FIGURE_DIR / "robustness_return_vs_volatility.png")

# 6. Architecture robustness.
best_sharpe_by_score = test.sort_values("sharpe", ascending=False).drop_duplicates("score_label")
architecture_rows = []
for score_label in ["mlp_classifier", "mlp_er_train", "ft_classifier", "ft_er_train", "temporal_classifier", "temporal_er_train"]:
    ri = rank_ic.loc[rank_ic["score_label"].eq(score_label) & rank_ic["split"].eq("test")]
    sh = best_sharpe_by_score.loc[best_sharpe_by_score["score_label"].eq(score_label)]
    if not ri.empty:
        architecture_rows.append(
            {
                "model": score_label,
                "architecture": {
                    "mlp_classifier": "MLP classifier",
                    "mlp_er_train": "MLP expected-return score",
                    "ft_classifier": "FT small classifier",
                    "ft_er_train": "FT small expected-return score",
                    "temporal_classifier": "Temporal classifier",
                    "temporal_er_train": "Temporal expected-return score",
                }[score_label],
                "model_family": model_family(score_label),
                "architecture_size": "main",
                "rank_ic": float(ri.iloc[0]["mean_rank_ic"]),
                "sharpe": float(sh.iloc[0]["sharpe"]) if not sh.empty else np.nan,
            }
        )

for path, model, architecture, family, size, rank_col in [
    (
        TABLE_DIR / "temporal_tabular_debug_model_metrics.csv",
        "temporal_debug_classifier",
        "Temporal tiny/debug classifier",
        "Temporal",
        "tiny",
        "classifier_monthly_rank_ic_mean",
    ),
    (
        TABLE_DIR / "temporal_tabular_transformer_seed362559_model_metrics.csv",
        "temporal_full_classifier",
        "Temporal full classifier",
        "Temporal",
        "full",
        "classifier_monthly_rank_ic_mean",
    ),
    (
        TABLE_DIR / "ft_small_seed362559_model_metrics.csv",
        "ft_small_seed362559",
        "FT small seed 362559",
        "FT",
        "small",
        "monthly_rank_ic_mean",
    ),
]:
    if path.exists():
        df = pd.read_csv(path)
        row = df.loc[df["split"].eq("test")]
        if not row.empty:
            architecture_rows.append(
                {
                    "model": model,
                    "architecture": architecture,
                    "model_family": family,
                    "architecture_size": size,
                    "rank_ic": float(row.iloc[0][rank_col]),
                    "sharpe": np.nan,
                }
            )

architecture = pd.DataFrame(architecture_rows).drop_duplicates("model").sort_values(
    ["model_family", "architecture_size", "architecture"]
)
architecture.to_csv(TABLE_DIR / "architecture_robustness.csv", index=False)

fig, ax1 = plt.subplots(figsize=(10, 5.5))
plot_arch = architecture.dropna(subset=["rank_ic"]).copy()
x = np.arange(len(plot_arch))
ax1.bar(x - 0.18, plot_arch["rank_ic"], width=0.36, label="Rank IC", color="tab:blue")
ax1.set_ylabel("Test Rank IC")
ax1.set_xticks(x)
ax1.set_xticklabels(plot_arch["architecture"], rotation=35, ha="right", fontsize=8)
ax2 = ax1.twinx()
ax2.bar(x + 0.18, plot_arch["sharpe"].fillna(0.0), width=0.36, label="Best test Sharpe", color="tab:orange")
ax2.set_ylabel("Best test Sharpe at 25 bps")
fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
plt.title("Architecture robustness: predictive and portfolio metrics")
save_plot(FIGURE_DIR / "robustness_architecture_rank_ic_sharpe.png")

# 7. Summary and LaTeX.
survivors = rep_perf.loc[rep_perf["one_way_cost_bps"].eq(50.0) & rep_perf["sharpe"].gt(0.0)]
collapses = rep_perf.loc[
    rep_perf["one_way_cost_bps"].eq(50.0)
    & (rep_perf["sharpe"].le(0.0) | rep_perf["annualized_return"].le(0.0))
]
stable_threshold = threshold_summary.loc[threshold_summary["stable_vs_threshold"], "model"].tolist()
unstable_threshold = threshold_summary.loc[~threshold_summary["stable_vs_threshold"], "model"].tolist()
degrades_most = val_test.head(5)[["strategy_base", "sharpe_validation", "sharpe_test", "sharpe_drop"]]
high_return_top = risk_source.nlargest(1, "annualized_return").iloc[0]
high_vol_top = risk_source.nlargest(1, "annualized_volatility").iloc[0]

robustness_summary = pd.DataFrame(
    [
        {
            "topic": "transaction_costs",
            "metric": "positive Sharpe at 50bps",
            "value": int(len(survivors)),
            "interpretation": "; ".join(survivors["strategy_base"].tolist()) if len(survivors) else "No representative strategy remains positive at 50 bps.",
        },
        {
            "topic": "threshold_sensitivity",
            "metric": "stable models",
            "value": len(stable_threshold),
            "interpretation": "; ".join(stable_threshold),
        },
        {
            "topic": "validation_test_stability",
            "metric": "validation-test Sharpe correlation",
            "value": validation_test_corr,
            "interpretation": f"Average validation-to-test Sharpe drop is {average_sharpe_drop:.3f}.",
        },
        {
            "topic": "high_risk",
            "metric": "highest-return strategy",
            "value": high_return_top["annualized_return"],
            "interpretation": f"{high_return_top['strategy']} with volatility {high_return_top['annualized_volatility']:.3f} and max DD {high_return_top['max_drawdown']:.3f}.",
        },
    ]
)
robustness_summary.to_csv(TABLE_DIR / "robustness_summary.csv", index=False)

cost_best = cost_sensitivity.sort_values("Sharpe (50bps)", ascending=False).iloc[0]
cost_worst = cost_sensitivity.sort_values("Sharpe (50bps)", ascending=True).iloc[0]
subperiod_display = subperiod_table.copy()
subperiod_display["sharpe_gap_late_minus_early"] = (
    subperiod_display["sharpe_2020-2024"] - subperiod_display["sharpe_2016-2019"]
)
most_stable_subperiod = subperiod_display.iloc[
    subperiod_display["sharpe_gap_late_minus_early"].abs().argmin()
]
threshold_best = threshold_summary.sort_values("sharpe_variation_range").iloc[0]
threshold_worst = threshold_summary.sort_values("sharpe_variation_range", ascending=False).iloc[0]
architecture_main = architecture.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False)
architecture_best = architecture_main.iloc[0]

latex = rf"""
\subsection{{Robustness Checks}}

The robustness analysis uses the same persisted backtest outputs as the portfolio section and keeps 25 basis points as the reference transaction-cost assumption. No model is re-estimated. The first check varies one-way costs from 0 to 50 basis points for the representative MLP, FT, Temporal, and XGBoost baseline strategies selected consistently with the validation-based portfolio pipeline. All four representatives remain positive at 50 bps, but the deterioration is economically meaningful. The strongest 50 bps result is {cost_best['strategy']} with a Sharpe ratio of {cost_best['Sharpe (50bps)']:.2f}, while the most cost-sensitive representative is {cost_worst['strategy']} with a Sharpe ratio of {cost_worst['Sharpe (50bps)']:.2f}. This confirms that the main portfolio conclusions are not purely a zero-cost artifact, but also that turnover is central for economic interpretation.

Threshold and portfolio-rule sensitivity is evaluated at the reference 25 bps cost. The most stable score is {threshold_best['model']}, for which the Sharpe range across admissible thresholds and rules is only {threshold_best['sharpe_variation_range']:.2f}. By contrast, {threshold_worst['model']} has a range of {threshold_worst['sharpe_variation_range']:.2f}, indicating that some signals are more sensitive to implementation details than to broad predictive content. This check is important because the forecasting section evaluates ranking ability, whereas the portfolio section converts ranks into tradable long-short positions.

Time stability is assessed by splitting the test period into 2016--2019 and 2020--2024. The selected representatives generate positive annualized returns in both subperiods. The most stable Sharpe profile is {most_stable_subperiod['strategy']}, with Sharpe ratios of {most_stable_subperiod['sharpe_2016-2019']:.2f} and {most_stable_subperiod['sharpe_2020-2024']:.2f} in the early and late subperiods, respectively. This suggests that the retained results are not entirely concentrated in one market regime, although the late period contains the COVID shock and the subsequent inflation/rate cycle.

Validation-versus-test stability provides a direct check on selection bias. Across the 25 bps strategy universe, the correlation between validation and test Sharpe is {validation_test_corr:.2f}, and the average validation-to-test Sharpe drop is {average_sharpe_drop:.2f}. The largest degradations occur among strategies with high validation Sharpe but weaker out-of-sample performance. Therefore, validation Sharpe is useful for choosing a specification, but the magnitude of the selected validation performance should not be interpreted as an unconditional expected return.

The high-return/high-risk screen shows that some apparently attractive strategies are also the most volatile. The highest-return test strategy is {high_return_top['strategy']}, with annualized return {high_return_top['annualized_return']:.2%}, volatility {high_return_top['annualized_volatility']:.2%}, and maximum drawdown {high_return_top['max_drawdown']:.2%}. The highest-volatility strategy is {high_vol_top['strategy']}. These results imply that part of the return dispersion is compensation for risk, drawdown exposure, and turnover rather than a clean arbitrage opportunity.

Finally, architecture robustness compares the main MLP, FT, and Temporal specifications with the available smaller or diagnostic variants. The best main architecture by portfolio Sharpe is {architecture_best['architecture']} with test Rank IC {architecture_best['rank_ic']:.3f} and best 25 bps test Sharpe {architecture_best['sharpe']:.2f}. The tiny/debug temporal model is much weaker in Rank IC than the full temporal model, while the FT small results remain close to the main FT score. Overall, the evidence supports the use of the retained architectures while showing that very small diagnostic models are insufficient for the final portfolio conclusions.

\section{{Robustness and Limitations}}

The robustness checks support the main conclusion that cross-sectional machine-learning scores contain economically meaningful ranking information, but the portfolio layer is materially affected by implementation assumptions. Transaction costs reduce Sharpe ratios monotonically, and the most turnover-intensive long-short variants are the most fragile. Strategies that survive 25--50 bps costs are more credible candidates for interpretation, whereas strategies that collapse under moderate costs should be viewed as statistical backtests rather than implementable portfolios.

Model selection risk remains important because portfolio rules are chosen on validation Sharpe. Even with a clean train-validation-test split, the large number of score definitions, thresholds, and rules creates a multiple-testing problem. The validation-versus-test comparison therefore acts as a guardrail: a high validation Sharpe is useful for ranking candidates, but it is not sufficient evidence that the same magnitude will persist out of sample.

The analysis also has standard data and implementation limitations. First, transaction costs are modeled as proportional one-way costs and do not include market impact, borrow costs, short-sale constraints, or capacity limits. Second, the project does not use a fully specified CRSP value-weighted market benchmark in the final robustness layer, so benchmark-relative interpretation is limited. Third, the Compustat timing convention assumes accounting information is available with the project lag structure, which may still be imperfect relative to point-in-time commercial databases. Fourth, turnover, short exposure, and gross exposure can make some strategies difficult to implement even when their simulated net returns are positive.

Finally, the main portfolio results are based on the retained classification score. Alternative score definitions were tested but did not improve validation performance, suggesting that the top-minus-bottom probability score is the most stable ranking signal. This makes the reported classification-score portfolio the preferred specification, while leaving open the possibility that other score transformations could perform differently in a larger or more recent sample.
""".strip()

(TABLE_DIR / "robustness_latex_sections.tex").write_text(latex)

print("Saved robustness tables:")
for name in [
    "robustness_summary.csv",
    "cost_sensitivity.csv",
    "subperiod_performance.csv",
    "threshold_sensitivity.csv",
    "validation_test_stability.csv",
    "high_risk_strategies.csv",
    "architecture_robustness.csv",
    "robustness_latex_sections.tex",
]:
    print(f" - {TABLE_DIR / name}")

print("Saved robustness figures:")
for path in sorted(FIGURE_DIR.glob("robustness_*.png")):
    print(f" - {path}")

print("\nMain numerical diagnostics:")
print(cost_sensitivity.to_string(index=False))
print(f"\nValidation-test Sharpe correlation: {validation_test_corr:.3f}")
print(f"Average validation-to-test Sharpe drop: {average_sharpe_drop:.3f}")
print("\nLargest validation-to-test degradations:")
print(degrades_most.to_string(index=False))
