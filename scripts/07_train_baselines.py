#!/usr/bin/env python3
"""Train first baseline models for monthly stock-return prediction."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, SGDClassifier, SGDRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 362559

DEFAULT_INPUT = Path("Dataset/Processed/model_panel_full_features_with_splits.parquet")
DEFAULT_FEATURE_GROUPS = Path("outputs/sanity_checks/tables/feature_groups.json")
DEFAULT_PREDICTIONS = Path("outputs/predictions/baseline_predictions.parquet")
DEFAULT_TABLE_DIR = Path("outputs/tables")
DEFAULT_MODEL_DIR = Path("outputs/models/baselines")
DEFAULT_FIGURE_DIR = Path("outputs/figures")

ID_COLUMNS = ["permno", "gvkey", "mthcaldt", "split"]
TARGET_COLUMNS = ["target_ret_1m", "target_quintile", "top_bottom_label"]
SPLITS = ["train", "validation", "test"]
FORBIDDEN_FEATURES = {
    "permno",
    "gvkey",
    "mthcaldt",
    "ticker",
    "siccd",
    "naics",
    "split",
    "target_ret_1m",
    "target_quintile",
    "top_bottom_label",
    "mthret",
    "sprtrn",
}

REGRESSION_PREDICTIONS = [
    "prediction_naive_momentum",
    "prediction_naive_reversal",
    "prediction_ridge",
    "prediction_elastic_net",
    "prediction_gradient_boosting_reg",
    "prediction_logistic_classifier_score",
    "prediction_gb_classifier_score",
]
CLASSIFIER_PROBABILITIES = [
    "prob_logistic_bottom",
    "prob_logistic_middle",
    "prob_logistic_top",
    "prob_gb_bottom",
    "prob_gb_middle",
    "prob_gb_top",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline stock-return prediction models.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--feature-groups", type=Path, default=DEFAULT_FEATURE_GROUPS)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--debug", action="store_true", help="Use a small deterministic sample.")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--skip-boosting", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def load_feature_groups(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as f:
        groups = json.load(f)
    return {key: list(value) for key, value in groups.items() if isinstance(value, list)}


def build_feature_list(groups: dict[str, list[str]], columns: list[str]) -> pd.DataFrame:
    group_order = [
        ("return_feature", groups.get("return_feature_columns", [])),
        ("jkp_feature", groups.get("jkp_feature_columns", [])),
        ("selected_compustat_feature", groups.get("selected_compustat_feature_columns", [])),
        (
            "selected_compustat_missing_indicator",
            groups.get("selected_compustat_missing_indicator_columns", []),
        ),
    ]
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    available = set(columns)
    for group_name, features in group_order:
        for feature in features:
            lower = feature.lower()
            if feature in seen:
                continue
            seen.add(feature)
            if feature not in available:
                raise ValueError(f"Feature listed in feature_groups.json is missing: {feature}")
            if lower in FORBIDDEN_FEATURES or "target" in lower or "future" in lower:
                raise ValueError(f"Forbidden target/date/raw-return feature selected: {feature}")
            rows.append({"feature": feature, "feature_group": group_name})
    if not rows:
        raise ValueError("No predictive features were selected.")
    return pd.DataFrame(rows)


def load_panel(path: Path, feature_list: list[str]) -> pd.DataFrame:
    columns = ID_COLUMNS + TARGET_COLUMNS + feature_list
    log(f"Loading modeling columns from {path}...")
    df = pd.read_parquet(path, columns=columns)
    df["mthcaldt"] = pd.to_datetime(df["mthcaldt"], errors="raise")
    df["split"] = df["split"].astype("string")
    for feature in feature_list:
        df[feature] = pd.to_numeric(df[feature], errors="coerce").astype("float32")
    df["target_ret_1m"] = pd.to_numeric(df["target_ret_1m"], errors="raise").astype("float32")
    return df


def apply_debug_sample(df: pd.DataFrame) -> pd.DataFrame:
    sampled_parts = []
    limits = {"train": 60_000, "validation": 25_000, "test": 25_000}
    date_limits = {
        "train": pd.Timestamp("2008-01-01"),
        "validation": pd.Timestamp("2014-01-01"),
        "test": pd.Timestamp("2022-01-01"),
    }
    for split in SPLITS:
        part = df.loc[df["split"].eq(split) & df["mthcaldt"].ge(date_limits[split])]
        if len(part) > limits[split]:
            part = part.sample(limits[split], random_state=RANDOM_SEED)
        sampled_parts.append(part)
    out = pd.concat(sampled_parts, ignore_index=True).sort_values(["mthcaldt", "permno"])
    log(f"Debug sample rows: {len(out):,}")
    return out.reset_index(drop=True)


def assert_split_integrity(df: pd.DataFrame) -> None:
    if df.duplicated(["permno", "mthcaldt"]).any():
        raise ValueError("Input contains duplicate PERMNO-MthCalDt rows.")
    if df["target_ret_1m"].isna().any():
        raise ValueError("target_ret_1m contains missing values.")
    split_dates = {
        split: df.loc[df["split"].eq(split), "mthcaldt"] for split in SPLITS
    }
    if any(values.empty for values in split_dates.values()):
        raise ValueError("At least one split is empty.")
    if not split_dates["train"].max() < split_dates["validation"].min():
        raise ValueError("Train dates are not strictly before validation dates.")
    if not split_dates["validation"].max() < split_dates["test"].min():
        raise ValueError("Validation dates are not strictly before test dates.")


def maybe_limit_train(train_idx: pd.Index, max_train_rows: int | None) -> pd.Index:
    if max_train_rows is None or len(train_idx) <= max_train_rows:
        return train_idx
    rng = np.random.default_rng(RANDOM_SEED)
    selected = rng.choice(train_idx.to_numpy(), size=max_train_rows, replace=False)
    return pd.Index(np.sort(selected))


def train_linear_preprocessor(
    train_df: pd.DataFrame, feature_list: list[str]
) -> tuple[SimpleImputer, StandardScaler, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = imputer.fit_transform(train_df[feature_list])
    x_train = scaler.fit_transform(x_train)
    return imputer, scaler, x_train


def transform_linear(df: pd.DataFrame, feature_list: list[str], imputer: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(imputer.transform(df[feature_list]))


def rank_ic_for_arrays(months: pd.Series, y_true: pd.Series, score: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "mthcaldt": months.to_numpy(),
            "target_ret_1m": np.asarray(y_true, dtype="float64"),
            "score": np.asarray(score, dtype="float64"),
        }
    ).dropna()
    rows = []
    for month, part in frame.groupby("mthcaldt", sort=True):
        if part["score"].nunique(dropna=True) <= 1 or part["target_ret_1m"].nunique(dropna=True) <= 1:
            ic = np.nan
        else:
            ic = part["score"].corr(part["target_ret_1m"], method="spearman")
        rows.append({"mthcaldt": month, "rank_ic": ic})
    return pd.DataFrame(rows)


def validation_rank_ic(df: pd.DataFrame, score: np.ndarray) -> float:
    ic = rank_ic_for_arrays(df["mthcaldt"], df["target_ret_1m"], score)["rank_ic"].dropna()
    return float(ic.mean()) if len(ic) else -np.inf


def class_probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if isinstance(model, SGDClassifier):
        decision = model.decision_function(x)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        decision = np.asarray(decision, dtype="float64")
        decision = decision - np.nanmax(decision, axis=1, keepdims=True)
        exp_decision = np.exp(np.clip(decision, -50, 50))
        probs = exp_decision / exp_decision.sum(axis=1, keepdims=True)
    else:
        probs = model.predict_proba(x)
    out = np.zeros((len(probs), 3), dtype="float64")
    for class_value in [0, 1, 2]:
        if class_value in classes:
            out[:, class_value] = probs[:, classes.index(class_value)]
    invalid = ~np.isfinite(out).all(axis=1)
    if invalid.any():
        out[invalid, :] = 1.0 / 3.0
    return out


def fit_ridge_grid(x_train: np.ndarray, y_train: np.ndarray, val_df: pd.DataFrame, x_val: np.ndarray) -> tuple[Ridge, dict[str, Any], list[dict[str, Any]]]:
    rows = []
    best_model: Ridge | None = None
    best_params: dict[str, Any] = {}
    best_ic = -np.inf
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        ic = validation_rank_ic(val_df, val_pred)
        rows.append({"model": "ridge", "alpha": alpha, "validation_monthly_rank_ic": ic})
        if ic > best_ic:
            best_ic = ic
            best_model = model
            best_params = {"alpha": alpha}
    assert best_model is not None
    return best_model, best_params, rows


def fit_elastic_net_grid(
    x_train: np.ndarray, y_train: np.ndarray, val_df: pd.DataFrame, x_val: np.ndarray
) -> tuple[SGDRegressor, dict[str, Any], list[dict[str, Any]]]:
    rows = []
    best_model: SGDRegressor | None = None
    best_params: dict[str, Any] = {}
    best_ic = -np.inf
    for alpha in [0.0001, 0.001, 0.01, 0.1]:
        for l1_ratio in [0.1, 0.5, 0.9]:
            model = SGDRegressor(
                loss="squared_error",
                penalty="elasticnet",
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=1000,
                tol=1e-4,
                random_state=RANDOM_SEED,
                early_stopping=False,
            )
            model.fit(x_train, y_train)
            val_pred = model.predict(x_val)
            ic = validation_rank_ic(val_df, val_pred)
            rows.append(
                {
                    "model": "elastic_net",
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "validation_monthly_rank_ic": ic,
                    "estimator": "SGDRegressor_elasticnet",
                }
            )
            if ic > best_ic:
                best_ic = ic
                best_model = model
                best_params = {
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "estimator": "SGDRegressor_elasticnet",
                }
    assert best_model is not None
    return best_model, best_params, rows


def fit_logistic_grid(
    x_train: np.ndarray, y_train: np.ndarray, val_df: pd.DataFrame, x_val: np.ndarray
) -> tuple[SGDClassifier, dict[str, Any], list[dict[str, Any]]]:
    rows = []
    best_model: SGDClassifier | None = None
    best_params: dict[str, Any] = {}
    best_ic = -np.inf
    for c_value in [0.01, 0.1, 1.0, 10.0]:
        for class_weight in [None, "balanced"]:
            alpha = 1.0 / (c_value * len(y_train))
            model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha,
                class_weight=class_weight,
                max_iter=50,
                tol=1e-3,
                n_jobs=-1,
                random_state=RANDOM_SEED,
                early_stopping=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x_train, y_train)
            probs = class_probabilities(model, x_val)
            score = probs[:, 2] - probs[:, 0]
            ic = validation_rank_ic(val_df, score)
            rows.append(
                {
                    "model": "logistic_classifier",
                    "C": c_value,
                    "alpha": alpha,
                    "class_weight": "none" if class_weight is None else "balanced",
                    "validation_monthly_rank_ic": ic,
                    "estimator": "SGDClassifier_log_loss",
                }
            )
            if ic > best_ic:
                best_ic = ic
                best_model = model
                best_params = {
                    "C": c_value,
                    "alpha": alpha,
                    "class_weight": "none" if class_weight is None else "balanced",
                    "estimator": "SGDClassifier_log_loss",
                }
    assert best_model is not None
    return best_model, best_params, rows


def fit_gb_reg_grid(
    x_train: pd.DataFrame, y_train: np.ndarray, val_df: pd.DataFrame, x_val: pd.DataFrame, debug: bool
) -> tuple[HistGradientBoostingRegressor, dict[str, Any], list[dict[str, Any]]]:
    grid = (
        [{"max_iter": 30, "learning_rate": 0.1, "max_leaf_nodes": 31}]
        if debug
        else [
            {"max_iter": 100, "learning_rate": 0.03, "max_leaf_nodes": 31},
            {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31},
            {"max_iter": 300, "learning_rate": 0.03, "max_leaf_nodes": 31},
            {"max_iter": 100, "learning_rate": 0.03, "max_leaf_nodes": 63},
        ]
    )
    rows = []
    best_model: HistGradientBoostingRegressor | None = None
    best_params: dict[str, Any] = {}
    best_ic = -np.inf
    for params in grid:
        model = HistGradientBoostingRegressor(
            **params,
            l2_regularization=0.0,
            random_state=RANDOM_SEED,
        )
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        ic = validation_rank_ic(val_df, val_pred)
        row = {"model": "gradient_boosting_reg", "validation_monthly_rank_ic": ic}
        row.update(params)
        rows.append(row)
        if ic > best_ic:
            best_ic = ic
            best_model = model
            best_params = params.copy()
    assert best_model is not None
    return best_model, best_params, rows


def fit_gb_clf_grid(
    x_train: pd.DataFrame, y_train: np.ndarray, val_df: pd.DataFrame, x_val: pd.DataFrame, debug: bool
) -> tuple[HistGradientBoostingClassifier, dict[str, Any], list[dict[str, Any]]]:
    grid = (
        [{"max_iter": 30, "learning_rate": 0.1, "max_leaf_nodes": 31}]
        if debug
        else [
            {"max_iter": 100, "learning_rate": 0.03, "max_leaf_nodes": 31},
            {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31},
            {"max_iter": 300, "learning_rate": 0.03, "max_leaf_nodes": 31},
            {"max_iter": 100, "learning_rate": 0.03, "max_leaf_nodes": 63},
        ]
    )
    rows = []
    best_model: HistGradientBoostingClassifier | None = None
    best_params: dict[str, Any] = {}
    best_ic = -np.inf
    for params in grid:
        model = HistGradientBoostingClassifier(**params, random_state=RANDOM_SEED)
        model.fit(x_train, y_train)
        probs = class_probabilities(model, x_val)
        score = probs[:, 2] - probs[:, 0]
        ic = validation_rank_ic(val_df, score)
        row = {"model": "gb_classifier", "validation_monthly_rank_ic": ic}
        row.update(params)
        rows.append(row)
        if ic > best_ic:
            best_ic = ic
            best_model = model
            best_params = params.copy()
    assert best_model is not None
    return best_model, best_params, rows


def add_naive_predictions(predictions: pd.DataFrame, panel: pd.DataFrame) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if "momentum_12m_excl_1m" not in panel.columns:
        raise ValueError("momentum_12m_excl_1m is required for the naive momentum baseline.")
    if "ret_lag_1m" not in panel.columns:
        raise ValueError("ret_lag_1m is required for the naive reversal baseline.")
    train = panel["split"].eq("train")
    momentum_fill = float(panel.loc[train, "momentum_12m_excl_1m"].median())
    reversal_fill = float((-panel.loc[train, "ret_lag_1m"]).median())
    predictions["prediction_naive_momentum"] = panel["momentum_12m_excl_1m"].fillna(momentum_fill)
    predictions["prediction_naive_reversal"] = (-panel["ret_lag_1m"]).fillna(reversal_fill)
    params["naive_momentum"] = {"feature": "momentum_12m_excl_1m", "missing_fill": momentum_fill}
    params["naive_reversal"] = {"feature": "negative ret_lag_1m", "missing_fill": reversal_fill}
    return params


def regression_metrics(predictions: pd.DataFrame, model_columns: dict[str, str]) -> pd.DataFrame:
    rows = []
    for model, column in model_columns.items():
        if column not in predictions.columns or predictions[column].isna().all():
            continue
        for split in SPLITS:
            part = predictions.loc[predictions["split"].eq(split), ["target_ret_1m", column]].dropna()
            y = part["target_ret_1m"]
            pred = part[column]
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "rows": int(len(part)),
                    "mse": mean_squared_error(y, pred),
                    "mae": mean_absolute_error(y, pred),
                    "pearson_corr": y.corr(pred, method="pearson"),
                    "spearman_corr": y.corr(pred, method="spearman"),
                    "directional_accuracy": float((np.sign(y) == np.sign(pred)).mean()),
                }
            )
    return pd.DataFrame(rows)


def monthly_rank_ic_table(predictions: pd.DataFrame, model_columns: dict[str, str]) -> pd.DataFrame:
    rows = []
    for model, column in model_columns.items():
        if column not in predictions.columns or predictions[column].isna().all():
            continue
        for split in SPLITS:
            part = predictions.loc[predictions["split"].eq(split)]
            ic = rank_ic_for_arrays(part["MthCalDt"], part["target_ret_1m"], part[column])
            for row in ic.to_dict(orient="records"):
                rows.append(
                    {
                        "model": model,
                        "split": split,
                        "month": pd.Timestamp(row["mthcaldt"]).date().isoformat(),
                        "rank_ic": row["rank_ic"],
                    }
                )
    return pd.DataFrame(rows)


def add_rank_ic_summaries(metrics: pd.DataFrame, monthly_ic: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    summary = (
        monthly_ic.groupby(["model", "split"])["rank_ic"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "monthly_rank_ic_mean",
                "std": "monthly_rank_ic_std",
                "count": "monthly_rank_ic_months",
            }
        )
    )
    summary["monthly_rank_ic_tstat"] = summary.apply(
        lambda row: row["monthly_rank_ic_mean"] / (row["monthly_rank_ic_std"] / math.sqrt(row["monthly_rank_ic_months"]))
        if row["monthly_rank_ic_std"] and row["monthly_rank_ic_months"] > 1
        else np.nan,
        axis=1,
    )
    return metrics.merge(summary, on=["model", "split"], how="left")


def classifier_metrics_and_confusion(predictions: pd.DataFrame, classifiers: dict[str, dict[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    confusion_rows = []
    for model, cols in classifiers.items():
        prob_cols = [cols["bottom"], cols["middle"], cols["top"]]
        if any(col not in predictions.columns for col in prob_cols) or predictions[prob_cols].isna().all().all():
            continue
        for split in SPLITS:
            part = predictions.loc[predictions["split"].eq(split)].copy()
            probs = part[prob_cols].to_numpy()
            pred_label = np.argmax(probs, axis=1)
            y = part["top_bottom_label"].astype(int).to_numpy()
            metric_rows.append(
                {
                    "model": model,
                    "split": split,
                    "rows": int(len(part)),
                    "accuracy": accuracy_score(y, pred_label),
                    "balanced_accuracy": balanced_accuracy_score(y, pred_label),
                    "macro_f1": f1_score(y, pred_label, average="macro", zero_division=0),
                    "precision_bottom": precision_score(y, pred_label, labels=[0], average="macro", zero_division=0),
                    "precision_top": precision_score(y, pred_label, labels=[2], average="macro", zero_division=0),
                    "recall_bottom": recall_score(y, pred_label, labels=[0], average="macro", zero_division=0),
                    "recall_top": recall_score(y, pred_label, labels=[2], average="macro", zero_division=0),
                }
            )
            matrix = pd.crosstab(
                pd.Series(y, name="actual_label"),
                pd.Series(pred_label, name="predicted_label"),
                dropna=False,
            ).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
            for actual in [0, 1, 2]:
                for predicted in [0, 1, 2]:
                    confusion_rows.append(
                        {
                            "model": model,
                            "split": split,
                            "actual_label": actual,
                            "predicted_label": predicted,
                            "count": int(matrix.loc[actual, predicted]),
                        }
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(confusion_rows)


def model_selection_summary(
    regression: pd.DataFrame, classifier: pd.DataFrame, hyperparams: pd.DataFrame
) -> pd.DataFrame:
    reg_rows = regression.loc[
        regression["split"].isin(["validation", "test"]),
        ["model", "split", "monthly_rank_ic_mean"],
    ]
    clf_rows = classifier.loc[
        classifier["split"].isin(["validation", "test"]),
        ["model", "split", "monthly_rank_ic_mean"],
    ]
    combined = pd.concat([reg_rows, clf_rows], ignore_index=True)
    pivot = combined.pivot_table(index="model", columns="split", values="monthly_rank_ic_mean", aggfunc="first")
    pivot = pivot.reset_index().rename(
        columns={
            "validation": "validation_monthly_rank_ic",
            "test": "test_monthly_rank_ic",
        }
    )
    selected = hyperparams.groupby("model")["selected"].max().reset_index()
    out = pivot.merge(selected, on="model", how="left")
    out["selected"] = out["selected"].fillna(False)
    out = out.sort_values("validation_monthly_rank_ic", ascending=False).reset_index(drop=True)
    out["best_by_validation_rank_ic"] = False
    if not out.empty:
        out.loc[0, "best_by_validation_rank_ic"] = True
    return out


def selected_hyperparameters_table(selected_params: dict[str, dict[str, Any]], grid_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model, params in selected_params.items():
        row = {"model": model, "selected": True}
        row.update(params)
        rows.append(row)
    for row in grid_rows:
        candidate = row.copy()
        candidate["selected"] = False
        rows.append(candidate)
    return pd.DataFrame(rows)


def plot_rank_ic_bar(summary: pd.DataFrame, split: str, path: Path) -> None:
    column = f"{split}_monthly_rank_ic"
    data = summary.dropna(subset=[column]).sort_values(column)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(data["model"], data[column], color="#2f6f8f")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{split.title()} Monthly Rank IC by Model")
    ax.set_xlabel("Mean monthly rank IC")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_monthly_rank_ic(monthly_ic: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    data = monthly_ic.copy()
    data["month"] = pd.to_datetime(data["month"])
    for model, part in data.groupby("model"):
        smoothed = part.sort_values("month").set_index("month")["rank_ic"].rolling(12, min_periods=3).mean()
        ax.plot(smoothed.index, smoothed.values, linewidth=1.2, label=model)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Monthly Rank IC Over Time (12-month rolling mean)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Rank IC")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_prediction_distribution(predictions: pd.DataFrame, model_columns: dict[str, str], path: Path) -> None:
    sample = predictions.sample(min(len(predictions), 200_000), random_state=RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, column in model_columns.items():
        if column in sample.columns and not sample[column].isna().all():
            values = sample[column].dropna()
            lo, hi = values.quantile([0.01, 0.99])
            ax.hist(values.clip(lo, hi), bins=70, histtype="step", density=True, linewidth=1.3, label=model)
    ax.set_title("Prediction Distribution by Model")
    ax.set_xlabel("Prediction or ranking score, clipped at model 1st/99th percentiles")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(confusion: pd.DataFrame, split: str, path: Path) -> None:
    models = list(confusion.loc[confusion["split"].eq(split), "model"].drop_duplicates())
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models):
        part = confusion.loc[confusion["split"].eq(split) & confusion["model"].eq(model)]
        matrix = part.pivot(index="actual_label", columns="predicted_label", values="count").reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
        image = ax.imshow(matrix.to_numpy(), cmap="Blues")
        ax.set_title(model)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{int(matrix.iloc[i, j]):,}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{split.title()} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def assert_prediction_output(predictions: pd.DataFrame, skipped_columns: set[str]) -> list[str]:
    warnings_out = []
    if predictions.duplicated(["PERMNO", "MthCalDt"]).any():
        raise ValueError("Prediction output has duplicate PERMNO-MthCalDt rows.")
    for column in REGRESSION_PREDICTIONS + CLASSIFIER_PROBABILITIES:
        if column in skipped_columns:
            warnings_out.append(f"{column} is missing because its model was skipped.")
            continue
        missing = predictions.loc[predictions["split"].isin(["validation", "test"]), column].isna().sum()
        if missing:
            raise ValueError(f"{column} has {missing:,} missing validation/test predictions.")
    return warnings_out


def clear_skipped_boosting_artifacts(model_dir: Path) -> None:
    for filename in ["gradient_boosting_regressor.joblib", "gradient_boosting_classifier.joblib"]:
        path = model_dir / filename
        if path.exists():
            path.unlink()
    (model_dir / "gradient_boosting_skipped.json").write_text(
        json.dumps({"skipped": True, "reason": "--skip-boosting"}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    np.random.seed(RANDOM_SEED)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    groups = load_feature_groups(args.feature_groups)
    all_columns = pq.ParquetFile(args.input).schema_arrow.names
    feature_table = build_feature_list(groups, all_columns)
    feature_list = feature_table["feature"].tolist()
    log(f"Predictive features selected: {len(feature_list):,}")

    panel = load_panel(args.input, feature_list)
    if args.debug:
        panel = apply_debug_sample(panel)
    assert_split_integrity(panel)

    train_idx = panel.index[panel["split"].eq("train")]
    fit_train_idx = maybe_limit_train(train_idx, args.max_train_rows)
    if len(fit_train_idx) < len(train_idx):
        log(f"Using {len(fit_train_idx):,} sampled train rows for fitting out of {len(train_idx):,}.")
    train_df = panel.loc[fit_train_idx]
    validation_df = panel.loc[panel["split"].eq("validation")]
    y_train_reg = train_df["target_ret_1m"].to_numpy(dtype="float64")
    y_train_clf = train_df["top_bottom_label"].astype(int).to_numpy()

    predictions = panel[ID_COLUMNS + TARGET_COLUMNS].copy()
    selected_params: dict[str, dict[str, Any]] = {}
    grid_rows: list[dict[str, Any]] = []
    skipped_columns: set[str] = set()
    warnings_out: list[str] = []

    log("Creating naive momentum and reversal scores...")
    selected_params.update(add_naive_predictions(predictions, panel))

    log("Fitting train-only median imputer and standard scaler for linear models...")
    linear_imputer, linear_scaler, x_train_linear = train_linear_preprocessor(train_df, feature_list)
    x_val_linear = transform_linear(validation_df, feature_list, linear_imputer, linear_scaler)

    log("Selecting Ridge alpha by validation monthly rank IC...")
    ridge, ridge_params, rows = fit_ridge_grid(x_train_linear, y_train_reg, validation_df, x_val_linear)
    selected_params["ridge"] = ridge_params
    grid_rows.extend(rows)

    log("Selecting Elastic Net hyperparameters by validation monthly rank IC...")
    elastic_net, elastic_params, rows = fit_elastic_net_grid(x_train_linear, y_train_reg, validation_df, x_val_linear)
    selected_params["elastic_net"] = elastic_params
    grid_rows.extend(rows)

    log("Selecting logistic classifier regularization by validation monthly rank IC...")
    logistic, logistic_params, rows = fit_logistic_grid(x_train_linear, y_train_clf, validation_df, x_val_linear)
    selected_params["logistic_classifier"] = logistic_params
    grid_rows.extend(rows)

    gb_reg = None
    gb_clf = None
    if args.skip_boosting:
        log("Skipping gradient boosting models by request.")
        clear_skipped_boosting_artifacts(args.model_dir)
        warnings_out.append("Gradient boosting models skipped via --skip-boosting.")
        skipped_columns.update(
            {
                "prediction_gradient_boosting_reg",
                "prediction_gb_classifier_score",
                "prob_gb_bottom",
                "prob_gb_middle",
                "prob_gb_top",
            }
        )
    else:
        log("Selecting histogram gradient boosting regressor by validation monthly rank IC...")
        gb_reg, gb_reg_params, rows = fit_gb_reg_grid(
            train_df[feature_list], y_train_reg, validation_df, validation_df[feature_list], args.debug
        )
        selected_params["gradient_boosting_reg"] = gb_reg_params
        grid_rows.extend(rows)

        log("Selecting histogram gradient boosting classifier by validation monthly rank IC...")
        gb_clf, gb_clf_params, rows = fit_gb_clf_grid(
            train_df[feature_list], y_train_clf, validation_df, validation_df[feature_list], args.debug
        )
        selected_params["gb_classifier"] = gb_clf_params
        grid_rows.extend(rows)

    log("Generating predictions for all splits...")
    x_all_linear = transform_linear(panel, feature_list, linear_imputer, linear_scaler)
    predictions["prediction_ridge"] = ridge.predict(x_all_linear)
    predictions["prediction_elastic_net"] = elastic_net.predict(x_all_linear)
    logistic_probs = class_probabilities(logistic, x_all_linear)
    predictions["prob_logistic_bottom"] = logistic_probs[:, 0]
    predictions["prob_logistic_middle"] = logistic_probs[:, 1]
    predictions["prob_logistic_top"] = logistic_probs[:, 2]
    predictions["prediction_logistic_classifier_score"] = logistic_probs[:, 2] - logistic_probs[:, 0]

    if gb_reg is not None:
        predictions["prediction_gradient_boosting_reg"] = gb_reg.predict(panel[feature_list])
    else:
        predictions["prediction_gradient_boosting_reg"] = np.nan
    if gb_clf is not None:
        gb_probs = class_probabilities(gb_clf, panel[feature_list])
        predictions["prob_gb_bottom"] = gb_probs[:, 0]
        predictions["prob_gb_middle"] = gb_probs[:, 1]
        predictions["prob_gb_top"] = gb_probs[:, 2]
        predictions["prediction_gb_classifier_score"] = gb_probs[:, 2] - gb_probs[:, 0]
    else:
        predictions["prob_gb_bottom"] = np.nan
        predictions["prob_gb_middle"] = np.nan
        predictions["prob_gb_top"] = np.nan
        predictions["prediction_gb_classifier_score"] = np.nan

    predictions = predictions.rename(columns={"permno": "PERMNO", "mthcaldt": "MthCalDt"})
    ordered_prediction_columns = [
        "PERMNO",
        "gvkey",
        "MthCalDt",
        "split",
        "target_ret_1m",
        "target_quintile",
        "top_bottom_label",
    ] + REGRESSION_PREDICTIONS + CLASSIFIER_PROBABILITIES
    predictions = predictions[ordered_prediction_columns]
    warnings_out.extend(assert_prediction_output(predictions, skipped_columns))

    log("Computing metrics...")
    model_columns = {
        "naive_momentum": "prediction_naive_momentum",
        "naive_reversal": "prediction_naive_reversal",
        "ridge": "prediction_ridge",
        "elastic_net": "prediction_elastic_net",
        "gradient_boosting_reg": "prediction_gradient_boosting_reg",
        "logistic_classifier": "prediction_logistic_classifier_score",
        "gb_classifier": "prediction_gb_classifier_score",
    }
    classifier_columns = {
        "logistic_classifier": {
            "bottom": "prob_logistic_bottom",
            "middle": "prob_logistic_middle",
            "top": "prob_logistic_top",
        },
        "gb_classifier": {
            "bottom": "prob_gb_bottom",
            "middle": "prob_gb_middle",
            "top": "prob_gb_top",
        },
    }
    monthly_ic = monthly_rank_ic_table(predictions, model_columns)
    regression = add_rank_ic_summaries(regression_metrics(predictions, model_columns), monthly_ic)
    classifier, confusion = classifier_metrics_and_confusion(predictions, classifier_columns)
    classifier = add_rank_ic_summaries(classifier, monthly_ic)
    hyperparams = selected_hyperparameters_table(selected_params, grid_rows)
    selection = model_selection_summary(regression, classifier, hyperparams)

    log("Writing predictions, metrics, plots, and model artifacts...")
    predictions.to_parquet(args.predictions_output, index=False)
    regression.to_csv(args.table_dir / "baseline_regression_metrics.csv", index=False)
    classifier.to_csv(args.table_dir / "baseline_classifier_metrics.csv", index=False)
    selection.to_csv(args.table_dir / "baseline_model_selection_summary.csv", index=False)
    hyperparams.to_csv(args.table_dir / "baseline_selected_hyperparameters.csv", index=False)
    monthly_ic.to_csv(args.table_dir / "baseline_monthly_rank_ic.csv", index=False)
    confusion.to_csv(args.table_dir / "baseline_classifier_confusion_matrices.csv", index=False)
    feature_table.to_csv(args.table_dir / "baseline_feature_list.csv", index=False)

    joblib.dump(ridge, args.model_dir / "ridge_model.joblib")
    joblib.dump(elastic_net, args.model_dir / "elastic_net_model.joblib")
    joblib.dump(logistic, args.model_dir / "logistic_classifier_model.joblib")
    joblib.dump(linear_imputer, args.model_dir / "linear_median_imputer.joblib")
    joblib.dump(linear_scaler, args.model_dir / "linear_standard_scaler.joblib")
    if gb_reg is not None:
        joblib.dump(gb_reg, args.model_dir / "gradient_boosting_regressor.joblib")
    if gb_clf is not None:
        joblib.dump(gb_clf, args.model_dir / "gradient_boosting_classifier.joblib")
    (args.model_dir / "feature_list.json").write_text(json.dumps(feature_list, indent=2) + "\n", encoding="utf-8")
    (args.model_dir / "selected_hyperparameters.json").write_text(
        json.dumps(selected_params, indent=2, default=str) + "\n", encoding="utf-8"
    )

    plot_rank_ic_bar(selection, "validation", args.figure_dir / "validation_rank_ic_by_model.png")
    plot_rank_ic_bar(selection, "test", args.figure_dir / "test_rank_ic_by_model.png")
    plot_monthly_rank_ic(monthly_ic, args.figure_dir / "monthly_rank_ic_over_time.png")
    plot_prediction_distribution(predictions, model_columns, args.figure_dir / "prediction_distribution_by_model.png")
    plot_confusion(confusion, "validation", args.figure_dir / "classifier_confusion_matrix_validation.png")
    plot_confusion(confusion, "test", args.figure_dir / "classifier_confusion_matrix_test.png")

    if warnings_out:
        (args.table_dir / "baseline_warnings.txt").write_text("\n".join(warnings_out) + "\n", encoding="utf-8")
        log("Warnings:")
        for warning in warnings_out:
            log(f"- {warning}")

    best = selection.iloc[0]
    log(f"Best validation monthly rank IC model: {best['model']} ({best['validation_monthly_rank_ic']:.6f})")
    log(f"Wrote predictions: {args.predictions_output}")
    log(f"Wrote metrics: {args.table_dir}")
    log(f"Wrote plots: {args.figure_dir}")
    log(f"Wrote model artifacts: {args.model_dir}")


if __name__ == "__main__":
    main()
