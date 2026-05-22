"""Shared helpers for the standalone MLP pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn


RANDOM_SEED = 362559
FULL_FEATURE_COUNT = 98
DEFAULT_INPUT = Path("Dataset/Processed/model_panel_full_features_with_splits.parquet")
DEFAULT_FEATURE_GROUPS = Path("outputs/sanity_checks/tables/feature_groups.json")
DEFAULT_OUTPUT_DIR = Path("outputs/models/deep_learning")
ID_COLUMNS = ["permno", "gvkey", "mthcaldt", "split"]
TARGET_COLUMNS = ["target_ret_1m", "target_quintile", "top_bottom_label"]
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


class MLPClassifier(nn.Module):
    """Fixed tabular architecture: 98 -> 128 -> 64 -> 32 -> 3."""

    def __init__(self, input_dim: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log(message: str) -> None:
    print(message, flush=True)


def load_feature_groups(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as handle:
        groups = json.load(handle)
    return {key: list(value) for key, value in groups.items() if isinstance(value, list)}


def build_feature_list(groups: dict[str, list[str]], columns: list[str]) -> list[str]:
    group_order = [
        groups.get("return_feature_columns", []),
        groups.get("jkp_feature_columns", []),
        groups.get("selected_compustat_feature_columns", []),
        groups.get("selected_compustat_missing_indicator_columns", []),
    ]
    features: list[str] = []
    seen: set[str] = set()
    available = set(columns)
    for group in group_order:
        for feature in group:
            lower = feature.lower()
            if feature in seen:
                continue
            seen.add(feature)
            if feature not in available:
                raise ValueError(f"Feature listed in feature_groups.json is missing: {feature}")
            if lower in FORBIDDEN_FEATURES or "target" in lower or "future" in lower:
                raise ValueError(f"Forbidden target/date/raw-return feature selected: {feature}")
            features.append(feature)
    if not features:
        raise ValueError("No predictive features found in feature_groups.json.")
    return features


def load_feature_list(input_path: Path, feature_groups_path: Path) -> list[str]:
    groups = load_feature_groups(feature_groups_path)
    all_columns = pq.ParquetFile(input_path).schema_arrow.names
    return build_feature_list(groups, all_columns)


def load_panel(input_path: Path, feature_list: list[str]) -> pd.DataFrame:
    columns = ID_COLUMNS + TARGET_COLUMNS + feature_list
    panel = pd.read_parquet(input_path, columns=columns)
    panel["mthcaldt"] = pd.to_datetime(panel["mthcaldt"], errors="raise")
    panel["split"] = panel["split"].astype("string")
    panel["target_ret_1m"] = pd.to_numeric(panel["target_ret_1m"], errors="raise").astype("float32")
    panel["top_bottom_label"] = panel["top_bottom_label"].astype("int64")
    for feature in feature_list:
        panel[feature] = pd.to_numeric(panel[feature], errors="coerce").astype("float32")
    return panel


def split_frame(panel: pd.DataFrame, split: str) -> pd.DataFrame:
    out = panel.loc[panel["split"].eq(split)].copy()
    if out.empty:
        raise ValueError(f"Split is empty: {split}")
    return out


def fit_preprocessor(train_df: pd.DataFrame, feature_list: list[str]) -> tuple[SimpleImputer, StandardScaler, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = imputer.fit_transform(train_df[feature_list])
    x_train = scaler.fit_transform(x_train).astype("float32")
    return imputer, scaler, x_train


def transform_features(
    df: pd.DataFrame,
    feature_list: list[str],
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    return scaler.transform(imputer.transform(df[feature_list])).astype("float32")


def transform_features_from_checkpoint(df: pd.DataFrame, checkpoint: dict[str, Any]) -> np.ndarray:
    feature_list = list(checkpoint["feature_list"])
    x = df[feature_list].to_numpy(dtype="float32", copy=True)
    medians = np.asarray(checkpoint["imputer_statistics"], dtype="float32")
    means = np.asarray(checkpoint["scaler_mean"], dtype="float32")
    scales = np.asarray(checkpoint["scaler_scale"], dtype="float32")
    row_idx, col_idx = np.where(np.isnan(x))
    if len(row_idx):
        x[row_idx, col_idx] = medians[col_idx]
    return ((x - means) / scales).astype("float32")


def class_return_means(train_df: pd.DataFrame) -> np.ndarray:
    means = (
        train_df.groupby("top_bottom_label", observed=True)["target_ret_1m"]
        .mean()
        .reindex([0, 1, 2])
    )
    if means.isna().any():
        raise ValueError("Training set is missing at least one class label.")
    return means.to_numpy(dtype="float32")


def scores_from_probabilities(probabilities: np.ndarray, class_means: np.ndarray) -> np.ndarray:
    return probabilities @ np.asarray(class_means, dtype="float64")


def classifier_score_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return probabilities[:, 2] - probabilities[:, 0]


def monthly_rank_ic(months: pd.Series, realized_returns: pd.Series, score: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "month": pd.to_datetime(months, errors="raise"),
            "target_ret_1m": np.asarray(realized_returns, dtype="float64"),
            "score": np.asarray(score, dtype="float64"),
        }
    ).dropna()
    rows: list[dict[str, Any]] = []
    for month, part in frame.groupby("month", sort=True):
        if part["score"].nunique(dropna=True) <= 1 or part["target_ret_1m"].nunique(dropna=True) <= 1:
            ic = np.nan
        else:
            ic = part["score"].corr(part["target_ret_1m"], method="spearman")
        rows.append({"month": pd.Timestamp(month).date().isoformat(), "rank_ic": ic})
    return pd.DataFrame(rows)


def summarize_rank_ic(rank_ic: pd.DataFrame) -> dict[str, float | int]:
    values = pd.to_numeric(rank_ic["rank_ic"], errors="coerce").dropna()
    return {
        "mean_monthly_rank_ic": float(values.mean()) if len(values) else float("nan"),
        "std_monthly_rank_ic": float(values.std()) if len(values) else float("nan"),
        "months": int(len(values)),
    }


def predict_proba(model: MLPClassifier, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            probs = torch.softmax(model(batch), dim=1).detach().cpu().numpy()
            probabilities.append(probs)
    return np.vstack(probabilities).astype("float64")


def prediction_frame(df: pd.DataFrame, probabilities: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    out = df[ID_COLUMNS + ["target_ret_1m", "top_bottom_label"]].copy()
    out["prob_mlp_bottom"] = probabilities[:, 0]
    out["prob_mlp_middle"] = probabilities[:, 1]
    out["prob_mlp_top"] = probabilities[:, 2]
    out["prediction_mlp_score"] = score
    return out.rename(columns={"permno": "PERMNO", "mthcaldt": "MthCalDt"})


def torch_load_checkpoint(path: Path, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
