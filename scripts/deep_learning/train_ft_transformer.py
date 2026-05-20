#!/usr/bin/env python3
"""Train an FT-Transformer classifier for tabular return ranking."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RANDOM_SEED = 362559

DEFAULT_INPUT = Path("Dataset/Processed/model_panel_full_features_with_splits.parquet")
DEFAULT_FEATURE_GROUPS = Path("outputs/sanity_checks/tables/feature_groups.json")
DEFAULT_TABLE_DIR = Path("outputs/tables")
DEFAULT_RUN_NAME = "ft_transformer"

ID_COLUMNS = ["permno", "gvkey", "mthcaldt", "split"]
TARGET_COLUMNS = ["target_ret_1m", "target_quintile", "top_bottom_label"]
SPLITS = ["train", "validation", "test"]
FORBIDDEN_FEATURES = {
    "mthret",
    "sprtrn",
    "target_ret_1m",
    "target_quintile",
    "top_bottom_label",
    "permno",
    "gvkey",
    "mthcaldt",
    "ticker",
    "siccd",
    "naics",
    "split",
    "target_month",
}
FEATURE_GROUP_KEYS = [
    "return_feature_columns",
    "jkp_feature_columns",
    "selected_compustat_feature_columns",
    "selected_compustat_missing_indicator_columns",
]


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an FT-Transformer classifier.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--feature-groups", type=Path, default=DEFAULT_FEATURE_GROUPS)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--model-output", type=Path, default=None)
    parser.add_argument("--log-output", type=Path, default=None)
    parser.add_argument("--d-token", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--ffn-dropout", type=float, default=0.1)
    parser.add_argument("--residual-dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--selection-metric",
        choices=["val_loss", "classifier_rank_ic", "er_train_rank_ic"],
        default="classifier_rank_ic",
        help="Checkpoint selection metric based on validation-only diagnostics.",
    )
    parser.add_argument(
        "--selection-score",
        choices=["er_train", "classifier"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--debug", action="store_true", help="Use a small deterministic sample and short run.")
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> None:
    if args.selection_score is not None:
        args.selection_metric = "er_train_rank_ic" if args.selection_score == "er_train" else "classifier_rank_ic"
    run_name = args.run_name
    args.predictions_output = args.predictions_output or Path(f"outputs/predictions/{run_name}_predictions.parquet")
    args.model_output = args.model_output or Path(f"outputs/models/deep_learning/{run_name}.pt")
    args.log_output = args.log_output or Path(f"outputs/logs/{run_name}_training.log")
    args.model_metrics_output = args.table_dir / f"{run_name}_model_metrics.csv"
    args.monthly_rank_ic_output = args.table_dir / f"{run_name}_monthly_rank_ic.csv"
    args.selected_hyperparameters_output = args.table_dir / f"{run_name}_selected_hyperparameters.csv"
    args.epoch_history_output = args.table_dir / f"{run_name}_epoch_history.csv"


def setup_logging(path: Path) -> tuple[Any, Any, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, handle)
    sys.stderr = Tee(original_stderr, handle)
    log(f"Logging FT-Transformer run to {path}")
    return handle, original_stdout, original_stderr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to inspect and read parquet files. "
            "Install the repo requirements before running this script."
        ) from exc
    return pq.ParquetFile(path).schema_arrow.names


def load_feature_groups(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as f:
        groups = json.load(f)
    return {key: list(value) for key, value in groups.items() if isinstance(value, list)}


def selected_features(groups: dict[str, list[str]], available_columns: list[str]) -> list[str]:
    available = set(available_columns)
    features: list[str] = []
    seen: set[str] = set()
    missing: list[str] = []
    forbidden: list[str] = []
    for group_key in FEATURE_GROUP_KEYS:
        for feature in groups.get(group_key, []):
            lower = feature.lower()
            if feature in seen:
                continue
            seen.add(feature)
            if feature not in available:
                missing.append(feature)
                continue
            if lower in FORBIDDEN_FEATURES:
                forbidden.append(feature)
                continue
            features.append(feature)
    if missing:
        raise ValueError(f"Selected features are missing from the panel: {missing}")
    if forbidden:
        raise ValueError(f"Forbidden columns were selected as features: {forbidden}")
    if not features:
        raise ValueError("No selected predictive features found.")
    return features


def load_panel(path: Path, feature_columns: list[str]) -> pd.DataFrame:
    columns = ID_COLUMNS + TARGET_COLUMNS + feature_columns
    log(f"Loading {len(columns)} columns from {path}...")
    df = pd.read_parquet(path, columns=columns)
    df["mthcaldt"] = pd.to_datetime(df["mthcaldt"], errors="raise")
    df["split"] = df["split"].astype("string")
    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float32")
    df["target_ret_1m"] = pd.to_numeric(df["target_ret_1m"], errors="raise").astype("float32")
    df["top_bottom_label"] = pd.to_numeric(df["top_bottom_label"], errors="raise").astype("int64")
    return df


def assert_panel_integrity(df: pd.DataFrame, feature_columns: list[str]) -> None:
    if df.duplicated(["permno", "mthcaldt"]).any():
        raise ValueError("Input contains duplicate PERMNO-MthCalDt rows.")
    if df["target_ret_1m"].isna().any():
        raise ValueError("target_ret_1m contains missing values.")
    labels = set(df["top_bottom_label"].dropna().astype(int).unique())
    if not labels.issubset({0, 1, 2}):
        raise ValueError(f"Unexpected top_bottom_label values: {sorted(labels)}")
    split_values = set(df["split"].dropna().astype(str).unique())
    if not split_values.issubset(set(SPLITS)):
        raise ValueError(f"Unexpected split values: {sorted(split_values)}")
    split_dates = {split: df.loc[df["split"].eq(split), "mthcaldt"] for split in SPLITS}
    empty_splits = [split for split, dates in split_dates.items() if dates.empty]
    if empty_splits:
        raise ValueError(f"Empty split(s): {empty_splits}")
    if not split_dates["train"].max() < split_dates["validation"].min():
        raise ValueError("Train dates are not strictly before validation dates.")
    if not split_dates["validation"].max() < split_dates["test"].min():
        raise ValueError("Validation dates are not strictly before test dates.")
    lower_features = {feature.lower() for feature in feature_columns}
    forbidden = sorted(lower_features.intersection(FORBIDDEN_FEATURES))
    if forbidden:
        raise ValueError(f"Forbidden columns were selected as features: {forbidden}")


def apply_debug_sample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    limits = {"train": 16_384, "validation": 8_192, "test": 8_192}
    parts: list[pd.DataFrame] = []
    for split in SPLITS:
        part = df.loc[df["split"].eq(split)].copy()
        if len(part) > limits[split]:
            part = part.sample(limits[split], random_state=seed)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True).sort_values(["mthcaldt", "permno"]).reset_index(drop=True)
    log(f"Debug sample rows: {len(out):,}")
    return out


@dataclass
class Preprocessor:
    feature_columns: list[str]
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        missing = np.isnan(x)
        if missing.any():
            x[missing] = np.take(self.median, np.where(missing)[1])
        x -= self.mean
        x /= self.std
        return x.astype(np.float32, copy=False)


@dataclass
class EvaluationResult:
    loss: float
    probabilities: np.ndarray


@dataclass
class TrainingResult:
    best_state: dict[str, torch.Tensor]
    best_epoch: int
    stopped_epoch: int
    early_stopped: bool
    best_validation_loss: float
    best_validation_classifier_rank_ic: float
    best_validation_er_train_rank_ic: float
    best_test_classifier_rank_ic: float
    best_test_er_train_rank_ic: float
    epoch_history: pd.DataFrame


def fit_preprocessor(train_df: pd.DataFrame, feature_columns: list[str]) -> Preprocessor:
    x = train_df[feature_columns].to_numpy(dtype=np.float32, copy=True)
    all_missing = np.isnan(x).all(axis=0)
    if all_missing.any():
        bad = [feature_columns[i] for i in np.flatnonzero(all_missing)]
        raise ValueError(f"Selected feature(s) are entirely missing in train: {bad}")
    median = np.nanmedian(x, axis=0).astype(np.float32)
    missing = np.isnan(x)
    if missing.any():
        x[missing] = np.take(median, np.where(missing)[1])
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    return Preprocessor(feature_columns=feature_columns, median=median, mean=mean, std=std)


def compute_train_class_means(train_df: pd.DataFrame) -> np.ndarray:
    means = []
    for label in [0, 1, 2]:
        values = train_df.loc[train_df["top_bottom_label"].eq(label), "target_ret_1m"]
        if values.empty:
            raise ValueError(f"Training split has no rows for class {label}.")
        means.append(float(values.mean()))
    return np.asarray(means, dtype=np.float32)


class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.bias, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * 4),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_token * 4, d_token),
        )
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.residual_dropout(attn_output)
        ffn_output = self.ffn(self.ffn_norm(x))
        x = x + self.residual_dropout(ffn_output)
        return x


class FTTransformerClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_token: int,
        n_layers: int,
        n_heads: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
    ) -> None:
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError("--d-token must be divisible by --n-heads.")
        self.tokenizer = NumericalFeatureTokenizer(n_features=n_features, d_token=d_token)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_token=d_token,
                    n_heads=n_heads,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_token, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens[:, 0])


def data_loader(x: np.ndarray, y: np.ndarray | None, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    x_tensor = torch.from_numpy(x.astype(np.float32, copy=False))
    if y is None:
        dataset = TensorDataset(x_tensor)
    else:
        dataset = TensorDataset(x_tensor, torch.from_numpy(y.astype(np.int64, copy=True)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> np.ndarray:
    model.eval()
    loader = data_loader(x, y=None, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs: list[np.ndarray] = []
    for (batch_x,) in loader:
        logits = model(batch_x.to(device, non_blocking=True))
        prob = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
        probs.append(prob)
    return np.vstack(probs)


@torch.no_grad()
def evaluate_probabilities_and_loss(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> EvaluationResult:
    model.eval()
    loader = data_loader(x, y=y, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_rows = 0
    probs: list[np.ndarray] = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        logits = model(batch_x)
        total_loss += float(criterion(logits, batch_y).detach().cpu())
        total_rows += len(batch_y)
        probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32))
    return EvaluationResult(loss=total_loss / total_rows, probabilities=np.vstack(probs))


def rank_ic_for_arrays(months: pd.Series, y_true: pd.Series, score: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "mthcaldt": pd.to_datetime(months.to_numpy()),
            "target_ret_1m": np.asarray(y_true, dtype="float64"),
            "score": np.asarray(score, dtype="float64"),
        }
    ).dropna()
    rows: list[dict[str, Any]] = []
    for month, part in frame.groupby("mthcaldt", sort=True):
        if part["score"].nunique(dropna=True) <= 1 or part["target_ret_1m"].nunique(dropna=True) <= 1:
            ic = np.nan
        else:
            ic = part["score"].corr(part["target_ret_1m"], method="spearman")
        rows.append({"mthcaldt": month, "rank_ic": ic})
    return pd.DataFrame(rows)


def monthly_rank_ic_mean(df: pd.DataFrame, score: np.ndarray) -> float:
    ic = rank_ic_for_arrays(df["mthcaldt"], df["target_ret_1m"], score)["rank_ic"].dropna()
    return float(ic.mean()) if len(ic) else -np.inf


def summarize_monthly_ic(values: pd.Series) -> dict[str, float | int]:
    clean = values.dropna()
    count = int(len(clean))
    mean = float(clean.mean()) if count else np.nan
    std = float(clean.std(ddof=1)) if count > 1 else np.nan
    tstat = mean / (std / math.sqrt(count)) if count > 1 and std and np.isfinite(std) else np.nan
    return {
        "monthly_rank_ic_mean": mean,
        "monthly_rank_ic_std": std,
        "monthly_rank_ic_months": count,
        "monthly_rank_ic_tstat": tstat,
    }


def monthly_rank_ic_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    score_columns = ["prediction_ft_er_train_score", "prediction_ft_classifier_score"]
    for split in SPLITS:
        part = predictions.loc[predictions["split"].eq(split)]
        for score_name in score_columns:
            ic = rank_ic_for_arrays(part["MthCalDt"], part["target_ret_1m"], part[score_name].to_numpy())
            for row in ic.to_dict(orient="records"):
                rows.append(
                    {
                        "model": "ft_transformer",
                        "split": split,
                        "score_name": score_name,
                        "month": pd.Timestamp(row["mthcaldt"]).date().isoformat(),
                        "rank_ic": row["rank_ic"],
                    }
                )
    return pd.DataFrame(rows)


def classification_metric_rows(predictions: pd.DataFrame, monthly_ic: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prob_cols = ["prob_ft_bottom", "prob_ft_middle", "prob_ft_top"]
    main_ic = monthly_ic.loc[monthly_ic["score_name"].eq("prediction_ft_er_train_score")]
    for split in SPLITS:
        part = predictions.loc[predictions["split"].eq(split)].copy()
        y_true = part["top_bottom_label"].astype(int).to_numpy()
        y_pred = np.argmax(part[prob_cols].to_numpy(dtype=np.float64), axis=1)
        labels = np.array([0, 1, 2])
        recalls = []
        f1s = []
        precisions = {}
        recall_by_label = {}
        for label in labels:
            tp = int(np.sum((y_true == label) & (y_pred == label)))
            fp = int(np.sum((y_true != label) & (y_pred == label)))
            fn = int(np.sum((y_true == label) & (y_pred != label)))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            recalls.append(recall)
            f1s.append(f1)
            precisions[int(label)] = precision
            recall_by_label[int(label)] = recall
        summary = summarize_monthly_ic(main_ic.loc[main_ic["split"].eq(split), "rank_ic"])
        row = {
            "model": "ft_transformer",
            "split": split,
            "rows": int(len(part)),
            "accuracy": float(np.mean(y_true == y_pred)),
            "balanced_accuracy": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1s)),
            "precision_bottom": precisions[0],
            "precision_top": precisions[2],
            "recall_bottom": recall_by_label[0],
            "recall_top": recall_by_label[2],
        }
        row.update(summary)
        row["er_train_monthly_rank_ic_mean"] = row["monthly_rank_ic_mean"]
        row["er_train_monthly_rank_ic_std"] = row["monthly_rank_ic_std"]
        row["er_train_monthly_rank_ic_months"] = row["monthly_rank_ic_months"]
        row["er_train_monthly_rank_ic_tstat"] = row["monthly_rank_ic_tstat"]
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_frame(panel: pd.DataFrame, probabilities: np.ndarray, mu_train: np.ndarray) -> pd.DataFrame:
    predictions = panel[ID_COLUMNS + TARGET_COLUMNS].copy()
    predictions = predictions.rename(columns={"permno": "PERMNO", "mthcaldt": "MthCalDt"})
    predictions["prob_ft_bottom"] = probabilities[:, 0].astype(np.float32)
    predictions["prob_ft_middle"] = probabilities[:, 1].astype(np.float32)
    predictions["prob_ft_top"] = probabilities[:, 2].astype(np.float32)
    predictions["prediction_ft_classifier_score"] = (
        predictions["prob_ft_top"] - predictions["prob_ft_bottom"]
    ).astype(np.float32)
    predictions["prediction_ft_er_train_score"] = probabilities @ mu_train
    predictions["prediction_ft_er_train_score"] = predictions["prediction_ft_er_train_score"].astype(np.float32)
    return predictions[
        [
            "PERMNO",
            "gvkey",
            "MthCalDt",
            "split",
            "target_ret_1m",
            "target_quintile",
            "top_bottom_label",
            "prob_ft_bottom",
            "prob_ft_middle",
            "prob_ft_top",
            "prediction_ft_classifier_score",
            "prediction_ft_er_train_score",
        ]
    ]


def split_arrays(panel: pd.DataFrame, preprocessor: Preprocessor) -> dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    out: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for split in SPLITS:
        part = panel.loc[panel["split"].eq(split)].copy()
        x = preprocessor.transform(part)
        y = part["top_bottom_label"].to_numpy(dtype=np.int64)
        out[split] = (part, x, y)
        log(f"{split}: rows={len(part):,}, X shape={x.shape}")
    return out


def train_model(
    model: nn.Module,
    arrays: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    mu_train: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> TrainingResult:
    _, x_train, y_train = arrays["train"]
    val_df, x_val, y_val = arrays["validation"]
    test_df, x_test, y_test = arrays["test"]
    loader = data_loader(
        x_train,
        y_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if args.selection_metric == "val_loss":
        best_selection_value = np.inf
    else:
        best_selection_value = -np.inf
    best_validation_loss = np.inf
    best_validation_er_ic = -np.inf
    best_validation_classifier_ic = -np.inf
    best_test_er_ic = -np.inf
    best_test_classifier_ic = -np.inf
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale_epochs = 0
    early_stopped = False
    stopped_epoch = args.max_epochs
    epoch_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(batch_y)
            total_rows += len(batch_y)

        avg_loss = total_loss / total_rows
        val_eval = evaluate_probabilities_and_loss(model, x_val, y_val, args.batch_size, device, args.num_workers)
        test_eval = evaluate_probabilities_and_loss(model, x_test, y_test, args.batch_size, device, args.num_workers)
        val_er_score = val_eval.probabilities @ mu_train
        val_classifier_score = val_eval.probabilities[:, 2] - val_eval.probabilities[:, 0]
        test_er_score = test_eval.probabilities @ mu_train
        test_classifier_score = test_eval.probabilities[:, 2] - test_eval.probabilities[:, 0]
        val_er_ic = monthly_rank_ic_mean(val_df, val_er_score)
        val_classifier_ic = monthly_rank_ic_mean(val_df, val_classifier_score)
        test_er_ic = monthly_rank_ic_mean(test_df, test_er_score)
        test_classifier_ic = monthly_rank_ic_mean(test_df, test_classifier_score)
        if args.selection_metric == "val_loss":
            selection_value = val_eval.loss
            improved = selection_value < best_selection_value
        elif args.selection_metric == "er_train_rank_ic":
            selection_value = val_er_ic
            improved = selection_value > best_selection_value
        else:
            selection_value = val_classifier_ic
            improved = selection_value > best_selection_value
        elapsed_seconds = time.perf_counter() - start_time
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "validation_loss": val_eval.loss,
                "validation_classifier_rank_ic": val_classifier_ic,
                "validation_er_train_rank_ic": val_er_ic,
                "test_classifier_rank_ic": test_classifier_ic,
                "test_er_train_rank_ic": test_er_ic,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": elapsed_seconds,
            }
        )
        log(
            f"Epoch {epoch:03d}: train_loss={avg_loss:.6f}, "
            f"validation_loss={val_eval.loss:.6f}, "
            f"validation_er_rank_ic={val_er_ic:.6f}, "
            f"validation_classifier_rank_ic={val_classifier_ic:.6f}, "
            f"validation_selection_metric={selection_value:.6f}"
        )
        if improved:
            best_selection_value = selection_value
            best_validation_loss = val_eval.loss
            best_validation_er_ic = val_er_ic
            best_validation_classifier_ic = val_classifier_ic
            best_test_er_ic = test_er_ic
            best_test_classifier_ic = test_classifier_ic
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                log(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                early_stopped = True
                stopped_epoch = epoch
                break

    if not early_stopped:
        stopped_epoch = args.max_epochs
    return TrainingResult(
        best_state=best_state,
        best_epoch=best_epoch,
        stopped_epoch=stopped_epoch,
        early_stopped=early_stopped,
        best_validation_loss=best_validation_loss,
        best_validation_classifier_rank_ic=best_validation_classifier_ic,
        best_validation_er_train_rank_ic=best_validation_er_ic,
        best_test_classifier_rank_ic=best_test_classifier_ic,
        best_test_er_train_rank_ic=best_test_er_ic,
        epoch_history=pd.DataFrame(epoch_rows),
    )


def hyperparameter_row(
    args: argparse.Namespace,
    mu_train: np.ndarray,
    result: TrainingResult,
    n_features: int,
) -> pd.DataFrame:
    row = {
        "run_name": args.run_name,
        "seed": args.seed,
        "selection_metric": args.selection_metric,
        "model": "ft_transformer",
        "selected": True,
        "selected_feature_count": n_features,
        "d_token": args.d_token,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "attention_dropout": args.attention_dropout,
        "ffn_dropout": args.ffn_dropout,
        "residual_dropout": args.residual_dropout,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "best_epoch": result.best_epoch,
        "mu_train_0": float(mu_train[0]),
        "mu_train_1": float(mu_train[1]),
        "mu_train_2": float(mu_train[2]),
        "best_validation_loss": result.best_validation_loss,
        "best_validation_classifier_rank_ic": result.best_validation_classifier_rank_ic,
        "best_validation_er_train_rank_ic": result.best_validation_er_train_rank_ic,
        "best_test_classifier_rank_ic": result.best_test_classifier_rank_ic,
        "best_test_er_train_rank_ic": result.best_test_er_train_rank_ic,
        "stopped_epoch": result.stopped_epoch,
        "early_stopped": bool(result.early_stopped),
        "selected_score_name": (
            "validation_loss"
            if args.selection_metric == "val_loss"
            else "prediction_ft_classifier_score"
            if args.selection_metric == "classifier_rank_ic"
            else "prediction_ft_er_train_score"
        ),
        "best_validation_monthly_rank_ic": (
            result.best_validation_classifier_rank_ic
            if args.selection_metric == "classifier_rank_ic"
            else result.best_validation_er_train_rank_ic
            if args.selection_metric == "er_train_rank_ic"
            else np.nan
        ),
        "debug": bool(args.debug),
    }
    return pd.DataFrame([row])


def adjust_args_for_debug(args: argparse.Namespace) -> None:
    if not args.debug:
        return
    args.max_epochs = min(args.max_epochs, 3)
    args.patience = min(args.patience, 2)
    args.batch_size = min(args.batch_size, 2048)
    args.d_token = min(args.d_token, 32)
    args.n_layers = min(args.n_layers, 2)
    args.n_heads = min(args.n_heads, 4)


def save_model(
    path: Path,
    model: nn.Module,
    preprocessor: Preprocessor,
    args: argparse.Namespace,
    mu_train: np.ndarray,
    feature_columns: list[str],
    result: TrainingResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "n_features": len(feature_columns),
            "d_token": args.d_token,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "attention_dropout": args.attention_dropout,
            "ffn_dropout": args.ffn_dropout,
            "residual_dropout": args.residual_dropout,
        },
        "feature_columns": feature_columns,
        "preprocessor": {
            "median": preprocessor.median,
            "mean": preprocessor.mean,
            "std": preprocessor.std,
        },
        "mu_train": mu_train,
        "best_validation_loss": result.best_validation_loss,
        "best_validation_classifier_rank_ic": result.best_validation_classifier_rank_ic,
        "best_validation_er_train_rank_ic": result.best_validation_er_train_rank_ic,
        "best_test_classifier_rank_ic": result.best_test_classifier_rank_ic,
        "best_test_er_train_rank_ic": result.best_test_er_train_rank_ic,
        "best_epoch": result.best_epoch,
        "stopped_epoch": result.stopped_epoch,
        "early_stopped": result.early_stopped,
        "args": vars(args),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    resolve_output_paths(args)
    adjust_args_for_debug(args)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    handle, original_stdout, original_stderr = setup_logging(args.log_output)
    try:
        set_seed(args.seed)
        available_columns = parquet_columns(args.input)
        groups = load_feature_groups(args.feature_groups)
        feature_columns = selected_features(groups, available_columns)
        log(f"Selected feature count: {len(feature_columns)}")

        panel = load_panel(args.input, feature_columns)
        assert_panel_integrity(panel, feature_columns)
        if args.debug:
            panel = apply_debug_sample(panel, args.seed)
            assert_panel_integrity(panel, feature_columns)

        train_df = panel.loc[panel["split"].eq("train")]
        mu_train = compute_train_class_means(train_df)
        log(
            "Training-set class means: "
            f"mu_train_0={mu_train[0]:.8f}, mu_train_1={mu_train[1]:.8f}, mu_train_2={mu_train[2]:.8f}"
        )

        preprocessor = fit_preprocessor(train_df, feature_columns)
        arrays = split_arrays(panel, preprocessor)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Using device: {device}")
        model = FTTransformerClassifier(
            n_features=len(feature_columns),
            d_token=args.d_token,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            attention_dropout=args.attention_dropout,
            ffn_dropout=args.ffn_dropout,
            residual_dropout=args.residual_dropout,
        ).to(device)
        log(
            "Model architecture: "
            f"FTTransformerClassifier(n_features={len(feature_columns)}, d_token={args.d_token}, "
            f"n_layers={args.n_layers}, n_heads={args.n_heads})"
        )

        training_result = train_model(model, arrays, mu_train, args, device)
        model.load_state_dict(training_result.best_state)
        model.to(device)

        probabilities_by_split: list[np.ndarray] = []
        panel_by_split: list[pd.DataFrame] = []
        for split in SPLITS:
            split_df, x_split, _ = arrays[split]
            probabilities_by_split.append(
                predict_probabilities(model, x_split, args.batch_size, device, args.num_workers)
            )
            panel_by_split.append(split_df)
        prediction_panel = pd.concat(panel_by_split, ignore_index=True)
        probabilities = np.vstack(probabilities_by_split)
        predictions = prediction_frame(prediction_panel, probabilities, mu_train)
        if predictions.duplicated(["PERMNO", "MthCalDt"]).any():
            raise ValueError("Prediction output has duplicate PERMNO-MthCalDt rows.")
        if predictions[["prob_ft_bottom", "prob_ft_middle", "prob_ft_top"]].isna().any().any():
            raise ValueError("Prediction output contains missing probabilities.")

        monthly_ic = monthly_rank_ic_table(predictions)
        metrics = classification_metric_rows(predictions, monthly_ic)
        hyperparams = hyperparameter_row(
            args,
            mu_train,
            training_result,
            len(feature_columns),
        )

        predictions.to_parquet(args.predictions_output, index=False)
        metrics.to_csv(args.model_metrics_output, index=False)
        monthly_ic.to_csv(args.monthly_rank_ic_output, index=False)
        hyperparams.to_csv(args.selected_hyperparameters_output, index=False)
        training_result.epoch_history.to_csv(args.epoch_history_output, index=False)
        save_model(args.model_output, model.cpu(), preprocessor, args, mu_train, feature_columns, training_result)

        val_er_ic = metrics.loc[metrics["split"].eq("validation"), "monthly_rank_ic_mean"].iloc[0]
        val_classifier_ic = monthly_ic.loc[
            monthly_ic["split"].eq("validation")
            & monthly_ic["score_name"].eq("prediction_ft_classifier_score"),
            "rank_ic",
        ].dropna().mean()
        log(f"Selection metric: {args.selection_metric}")
        log(f"Best epoch: {training_result.best_epoch}")
        log(f"Stopped epoch: {training_result.stopped_epoch}")
        log(f"Early stopped: {training_result.early_stopped}")
        log(f"Best validation loss at selected checkpoint: {training_result.best_validation_loss:.6f}")
        log(
            "Best validation ER Rank IC at selected checkpoint: "
            f"{training_result.best_validation_er_train_rank_ic:.6f}"
        )
        log(
            "Best validation classifier-score Rank IC at selected checkpoint: "
            f"{training_result.best_validation_classifier_rank_ic:.6f}"
        )
        log(f"Output validation ER Rank IC: {val_er_ic:.6f}")
        log(f"Output validation classifier-score Rank IC: {val_classifier_ic:.6f}")
        log(f"Wrote predictions to {args.predictions_output}")
        log(f"Wrote model to {args.model_output}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        handle.close()


if __name__ == "__main__":
    main()
