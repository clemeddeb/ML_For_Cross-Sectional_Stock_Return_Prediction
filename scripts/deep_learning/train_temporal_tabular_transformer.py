#!/usr/bin/env python3
"""Train a two-branch temporal + tabular Transformer classifier."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


RANDOM_SEED = 362559

DEFAULT_INPUT = Path("Dataset/Processed/model_panel_full_features_with_splits.parquet")
DEFAULT_FEATURE_GROUPS = Path("outputs/sanity_checks/tables/feature_groups.json")
DEFAULT_TABLE_DIR = Path("outputs/tables")
DEFAULT_RUN_NAME = "temporal_tabular_transformer"

ID_COLUMNS = ["permno", "gvkey", "mthcaldt", "split"]
TARGET_COLUMNS = ["target_ret_1m", "target_quintile", "top_bottom_label"]
SPLITS = ["train", "validation", "test"]
FEATURE_GROUP_KEYS = [
    "return_feature_columns",
    "jkp_feature_columns",
    "selected_compustat_feature_columns",
    "selected_compustat_missing_indicator_columns",
]
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
PROB_COLUMNS = [
    "prob_temporal_tabular_bottom",
    "prob_temporal_tabular_middle",
    "prob_temporal_tabular_top",
]
CLASSIFIER_SCORE = "prediction_temporal_tabular_classifier_score"
ER_TRAIN_SCORE = "prediction_temporal_tabular_er_train_score"


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
    parser = argparse.ArgumentParser(description="Train a temporal + tabular Transformer classifier.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--feature-groups", type=Path, default=DEFAULT_FEATURE_GROUPS)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--model-output", type=Path, default=None)
    parser.add_argument("--log-output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--temporal-layers", type=int, default=3)
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--temporal-dropout", type=float, default=0.1)
    parser.add_argument("--tabular-branch", choices=["ft_transformer", "mlp"], default="ft_transformer")
    parser.add_argument("--tabular-d-token", type=int, default=32)
    parser.add_argument("--tabular-layers", type=int, default=2)
    parser.add_argument("--tabular-heads", type=int, default=4)
    parser.add_argument("--tabular-dropout", type=float, default=0.1)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--class-weights", action="store_true", help="Use inverse-frequency train class weights.")
    parser.add_argument("--debug", action="store_true", help="Use a small deterministic sample and short run.")
    parser.add_argument("--init-tabular-from-ft-checkpoint", type=Path, default=None)
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> None:
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
    log(f"Logging temporal-tabular Transformer run to {path}")
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
        raise ImportError("pyarrow is required to inspect and read parquet files.") from exc
    return pq.ParquetFile(path).schema_arrow.names


def load_feature_groups(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as f:
        groups = json.load(f)
    return {key: list(value) for key, value in groups.items() if isinstance(value, list)}


def selected_tabular_features(groups: dict[str, list[str]], available_columns: list[str]) -> list[str]:
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
        raise ValueError(f"Selected tabular features are missing from the panel: {missing}")
    if forbidden:
        raise ValueError(f"Forbidden columns were selected as tabular features: {forbidden}")
    if not features:
        raise ValueError("No selected predictive tabular features found.")
    return features


def selected_sequence_features(available_columns: list[str]) -> list[str]:
    candidates = ["mthret", "MthRet", "sprtrn"]
    features = [column for column in candidates if column in available_columns]
    stock_return = [column for column in features if column.lower() == "mthret"]
    if not stock_return:
        raise ValueError("No raw monthly stock return column found for the temporal branch.")
    out: list[str] = []
    for column in features:
        if column not in out:
            out.append(column)
    if "target_ret_1m" in {column.lower() for column in out}:
        raise ValueError("target_ret_1m cannot be used as a sequence feature.")
    return out


def load_panel(path: Path, tabular_features: list[str], sequence_features: list[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(ID_COLUMNS + TARGET_COLUMNS + tabular_features + sequence_features))
    log(f"Loading {len(columns)} columns from {path}...")
    df = pd.read_parquet(path, columns=columns)
    df["mthcaldt"] = pd.to_datetime(df["mthcaldt"], errors="raise")
    df["split"] = df["split"].astype("string")
    for column in tabular_features + sequence_features:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float32")
    df["target_ret_1m"] = pd.to_numeric(df["target_ret_1m"], errors="raise").astype("float32")
    df["top_bottom_label"] = pd.to_numeric(df["top_bottom_label"], errors="raise").astype("int64")
    return df


def assert_panel_integrity(df: pd.DataFrame, tabular_features: list[str], sequence_features: list[str]) -> None:
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
    forbidden = sorted({feature.lower() for feature in tabular_features}.intersection(FORBIDDEN_FEATURES))
    if forbidden:
        raise ValueError(f"Forbidden columns were selected as tabular features: {forbidden}")
    if "target_ret_1m" in {feature.lower() for feature in sequence_features}:
        raise ValueError("target_ret_1m cannot be used as a sequence feature.")


def apply_debug_sample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    limits = {"train": 8_192, "validation": 4_096, "test": 4_096}
    parts: list[pd.DataFrame] = []
    for split in SPLITS:
        part = df.loc[df["split"].eq(split)].copy()
        if len(part) > limits[split]:
            part = part.sample(limits[split], random_state=seed)
        parts.append(part)
    out = pd.concat(parts, ignore_index=False).sort_values(["mthcaldt", "permno"])
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
class SequencePreprocessor:
    feature_columns: list[str]
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def transform(self, seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = seq.astype(np.float32, copy=True)
        for i in range(out.shape[2]):
            valid = mask & np.isfinite(out[:, :, i])
            out[:, :, i] = np.where(valid, out[:, :, i], self.median[i])
        out -= self.mean.reshape(1, 1, -1)
        out /= self.std.reshape(1, 1, -1)
        out[~mask] = 0.0
        return out.astype(np.float32, copy=False)


@dataclass
class SplitArrays:
    frame: pd.DataFrame
    sequence: np.ndarray
    sequence_mask: np.ndarray
    tabular: np.ndarray
    labels: np.ndarray


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


def fit_sequence_preprocessor(
    train_sequence: np.ndarray,
    train_mask: np.ndarray,
    sequence_features: list[str],
) -> SequencePreprocessor:
    medians = []
    means = []
    stds = []
    for i, feature in enumerate(sequence_features):
        valid = train_mask & np.isfinite(train_sequence[:, :, i])
        values = train_sequence[:, :, i][valid]
        if values.size == 0:
            raise ValueError(f"Sequence feature is entirely missing in train: {feature}")
        median = np.float32(np.median(values))
        filled = np.where(valid, train_sequence[:, :, i], median).astype(np.float32)
        train_valid_values = filled[train_mask]
        mean = np.float32(train_valid_values.mean(dtype=np.float64))
        std = np.float32(train_valid_values.std(dtype=np.float64))
        if not np.isfinite(std) or std < 1e-6:
            std = np.float32(1.0)
        medians.append(median)
        means.append(mean)
        stds.append(std)
    return SequencePreprocessor(
        feature_columns=sequence_features,
        median=np.asarray(medians, dtype=np.float32),
        mean=np.asarray(means, dtype=np.float32),
        std=np.asarray(stds, dtype=np.float32),
    )


def compute_train_class_means(train_df: pd.DataFrame) -> np.ndarray:
    means = []
    for label in [0, 1, 2]:
        values = train_df.loc[train_df["top_bottom_label"].eq(label), "target_ret_1m"]
        if values.empty:
            raise ValueError(f"Training split has no rows for class {label}.")
        means.append(float(values.mean()))
    return np.asarray(means, dtype=np.float32)


def month_ordinal(dates: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(dates)
    return (dt.dt.year.to_numpy(dtype=np.int32) * 12 + dt.dt.month.to_numpy(dtype=np.int32)).astype(np.int32)


def build_raw_sequences(
    df: pd.DataFrame,
    sequence_features: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    if seq_len < 1:
        raise ValueError("--seq-len must be positive.")
    work = df[["permno", "mthcaldt"] + sequence_features].copy()
    work["_row_id"] = np.arange(len(work), dtype=np.int64)
    work["_month_ord"] = month_ordinal(work["mthcaldt"])
    work = work.sort_values(["permno", "mthcaldt"], kind="mergesort")

    n_rows = len(work)
    n_features = len(sequence_features)
    sequences = np.full((n_rows, seq_len, n_features), np.nan, dtype=np.float32)
    mask = np.zeros((n_rows, seq_len), dtype=bool)
    values = work[sequence_features].to_numpy(dtype=np.float32, copy=True)
    row_ids = work["_row_id"].to_numpy(dtype=np.int64)
    month_ord = work["_month_ord"].to_numpy(dtype=np.int32)

    for _, index in work.groupby("permno", sort=False).indices.items():
        idx = np.asarray(index, dtype=np.int64)
        group_rows = row_ids[idx]
        group_months = month_ord[idx]
        group_values = values[idx]
        for pos in range(len(idx)):
            current_month = group_months[pos]
            start = max(0, pos - seq_len + 1)
            hist_months = group_months[start : pos + 1]
            offsets = seq_len - 1 - (current_month - hist_months)
            valid_offsets = (offsets >= 0) & (offsets < seq_len)
            target_row = group_rows[pos]
            valid_offsets_array = offsets[valid_offsets]
            valid_values = group_values[start : pos + 1][valid_offsets]
            sequences[target_row, valid_offsets_array, :] = valid_values
            finite_step = np.isfinite(valid_values).any(axis=1)
            mask[target_row, valid_offsets_array] = finite_step
    if not mask[:, -1].any():
        raise ValueError("No sequence contains current-month information.")
    return sequences, mask


class TemporalTabularDataset(Dataset):
    def __init__(self, arrays: SplitArrays) -> None:
        self.sequence = torch.from_numpy(arrays.sequence.astype(np.float32, copy=False))
        self.sequence_mask = torch.from_numpy(arrays.sequence_mask.astype(bool, copy=False))
        self.tabular = torch.from_numpy(arrays.tabular.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(arrays.labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequence[idx], self.sequence_mask[idx], self.tabular[idx], self.labels[idx]


def data_loader(arrays: SplitArrays, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        TemporalTabularDataset(arrays),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


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


class FTTabularBranch(nn.Module):
    def __init__(self, n_features: int, d_token: int, n_layers: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError("--tabular-d-token must be divisible by --tabular-heads.")
        self.output_dim = d_token
        self.tokenizer = NumericalFeatureTokenizer(n_features=n_features, d_token=d_token)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_token=d_token,
                    n_heads=n_heads,
                    attention_dropout=dropout,
                    ffn_dropout=dropout,
                    residual_dropout=0.0,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens[:, 0])


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MLPTabularBranch(nn.Module):
    def __init__(self, n_features: int, dropout: float) -> None:
        super().__init__()
        self.output_dim = 128
        self.net = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            ResidualMLPBlock(128, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalTransformerBranch(nn.Module):
    def __init__(
        self,
        n_sequence_features: int,
        seq_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("--d-model must be divisible by --temporal-heads.")
        self.output_dim = d_model
        self.value_projection = nn.Linear(n_sequence_features, d_model)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.position = nn.Parameter(torch.empty(1, seq_len + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.value_projection(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.position[:, : tokens.shape[1], :]
        cls_mask = torch.zeros((mask.shape[0], 1), dtype=torch.bool, device=mask.device)
        key_padding_mask = torch.cat([cls_mask, ~mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        return self.norm(encoded[:, 0])


class TemporalTabularTransformerClassifier(nn.Module):
    def __init__(
        self,
        n_sequence_features: int,
        seq_len: int,
        n_tabular_features: int,
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.temporal_branch = TemporalTransformerBranch(
            n_sequence_features=n_sequence_features,
            seq_len=seq_len,
            d_model=args.d_model,
            n_layers=args.temporal_layers,
            n_heads=args.temporal_heads,
            dropout=args.temporal_dropout,
        )
        if args.tabular_branch == "ft_transformer":
            self.tabular_branch = FTTabularBranch(
                n_features=n_tabular_features,
                d_token=args.tabular_d_token,
                n_layers=args.tabular_layers,
                n_heads=args.tabular_heads,
                dropout=args.tabular_dropout,
            )
        else:
            self.tabular_branch = MLPTabularBranch(n_features=n_tabular_features, dropout=args.tabular_dropout)
        seq_dim = self.temporal_branch.output_dim
        tab_dim = self.tabular_branch.output_dim
        fusion_dim = args.fusion_hidden_dim
        self.seq_projection = nn.Linear(seq_dim, fusion_dim)
        self.tab_projection = nn.Linear(tab_dim, fusion_dim)
        self.gate = nn.Sequential(nn.Linear(seq_dim + tab_dim, fusion_dim), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.LayerNorm(seq_dim + tab_dim + fusion_dim),
            nn.Linear(seq_dim + tab_dim + fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(args.fusion_dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(args.fusion_dropout),
            nn.Linear(fusion_dim // 2, 3),
        )

    def forward(self, sequence: torch.Tensor, sequence_mask: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        h_seq = self.temporal_branch(sequence, sequence_mask)
        h_tab = self.tabular_branch(tabular)
        h_both = torch.cat([h_seq, h_tab], dim=1)
        gate = self.gate(h_both)
        h_fused = gate * self.seq_projection(h_seq) + (1.0 - gate) * self.tab_projection(h_tab)
        return self.classifier(torch.cat([h_seq, h_tab, h_fused], dim=1))


def initialize_tabular_from_ft_checkpoint(model: nn.Module, checkpoint_path: Path | None) -> bool:
    if checkpoint_path is None:
        return False
    if not checkpoint_path.exists():
        log(f"Warning: FT checkpoint not found at {checkpoint_path}; continuing with random tabular init.")
        return False
    if not isinstance(model.tabular_branch, FTTabularBranch):
        log("Warning: FT checkpoint warm start is only supported for --tabular-branch ft_transformer.")
        return False
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        log(f"Warning: could not load FT checkpoint {checkpoint_path}: {exc}")
        return False
    source = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    target = model.state_dict()
    updates: dict[str, torch.Tensor] = {}
    for source_key, value in source.items():
        if source_key.startswith("tokenizer."):
            target_key = f"tabular_branch.{source_key}"
        elif source_key.startswith("blocks."):
            target_key = f"tabular_branch.{source_key}"
        else:
            continue
        if target_key in target and target[target_key].shape == value.shape:
            updates[target_key] = value
    if not updates:
        log(f"Warning: no compatible FT tabular weights found in {checkpoint_path}; using random init.")
        return False
    target.update(updates)
    model.load_state_dict(target)
    log(f"Initialized {len(updates)} tabular tensors from FT checkpoint {checkpoint_path}.")
    return True


@torch.no_grad()
def evaluate_probabilities_and_loss(
    model: nn.Module,
    arrays: SplitArrays,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    criterion: nn.Module,
) -> EvaluationResult:
    model.eval()
    loader = data_loader(arrays, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    total_loss = 0.0
    total_rows = 0
    probs: list[np.ndarray] = []
    for sequence, sequence_mask, tabular, labels in loader:
        sequence = sequence.to(device, non_blocking=True)
        sequence_mask = sequence_mask.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(sequence, sequence_mask, tabular)
        total_loss += float(criterion(logits, labels).detach().cpu()) * len(labels)
        total_rows += len(labels)
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
        rows.append({"mthcaldt": month, "rank_ic": ic, "n_stocks": int(len(part))})
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


def monthly_rank_ic_table(predictions: pd.DataFrame, run_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        part = predictions.loc[predictions["split"].eq(split)]
        for score_name in [CLASSIFIER_SCORE, ER_TRAIN_SCORE]:
            ic = rank_ic_for_arrays(part["MthCalDt"], part["target_ret_1m"], part[score_name].to_numpy())
            for row in ic.to_dict(orient="records"):
                rows.append(
                    {
                        "run_name": run_name,
                        "score_name": score_name,
                        "split": split,
                        "month": pd.Timestamp(row["mthcaldt"]).date().isoformat(),
                        "rank_ic": row["rank_ic"],
                        "n_stocks": row["n_stocks"],
                    }
                )
    return pd.DataFrame(rows)


def classification_metric_rows(
    predictions: pd.DataFrame,
    monthly_ic: pd.DataFrame,
    losses: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        part = predictions.loc[predictions["split"].eq(split)].copy()
        y_true = part["top_bottom_label"].astype(int).to_numpy()
        y_pred = np.argmax(part[PROB_COLUMNS].to_numpy(dtype=np.float64), axis=1)
        labels = np.array([0, 1, 2])
        recalls = []
        f1s = []
        precisions: dict[int, float] = {}
        recall_by_label: dict[int, float] = {}
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
        classifier_summary = summarize_monthly_ic(
            monthly_ic.loc[
                monthly_ic["split"].eq(split) & monthly_ic["score_name"].eq(CLASSIFIER_SCORE),
                "rank_ic",
            ]
        )
        er_summary = summarize_monthly_ic(
            monthly_ic.loc[
                monthly_ic["split"].eq(split) & monthly_ic["score_name"].eq(ER_TRAIN_SCORE),
                "rank_ic",
            ]
        )
        row = {
            "model": "temporal_tabular_transformer",
            "split": split,
            "rows": int(len(part)),
            "loss": losses.get(split, np.nan),
            "accuracy": float(np.mean(y_true == y_pred)),
            "balanced_accuracy": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1s)),
            "precision_bottom": precisions[0],
            "recall_bottom": recall_by_label[0],
            "precision_top": precisions[2],
            "recall_top": recall_by_label[2],
            "classifier_monthly_rank_ic_mean": classifier_summary["monthly_rank_ic_mean"],
            "classifier_monthly_rank_ic_std": classifier_summary["monthly_rank_ic_std"],
            "classifier_monthly_rank_ic_months": classifier_summary["monthly_rank_ic_months"],
            "classifier_monthly_rank_ic_tstat": classifier_summary["monthly_rank_ic_tstat"],
            "er_train_monthly_rank_ic_mean": er_summary["monthly_rank_ic_mean"],
            "er_train_monthly_rank_ic_std": er_summary["monthly_rank_ic_std"],
            "er_train_monthly_rank_ic_months": er_summary["monthly_rank_ic_months"],
            "er_train_monthly_rank_ic_tstat": er_summary["monthly_rank_ic_tstat"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_frame(panel: pd.DataFrame, probabilities: np.ndarray, mu_train: np.ndarray) -> pd.DataFrame:
    predictions = panel[ID_COLUMNS + TARGET_COLUMNS].copy()
    predictions = predictions.rename(columns={"permno": "PERMNO", "mthcaldt": "MthCalDt"})
    predictions["prob_temporal_tabular_bottom"] = probabilities[:, 0].astype(np.float32)
    predictions["prob_temporal_tabular_middle"] = probabilities[:, 1].astype(np.float32)
    predictions["prob_temporal_tabular_top"] = probabilities[:, 2].astype(np.float32)
    predictions[CLASSIFIER_SCORE] = (
        predictions["prob_temporal_tabular_top"] - predictions["prob_temporal_tabular_bottom"]
    ).astype(np.float32)
    predictions[ER_TRAIN_SCORE] = (probabilities @ mu_train).astype(np.float32)
    return predictions[
        [
            "PERMNO",
            "gvkey",
            "MthCalDt",
            "split",
            "target_ret_1m",
            "target_quintile",
            "top_bottom_label",
            "prob_temporal_tabular_bottom",
            "prob_temporal_tabular_middle",
            "prob_temporal_tabular_top",
            CLASSIFIER_SCORE,
            ER_TRAIN_SCORE,
        ]
    ]


def split_arrays(
    panel: pd.DataFrame,
    sequences: np.ndarray,
    masks: np.ndarray,
    tabular_preprocessor: Preprocessor,
    sequence_preprocessor: SequencePreprocessor,
) -> dict[str, SplitArrays]:
    out: dict[str, SplitArrays] = {}
    transformed_sequences = sequence_preprocessor.transform(sequences, masks)
    transformed_tabular = tabular_preprocessor.transform(panel)
    for split in SPLITS:
        idx = panel["split"].eq(split).to_numpy()
        split_df = panel.loc[idx].copy()
        arrays = SplitArrays(
            frame=split_df,
            sequence=transformed_sequences[idx],
            sequence_mask=masks[idx],
            tabular=transformed_tabular[idx],
            labels=split_df["top_bottom_label"].to_numpy(dtype=np.int64),
        )
        out[split] = arrays
        valid_steps = arrays.sequence_mask.sum(axis=1)
        log(
            f"{split}: rows={len(split_df):,}, sequence_shape={arrays.sequence.shape}, "
            f"tabular_shape={arrays.tabular.shape}, mean_valid_steps={valid_steps.mean():.2f}"
        )
    return out


def class_weight_tensor(train_labels: np.ndarray, device: torch.device, enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None
    counts = np.bincount(train_labels, minlength=3).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_model(
    model: nn.Module,
    arrays: dict[str, SplitArrays],
    mu_train: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> TrainingResult:
    train_arrays = arrays["train"]
    val_arrays = arrays["validation"]
    test_arrays = arrays["test"]
    loader = data_loader(train_arrays, args.batch_size, shuffle=True, num_workers=args.num_workers)
    weights = class_weight_tensor(train_arrays.labels, device, args.class_weights)
    criterion = nn.CrossEntropyLoss(weight=weights)
    eval_criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_selection_value = -np.inf
    best_validation_loss = np.inf
    best_validation_classifier_ic = -np.inf
    best_validation_er_ic = -np.inf
    best_test_classifier_ic = -np.inf
    best_test_er_ic = -np.inf
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
        for sequence, sequence_mask, tabular, labels in loader:
            sequence = sequence.to(device, non_blocking=True)
            sequence_mask = sequence_mask.to(device, non_blocking=True)
            tabular = tabular.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(sequence, sequence_mask, tabular)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(labels)
            total_rows += len(labels)

        avg_loss = total_loss / total_rows
        val_eval = evaluate_probabilities_and_loss(
            model, val_arrays, args.batch_size, device, args.num_workers, eval_criterion
        )
        test_eval = evaluate_probabilities_and_loss(
            model, test_arrays, args.batch_size, device, args.num_workers, eval_criterion
        )
        val_classifier_score = val_eval.probabilities[:, 2] - val_eval.probabilities[:, 0]
        val_er_score = val_eval.probabilities @ mu_train
        test_classifier_score = test_eval.probabilities[:, 2] - test_eval.probabilities[:, 0]
        test_er_score = test_eval.probabilities @ mu_train
        val_classifier_ic = monthly_rank_ic_mean(val_arrays.frame, val_classifier_score)
        val_er_ic = monthly_rank_ic_mean(val_arrays.frame, val_er_score)
        test_classifier_ic = monthly_rank_ic_mean(test_arrays.frame, test_classifier_score)
        test_er_ic = monthly_rank_ic_mean(test_arrays.frame, test_er_score)
        improved = val_classifier_ic > best_selection_value
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
                "elapsed_seconds": elapsed_seconds,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        log(
            f"Epoch {epoch:03d}: train_loss={avg_loss:.6f}, validation_loss={val_eval.loss:.6f}, "
            f"validation_classifier_rank_ic={val_classifier_ic:.6f}, validation_er_rank_ic={val_er_ic:.6f}, "
            f"test_classifier_rank_ic={test_classifier_ic:.6f}"
        )
        if improved:
            best_selection_value = val_classifier_ic
            best_validation_loss = val_eval.loss
            best_validation_classifier_ic = val_classifier_ic
            best_validation_er_ic = val_er_ic
            best_test_classifier_ic = test_classifier_ic
            best_test_er_ic = test_er_ic
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


@torch.no_grad()
def predict_all(
    model: nn.Module,
    arrays: dict[str, SplitArrays],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
    criterion = nn.CrossEntropyLoss(reduction="mean")
    panels = []
    probabilities = []
    losses = {}
    for split in SPLITS:
        result = evaluate_probabilities_and_loss(
            model, arrays[split], args.batch_size, device, args.num_workers, criterion
        )
        panels.append(arrays[split].frame)
        probabilities.append(result.probabilities)
        losses[split] = result.loss
    return pd.concat(panels, ignore_index=True), np.vstack(probabilities), losses


def hyperparameter_row(
    args: argparse.Namespace,
    mu_train: np.ndarray,
    result: TrainingResult,
    tabular_features: list[str],
    sequence_features: list[str],
    ft_checkpoint_used: bool,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_name": args.run_name,
                "seed": args.seed,
                "selected_feature_count": len(tabular_features),
                "sequence_feature_count": len(sequence_features),
                "sequence_features": "|".join(sequence_features),
                "sequence_length": args.seq_len,
                "d_model": args.d_model,
                "temporal_layers": args.temporal_layers,
                "temporal_heads": args.temporal_heads,
                "temporal_dropout": args.temporal_dropout,
                "tabular_branch": args.tabular_branch,
                "tabular_d_token": args.tabular_d_token,
                "tabular_layers": args.tabular_layers,
                "tabular_heads": args.tabular_heads,
                "tabular_dropout": args.tabular_dropout,
                "fusion_hidden_dim": args.fusion_hidden_dim,
                "fusion_dropout": args.fusion_dropout,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "best_epoch": result.best_epoch,
                "stopped_epoch": result.stopped_epoch,
                "early_stopped": bool(result.early_stopped),
                "best_validation_classifier_rank_ic": result.best_validation_classifier_rank_ic,
                "best_validation_er_train_rank_ic": result.best_validation_er_train_rank_ic,
                "best_validation_loss": result.best_validation_loss,
                "test_classifier_rank_ic_at_best_checkpoint": result.best_test_classifier_rank_ic,
                "test_er_train_rank_ic_at_best_checkpoint": result.best_test_er_train_rank_ic,
                "mu_train_0": float(mu_train[0]),
                "mu_train_1": float(mu_train[1]),
                "mu_train_2": float(mu_train[2]),
                "ft_checkpoint_initialization_used": bool(ft_checkpoint_used),
                "debug": bool(args.debug),
            }
        ]
    )


def save_model(
    path: Path,
    model: nn.Module,
    tabular_preprocessor: Preprocessor,
    sequence_preprocessor: SequencePreprocessor,
    args: argparse.Namespace,
    mu_train: np.ndarray,
    tabular_features: list[str],
    sequence_features: list[str],
    result: TrainingResult,
    ft_checkpoint_used: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "n_tabular_features": len(tabular_features),
            "n_sequence_features": len(sequence_features),
            "seq_len": args.seq_len,
            "d_model": args.d_model,
            "temporal_layers": args.temporal_layers,
            "temporal_heads": args.temporal_heads,
            "temporal_dropout": args.temporal_dropout,
            "tabular_branch": args.tabular_branch,
            "tabular_d_token": args.tabular_d_token,
            "tabular_layers": args.tabular_layers,
            "tabular_heads": args.tabular_heads,
            "tabular_dropout": args.tabular_dropout,
            "fusion_hidden_dim": args.fusion_hidden_dim,
            "fusion_dropout": args.fusion_dropout,
        },
        "tabular_feature_columns": tabular_features,
        "sequence_feature_columns": sequence_features,
        "tabular_preprocessor": {
            "median": tabular_preprocessor.median,
            "mean": tabular_preprocessor.mean,
            "std": tabular_preprocessor.std,
        },
        "sequence_preprocessor": {
            "median": sequence_preprocessor.median,
            "mean": sequence_preprocessor.mean,
            "std": sequence_preprocessor.std,
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
        "ft_checkpoint_initialization_used": ft_checkpoint_used,
        "args": vars(args),
    }
    torch.save(payload, path)


def adjust_args_for_debug(args: argparse.Namespace) -> None:
    if not args.debug:
        return
    args.max_epochs = min(args.max_epochs, 3)
    args.patience = min(args.patience, 2)
    args.batch_size = min(args.batch_size, 1024)
    args.d_model = min(args.d_model, 32)
    args.temporal_layers = min(args.temporal_layers, 1)
    args.temporal_heads = min(args.temporal_heads, 4)
    args.tabular_d_token = min(args.tabular_d_token, 16)
    args.tabular_layers = min(args.tabular_layers, 1)
    args.tabular_heads = min(args.tabular_heads, 4)
    args.fusion_hidden_dim = min(args.fusion_hidden_dim, 128)


def main() -> None:
    args = parse_args()
    resolve_output_paths(args)
    adjust_args_for_debug(args)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    handle, original_stdout, original_stderr = setup_logging(args.log_output)
    try:
        warnings.filterwarnings("once", category=UserWarning)
        set_seed(args.seed)
        available_columns = parquet_columns(args.input)
        groups = load_feature_groups(args.feature_groups)
        tabular_features = selected_tabular_features(groups, available_columns)
        sequence_features = selected_sequence_features(available_columns)
        log(f"Selected tabular feature count: {len(tabular_features)}")
        log(f"Selected sequence features: {sequence_features}")

        panel = load_panel(args.input, tabular_features, sequence_features)
        assert_panel_integrity(panel, tabular_features, sequence_features)
        raw_sequences, raw_masks = build_raw_sequences(panel, sequence_features, args.seq_len)
        log(
            f"Built raw sequences with shape={raw_sequences.shape}; "
            f"mean_valid_steps={raw_masks.sum(axis=1).mean():.2f}"
        )

        if args.debug:
            sampled = apply_debug_sample(panel, args.seed)
            debug_idx = sampled.index.to_numpy()
            panel = sampled.reset_index(drop=True)
            raw_sequences = raw_sequences[debug_idx]
            raw_masks = raw_masks[debug_idx]
            assert_panel_integrity(panel, tabular_features, sequence_features)

        train_df = panel.loc[panel["split"].eq("train")]
        train_mask = panel["split"].eq("train").to_numpy()
        mu_train = compute_train_class_means(train_df)
        log(
            "Training-set class means: "
            f"mu_train_0={mu_train[0]:.8f}, mu_train_1={mu_train[1]:.8f}, mu_train_2={mu_train[2]:.8f}"
        )

        tabular_preprocessor = fit_preprocessor(train_df, tabular_features)
        sequence_preprocessor = fit_sequence_preprocessor(
            raw_sequences[train_mask],
            raw_masks[train_mask],
            sequence_features,
        )
        arrays = split_arrays(panel, raw_sequences, raw_masks, tabular_preprocessor, sequence_preprocessor)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Using device: {device}")
        model = TemporalTabularTransformerClassifier(
            n_sequence_features=len(sequence_features),
            seq_len=args.seq_len,
            n_tabular_features=len(tabular_features),
            args=args,
        )
        ft_checkpoint_used = initialize_tabular_from_ft_checkpoint(model, args.init_tabular_from_ft_checkpoint)
        model.to(device)
        log(
            "Model architecture: "
            f"TemporalTransformer(seq_len={args.seq_len}, n_sequence_features={len(sequence_features)}, "
            f"d_model={args.d_model}, layers={args.temporal_layers}, heads={args.temporal_heads}) + "
            f"{args.tabular_branch}(n_features={len(tabular_features)}) + gated fusion"
        )

        training_result = train_model(model, arrays, mu_train, args, device)
        model.load_state_dict(training_result.best_state)
        model.to(device)
        prediction_panel, probabilities, losses = predict_all(model, arrays, args, device)
        predictions = prediction_frame(prediction_panel, probabilities, mu_train)
        if predictions.duplicated(["PERMNO", "MthCalDt"]).any():
            raise ValueError("Prediction output has duplicate PERMNO-MthCalDt rows.")
        if predictions[PROB_COLUMNS].isna().any().any():
            raise ValueError("Prediction output contains missing probabilities.")

        monthly_ic = monthly_rank_ic_table(predictions, args.run_name)
        metrics = classification_metric_rows(predictions, monthly_ic, losses)
        hyperparams = hyperparameter_row(
            args, mu_train, training_result, tabular_features, sequence_features, ft_checkpoint_used
        )

        predictions.to_parquet(args.predictions_output, index=False)
        metrics.to_csv(args.model_metrics_output, index=False)
        monthly_ic.to_csv(args.monthly_rank_ic_output, index=False)
        hyperparams.to_csv(args.selected_hyperparameters_output, index=False)
        training_result.epoch_history.to_csv(args.epoch_history_output, index=False)
        save_model(
            args.model_output,
            model.cpu(),
            tabular_preprocessor,
            sequence_preprocessor,
            args,
            mu_train,
            tabular_features,
            sequence_features,
            training_result,
            ft_checkpoint_used,
        )

        val_classifier_ic = monthly_ic.loc[
            monthly_ic["split"].eq("validation") & monthly_ic["score_name"].eq(CLASSIFIER_SCORE),
            "rank_ic",
        ].dropna().mean()
        val_er_ic = monthly_ic.loc[
            monthly_ic["split"].eq("validation") & monthly_ic["score_name"].eq(ER_TRAIN_SCORE),
            "rank_ic",
        ].dropna().mean()
        log(f"Best epoch: {training_result.best_epoch}")
        log(f"Stopped epoch: {training_result.stopped_epoch}")
        log(f"Early stopped: {training_result.early_stopped}")
        log(f"Best validation loss at selected checkpoint: {training_result.best_validation_loss:.6f}")
        log(
            "Best validation classifier-score Rank IC at selected checkpoint: "
            f"{training_result.best_validation_classifier_rank_ic:.6f}"
        )
        log(
            "Best validation ER Rank IC at selected checkpoint: "
            f"{training_result.best_validation_er_train_rank_ic:.6f}"
        )
        log(f"Output validation classifier-score Rank IC: {val_classifier_ic:.6f}")
        log(f"Output validation ER Rank IC: {val_er_ic:.6f}")
        log(f"Rows retained after sequence construction: {len(panel):,}")
        log(f"Wrote predictions to {args.predictions_output}")
        log(f"Wrote model to {args.model_output}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        handle.close()


if __name__ == "__main__":
    main()
