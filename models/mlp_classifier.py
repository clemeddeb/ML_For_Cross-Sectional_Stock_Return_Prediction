"""PyTorch multilayer perceptron classifier for tabular stock prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class MLPTrainingConfig:
    input_dim: int
    hidden_dims: tuple[int, int, int] = (128, 64, 32)
    dropout: float = 0.2
    num_classes: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4096
    max_epochs: int = 15
    patience: int = 3
    random_seed: int = 362559


class MLPClassifier(nn.Module):
    """Fixed MLP architecture: 98 -> 128 -> 64 -> 32 -> 3."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int, int] = (128, 64, 32),
        dropout: float = 0.2,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mean_monthly_rank_ic(months: pd.Series, y_true: pd.Series, score: np.ndarray) -> float:
    frame = pd.DataFrame(
        {
            "mthcaldt": months.to_numpy(),
            "target_ret_1m": np.asarray(y_true, dtype="float64"),
            "score": np.asarray(score, dtype="float64"),
        }
    ).dropna()
    monthly_ic = []
    for _, part in frame.groupby("mthcaldt", sort=True):
        if part["score"].nunique(dropna=True) <= 1 or part["target_ret_1m"].nunique(dropna=True) <= 1:
            continue
        monthly_ic.append(part["score"].corr(part["target_ret_1m"], method="spearman"))
    return float(np.mean(monthly_ic)) if monthly_ic else -np.inf


def _finite_float32(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype="float32")
    if not np.isfinite(out).all():
        raise ValueError("MLP input contains non-finite values after preprocessing.")
    return out


def predict_proba(
    model: MLPClassifier,
    x: np.ndarray,
    batch_size: int = 4096,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or choose_device()
    model.eval()
    model.to(device)
    x = _finite_float32(x)
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            probs = torch.softmax(model(batch), dim=1).detach().cpu().numpy()
            probabilities.append(probs)
    return np.vstack(probabilities).astype("float64")


def train_mlp_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    validation_months: pd.Series,
    validation_target_ret: pd.Series,
    config: MLPTrainingConfig,
    device: torch.device | None = None,
) -> tuple[MLPClassifier, dict[str, Any], pd.DataFrame]:
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    device = device or choose_device()

    x_train = _finite_float32(x_train)
    x_validation = _finite_float32(x_validation)
    y_train = np.asarray(y_train, dtype="int64")

    model = MLPClassifier(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        num_classes=config.num_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    generator = torch.Generator()
    generator.manual_seed(config.random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    best_ic = -np.inf
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    stale_epochs = 0
    rows: list[dict[str, Any]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            batch_rows = len(batch_x)
            total_loss += float(loss.detach().cpu()) * batch_rows
            total_rows += batch_rows

        probs = predict_proba(model, x_validation, batch_size=config.batch_size, device=device)
        score = probs[:, 2] - probs[:, 0]
        validation_ic = mean_monthly_rank_ic(validation_months, validation_target_ret, score)
        train_loss = total_loss / total_rows if total_rows else np.nan
        rows.append(
            {
                "model": "mlp_classifier",
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_monthly_rank_ic": validation_ic,
                "selected": False,
            }
        )

        if validation_ic > best_ic:
            best_ic = validation_ic
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    history = pd.DataFrame(rows)
    if not history.empty:
        history.loc[history["epoch"].eq(best_epoch), "selected"] = True

    params = {
        "input_dim": config.input_dim,
        "hidden_dims": str(config.hidden_dims),
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "max_epochs": config.max_epochs,
        "patience": config.patience,
        "best_epoch": best_epoch,
        "validation_monthly_rank_ic": best_ic,
        "device": str(device),
        "estimator": "PyTorch_MLPClassifier",
    }
    return model, params, history
