#!/usr/bin/env python3
"""Train a standalone PyTorch MLP regressor for next-month returns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent))

from mlp_common import (  # noqa: E402
    DEFAULT_FEATURE_GROUPS,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    ID_COLUMNS,
    RANDOM_SEED,
    choose_device,
    fit_preprocessor,
    load_feature_list,
    load_panel,
    log,
    monthly_rank_ic,
    split_frame,
    summarize_rank_ic,
    transform_features,
)


DEFAULT_REGRESSION_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "regression"


class MLPRegressor(nn.Module):
    """Fixed tabular architecture: 98 -> 128 -> 64 -> 32 -> 1."""

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
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone MLP return regressor.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--feature-groups", type=Path, default=DEFAULT_FEATURE_GROUPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REGRESSION_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--debug", action="store_true", help="Use a small deterministic subset for a smoke test.")
    return parser.parse_args()


def debug_sample(train_df: pd.DataFrame, validation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.sample(min(len(train_df), 60_000), random_state=RANDOM_SEED)
    validation = validation_df.sample(min(len(validation_df), 25_000), random_state=RANDOM_SEED)
    return train.sort_values(["mthcaldt", "permno"]), validation.sort_values(["mthcaldt", "permno"])


def percentile_score_by_month(months: pd.Series, predictions: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "month": pd.to_datetime(months, errors="raise"),
            "prediction": np.asarray(predictions, dtype="float64"),
        }
    )
    return (
        frame.groupby("month", sort=False, observed=True)["prediction"]
        .rank(method="average", pct=True)
        .to_numpy(dtype="float64")
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = np.asarray(y_pred, dtype="float64") - np.asarray(y_true, dtype="float64")
    return {
        "mse": float(np.mean(errors**2)),
        "mae": float(np.mean(np.abs(errors))),
    }


def predict_returns(model: MLPRegressor, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            pred = model(batch).detach().cpu().numpy()
            predictions.append(pred)
    return np.concatenate(predictions).astype("float64")


def prediction_frame(df: pd.DataFrame, pred_return: np.ndarray, score_cls: np.ndarray) -> pd.DataFrame:
    out = df[ID_COLUMNS + ["target_ret_1m", "top_bottom_label"]].copy()
    out["prediction_mlp_reg_return"] = pred_return
    out["score_mlp_reg_er"] = pred_return
    out["score_mlp_reg_cls"] = score_cls
    return out.rename(columns={"permno": "PERMNO", "mthcaldt": "MthCalDt"})


def train_model(
    model: MLPRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    validation_df: pd.DataFrame,
    x_validation: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[MLPRegressor, pd.DataFrame, int, float]:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype("float32")))
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)

    y_validation = validation_df["target_ret_1m"].to_numpy(dtype="float64")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_ic_cls = -np.inf
    best_epoch = 0
    stale_epochs = 0
    rows: list[dict[str, float | int | bool]] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_x)
            total_rows += len(batch_x)

        pred_return = predict_returns(model, x_validation, args.batch_size, device)
        score_er = pred_return
        score_cls = percentile_score_by_month(validation_df["mthcaldt"], pred_return)
        rank_ic_er = monthly_rank_ic(validation_df["mthcaldt"], validation_df["target_ret_1m"], score_er)
        rank_ic_cls = monthly_rank_ic(validation_df["mthcaldt"], validation_df["target_ret_1m"], score_cls)
        validation_ic_er = summarize_rank_ic(rank_ic_er)["mean_monthly_rank_ic"]
        validation_ic_cls = summarize_rank_ic(rank_ic_cls)["mean_monthly_rank_ic"]
        metrics = regression_metrics(y_validation, pred_return)
        train_loss = total_loss / total_rows if total_rows else np.nan
        rows.append(
            {
                "epoch": epoch,
                "train_loss_mse": train_loss,
                "validation_mse": metrics["mse"],
                "validation_mae": metrics["mae"],
                "validation_ic_er": validation_ic_er,
                "validation_ic_cls": validation_ic_cls,
                "selected": False,
            }
        )
        log(
            f"epoch={epoch} train_mse={train_loss:.6f} "
            f"val_mse={metrics['mse']:.6f} val_mae={metrics['mae']:.6f} "
            f"val_ic_er={validation_ic_er:.6f} val_ic_cls={validation_ic_cls:.6f}"
        )

        if validation_ic_cls > best_ic_cls:
            best_ic_cls = float(validation_ic_cls)
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    model.load_state_dict(best_state)
    history = pd.DataFrame(rows)
    if not history.empty:
        history.loc[history["epoch"].eq(best_epoch), "selected"] = True
    return model, history, best_epoch, best_ic_cls


def main() -> None:
    args = parse_args()
    if args.debug:
        args.max_epochs = min(args.max_epochs, 2)
        args.batch_size = min(args.batch_size, 1024)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log("Loading feature list and panel...")
    feature_list = load_feature_list(args.input, args.feature_groups)
    panel = load_panel(args.input, feature_list)
    train_df = split_frame(panel, "train")
    validation_df = split_frame(panel, "validation")
    if args.debug:
        train_df, validation_df = debug_sample(train_df, validation_df)

    log(f"Using {len(feature_list)} predictive features.")
    log("Fitting train-only imputer/scaler...")
    imputer, scaler, x_train = fit_preprocessor(train_df, feature_list)
    x_validation = transform_features(validation_df, feature_list, imputer, scaler)
    y_train = train_df["target_ret_1m"].to_numpy(dtype="float32")

    device = choose_device()
    model = MLPRegressor(input_dim=len(feature_list), dropout=args.dropout).to(device)
    log(f"Training MLP regressor on {device}...")
    model, history, best_epoch, best_ic_cls = train_model(
        model,
        x_train,
        y_train,
        validation_df,
        x_validation,
        args,
        device,
    )

    pred_return = predict_returns(model, x_validation, args.batch_size, device)
    score_er = pred_return
    score_cls = percentile_score_by_month(validation_df["mthcaldt"], pred_return)
    rank_ic_er = monthly_rank_ic(validation_df["mthcaldt"], validation_df["target_ret_1m"], score_er)
    rank_ic_cls = monthly_rank_ic(validation_df["mthcaldt"], validation_df["target_ret_1m"], score_cls)
    summary_er = summarize_rank_ic(rank_ic_er)
    summary_cls = summarize_rank_ic(rank_ic_cls)
    metrics = regression_metrics(validation_df["target_ret_1m"].to_numpy(dtype="float64"), pred_return)
    predictions = prediction_frame(validation_df, pred_return, score_cls)

    log(f"Best validation IC (rank score): {best_ic_cls:.6f} at epoch {best_epoch}")
    log(f"Final validation IC (raw ER score): {summary_er['mean_monthly_rank_ic']:.6f}")
    log(f"Final validation IC (rank score): {summary_cls['mean_monthly_rank_ic']:.6f}")
    log(f"Final validation MSE={metrics['mse']:.6f} MAE={metrics['mae']:.6f}")

    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "input_dim": len(feature_list),
        "dropout": args.dropout,
        "feature_list": feature_list,
        "imputer_statistics": imputer.statistics_.astype("float32").tolist(),
        "scaler_mean": scaler.mean_.astype("float32").tolist(),
        "scaler_scale": scaler.scale_.astype("float32").tolist(),
        "batch_size": args.batch_size,
        "best_epoch": best_epoch,
        "best_validation_ic_cls": best_ic_cls,
        "validation_summary_er": summary_er,
        "validation_summary_cls": summary_cls,
        "validation_mse": metrics["mse"],
        "validation_mae": metrics["mae"],
        "training_history": history.to_dict(orient="records"),
    }
    torch.save(checkpoint, args.output_dir / "mlp_regressor.pt")
    predictions.to_parquet(args.output_dir / "mlp_reg_validation_predictions.parquet", index=False)
    rank_ic_er.to_csv(args.output_dir / "mlp_reg_validation_rank_ic_er.csv", index=False)
    rank_ic_cls.to_csv(args.output_dir / "mlp_reg_validation_rank_ic_cls.csv", index=False)
    history.to_csv(args.output_dir / "mlp_reg_training_history.csv", index=False)
    log(f"Wrote regression artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
