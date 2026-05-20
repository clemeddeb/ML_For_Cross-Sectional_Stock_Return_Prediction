#!/usr/bin/env python3
"""Train the standalone PyTorch MLP classifier."""

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
    MLPClassifier,
    RANDOM_SEED,
    choose_device,
    class_return_means,
    classifier_score_from_probabilities,
    fit_preprocessor,
    load_feature_list,
    load_panel,
    log,
    monthly_rank_ic,
    predict_proba,
    prediction_frame,
    scores_from_probabilities,
    split_frame,
    summarize_rank_ic,
    transform_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone MLP classifier.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--feature-groups", type=Path, default=DEFAULT_FEATURE_GROUPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def train_model(
    model: MLPClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    validation_df: pd.DataFrame,
    x_validation: np.ndarray,
    class_means: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[MLPClassifier, pd.DataFrame, int, float]:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype("int64")))
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)

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

        probabilities = predict_proba(model, x_validation, args.batch_size, device)
        score_er = scores_from_probabilities(probabilities, class_means)
        score_cls = classifier_score_from_probabilities(probabilities)
        rank_ic_er = monthly_rank_ic(validation_df["mthcaldt"], validation_df["target_ret_1m"], score_er)
        rank_ic_cls = monthly_rank_ic(validation_df["mthcaldt"], validation_df["target_ret_1m"], score_cls)
        validation_ic_er = summarize_rank_ic(rank_ic_er)["mean_monthly_rank_ic"]
        validation_ic_cls = summarize_rank_ic(rank_ic_cls)["mean_monthly_rank_ic"]
        train_loss = total_loss / total_rows if total_rows else np.nan
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_ic_er": validation_ic_er,
                "validation_ic_cls": validation_ic_cls,
                "selected": False,
            }
        )
        log(
            f"epoch={epoch} train_loss={train_loss:.6f} "
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
    y_train = train_df["top_bottom_label"].to_numpy(dtype="int64")
    class_means = class_return_means(train_df)
    log(
        "Train class return means: "
        f"bottom={class_means[0]:.6f}, middle={class_means[1]:.6f}, top={class_means[2]:.6f}"
    )

    device = choose_device()
    model = MLPClassifier(input_dim=len(feature_list), dropout=args.dropout).to(device)
    log(f"Training MLP on {device}...")
    model, history, best_epoch, best_ic_cls = train_model(
        model,
        x_train,
        y_train,
        validation_df,
        x_validation,
        class_means,
        args,
        device,
    )

    probabilities = predict_proba(model, x_validation, args.batch_size, device)
    score_er = scores_from_probabilities(probabilities, class_means)
    score_cls = classifier_score_from_probabilities(probabilities)
    predictions = prediction_frame(validation_df, probabilities, score_er)
    rank_ic_er = monthly_rank_ic(validation_df["mthcaldt"], validation_df["target_ret_1m"], score_er)
    rank_ic_cls = monthly_rank_ic(validation_df["mthcaldt"], validation_df["target_ret_1m"], score_cls)
    summary_er = summarize_rank_ic(rank_ic_er)
    summary_cls = summarize_rank_ic(rank_ic_cls)
    log(f"Best validation IC (Classifier score): {best_ic_cls:.6f} at epoch {best_epoch}")
    log(f"Final validation IC (ER score): {summary_er['mean_monthly_rank_ic']:.6f}")
    log(f"Final validation IC (Classifier score): {summary_cls['mean_monthly_rank_ic']:.6f}")

    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "input_dim": len(feature_list),
        "dropout": args.dropout,
        "feature_list": feature_list,
        "imputer_statistics": imputer.statistics_.astype("float32").tolist(),
        "scaler_mean": scaler.mean_.astype("float32").tolist(),
        "scaler_scale": scaler.scale_.astype("float32").tolist(),
        "class_means": class_means.astype("float32").tolist(),
        "batch_size": args.batch_size,
        "best_epoch": best_epoch,
        "best_validation_ic_cls": best_ic_cls,
        "best_validation_rank_ic": best_ic_cls,
        "validation_summary_er": summary_er,
        "validation_summary_cls": summary_cls,
        "validation_summary": summary_er,
        "training_history": history.to_dict(orient="records"),
    }
    torch.save(checkpoint, args.output_dir / "mlp_model.pt")
    predictions.to_parquet(args.output_dir / "mlp_validation_predictions.parquet", index=False)
    rank_ic_er.to_csv(args.output_dir / "mlp_validation_rank_ic_er.csv", index=False)
    rank_ic_cls.to_csv(args.output_dir / "mlp_validation_rank_ic_cls.csv", index=False)
    rank_ic_er.to_csv(args.output_dir / "mlp_validation_rank_ic.csv", index=False)
    history.to_csv(args.output_dir / "mlp_training_history.csv", index=False)
    log(f"Wrote artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
