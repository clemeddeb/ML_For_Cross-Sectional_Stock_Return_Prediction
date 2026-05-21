#!/usr/bin/env python3
"""Evaluate the standalone PyTorch MLP return regressor on the test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent))

from mlp_common import (  # noqa: E402
    DEFAULT_FEATURE_GROUPS,
    DEFAULT_INPUT,
    choose_device,
    load_feature_list,
    load_panel,
    log,
    monthly_rank_ic,
    split_frame,
    summarize_rank_ic,
    torch_load_checkpoint,
    transform_features_from_checkpoint,
)
from train_mlp_regressor import (  # noqa: E402
    DEFAULT_REGRESSION_OUTPUT_DIR,
    MLPRegressor,
    percentile_score_by_month,
    predict_returns,
    prediction_frame,
    regression_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate standalone MLP return regressor.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--feature-groups", type=Path, default=DEFAULT_FEATURE_GROUPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REGRESSION_OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_REGRESSION_OUTPUT_DIR / "mlp_regressor.pt")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="Use a small deterministic test subset for a smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    checkpoint = torch_load_checkpoint(args.model, map_location=device)
    batch_size = args.batch_size or int(checkpoint.get("batch_size", 4096))

    log("Loading panel and test split...")
    feature_list = load_feature_list(args.input, args.feature_groups)
    if feature_list != list(checkpoint["feature_list"]):
        raise ValueError("Current feature list does not match the trained MLP regressor checkpoint.")
    panel = load_panel(args.input, feature_list)
    test_df = split_frame(panel, "test")
    if args.debug:
        test_df = test_df.sample(min(len(test_df), 25_000), random_state=362559).sort_values(["mthcaldt", "permno"])

    x_test = transform_features_from_checkpoint(test_df, checkpoint)
    model = MLPRegressor(
        input_dim=int(checkpoint["input_dim"]),
        dropout=float(checkpoint.get("dropout", 0.25)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    pred_return = predict_returns(model, x_test, batch_size, device)
    score_er = pred_return
    score_cls = percentile_score_by_month(test_df["mthcaldt"], pred_return)
    predictions = prediction_frame(test_df, pred_return, score_cls)
    rank_ic_er = monthly_rank_ic(test_df["mthcaldt"], test_df["target_ret_1m"], score_er)
    rank_ic_cls = monthly_rank_ic(test_df["mthcaldt"], test_df["target_ret_1m"], score_cls)
    summary_er = summarize_rank_ic(rank_ic_er)
    summary_cls = summarize_rank_ic(rank_ic_cls)
    metrics = regression_metrics(test_df["target_ret_1m"].to_numpy(dtype="float64"), pred_return)
    summary = {
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "mean_monthly_rank_ic_er": summary_er["mean_monthly_rank_ic"],
        "std_monthly_rank_ic_er": summary_er["std_monthly_rank_ic"],
        "months_er": summary_er["months"],
        "mean_monthly_rank_ic_cls": summary_cls["mean_monthly_rank_ic"],
        "std_monthly_rank_ic_cls": summary_cls["std_monthly_rank_ic"],
        "months_cls": summary_cls["months"],
    }

    log(f"Test MSE={metrics['mse']:.6f} MAE={metrics['mae']:.6f}")
    log(f"Test IC (raw ER score): {summary_er['mean_monthly_rank_ic']:.6f}")
    log(f"Test IC (rank score): {summary_cls['mean_monthly_rank_ic']:.6f}")

    predictions.to_parquet(args.output_dir / "mlp_reg_test_predictions.parquet", index=False)
    rank_ic_er.to_csv(args.output_dir / "mlp_reg_test_rank_ic_er.csv", index=False)
    rank_ic_cls.to_csv(args.output_dir / "mlp_reg_test_rank_ic_cls.csv", index=False)
    pd.DataFrame([summary]).to_csv(args.output_dir / "mlp_reg_test_summary_metrics.csv", index=False)
    log(f"Wrote regression test artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
