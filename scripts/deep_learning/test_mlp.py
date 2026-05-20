#!/usr/bin/env python3
"""Evaluate the standalone PyTorch MLP classifier on the test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent))

from mlp_common import (  # noqa: E402
    DEFAULT_FEATURE_GROUPS,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    MLPClassifier,
    choose_device,
    load_feature_list,
    load_panel,
    log,
    monthly_rank_ic,
    predict_proba,
    prediction_frame,
    scores_from_probabilities,
    split_frame,
    summarize_rank_ic,
    torch_load_checkpoint,
    transform_features_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate standalone MLP classifier.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--feature-groups", type=Path, default=DEFAULT_FEATURE_GROUPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUTPUT_DIR / "mlp_model.pt")
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
        raise ValueError("Current feature list does not match the trained MLP checkpoint.")
    panel = load_panel(args.input, feature_list)
    test_df = split_frame(panel, "test")
    if args.debug:
        test_df = test_df.sample(min(len(test_df), 25_000), random_state=362559).sort_values(["mthcaldt", "permno"])

    x_test = transform_features_from_checkpoint(test_df, checkpoint)
    model = MLPClassifier(
        input_dim=int(checkpoint["input_dim"]),
        dropout=float(checkpoint.get("dropout", 0.25)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    class_means = np.asarray(checkpoint["class_means"], dtype="float64")

    probabilities = predict_proba(model, x_test, batch_size, device)
    score = scores_from_probabilities(probabilities, class_means)
    predictions = prediction_frame(test_df, probabilities, score)
    rank_ic = monthly_rank_ic(test_df["mthcaldt"], test_df["target_ret_1m"], score)
    summary = summarize_rank_ic(rank_ic)
    log(f"Test mean monthly Rank IC: {summary['mean_monthly_rank_ic']:.6f}")
    log(f"Test months: {summary['months']}")

    predictions.to_parquet(args.output_dir / "mlp_test_predictions.parquet", index=False)
    rank_ic.to_csv(args.output_dir / "mlp_test_rank_ic.csv", index=False)
    pd.DataFrame([summary]).to_csv(args.output_dir / "mlp_test_summary_metrics.csv", index=False)
    log(f"Wrote test artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
