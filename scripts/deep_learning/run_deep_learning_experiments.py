#!/usr/bin/env python3
"""Launch and track deep-learning model experiments.

Examples
--------
Dry-run the full grid:

    python scripts/deep_learning/run_deep_learning_experiments.py \
      --dry-run \
      --experiments all

Run all available experiments, skipping completed prediction files:

    python scripts/deep_learning/run_deep_learning_experiments.py \
      --experiments all \
      --skip-existing \
      --continue-on-error

Run a debug smoke test for selected temporal-tabular experiments:

    python scripts/deep_learning/run_deep_learning_experiments.py \
      --debug \
      --experiments temporal_tabular_small_random temporal_tabular_small_mlp_branch \
      --continue-on-error

Warm-start temporal-tabular branches from selected checkpoints:

    python scripts/deep_learning/run_deep_learning_experiments.py \
      --experiments temporal_tabular_small_ft_warmstart temporal_tabular_small_mlp_warmstart \
      --selected-ft-checkpoint outputs/models/deep_learning/ft_small_seed362559.pt \
      --selected-mlp-checkpoint outputs/models/deep_learning/mlp_model.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FEATURE_GROUPS = Path("outputs/sanity_checks/tables/feature_groups.json")
RETURN_ONLY_FEATURE_GROUPS = Path("outputs/tables/deep_learning_runner_return_only_feature_groups.json")
DEFAULT_SUMMARY = Path("outputs/tables/deep_learning_experiment_run_summary.csv")
RUNNER_LOG = Path("outputs/logs/deep_learning_experiment_runner.log")
LOG_DIR = Path("outputs/logs")
PREDICTION_DIR = Path("outputs/predictions")
TABLE_DIR = Path("outputs/tables")
DL_MODEL_DIR = Path("outputs/models/deep_learning")
BASELINE_MODEL_DIR = Path("outputs/models/baselines")


SUMMARY_COLUMNS = [
    "experiment_name",
    "run_name",
    "command",
    "status",
    "start_time",
    "end_time",
    "elapsed_seconds",
    "return_code",
    "prediction_file",
    "metrics_file",
    "monthly_rank_ic_file",
    "model_file",
    "log_file",
    "error_message",
]


@dataclass
class Experiment:
    experiment_name: str
    run_name: str
    script_path: Path
    args: list[str]
    prediction_file: Path
    metrics_file: Path
    monthly_rank_ic_file: Path
    model_file: Path | None
    log_file: Path
    supports_debug: bool = True
    supports_seed: bool = False
    supports_max_train_rows: bool = False
    prerequisites: list[tuple[Path, str]] = field(default_factory=list)

    def command(self, python_bin: str, debug: bool, seed: int, max_train_rows: int | None) -> list[str]:
        command = [python_bin, str(self.script_path)] + list(self.args)
        if debug and self.supports_debug:
            command.append("--debug")
        if self.supports_seed:
            command.extend(["--seed", str(seed)])
        if self.supports_max_train_rows and max_train_rows is not None:
            command.extend(["--max-train-rows", str(max_train_rows)])
        return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and track deep-learning experiment commands.")
    parser.add_argument("--debug", action="store_true", help="Append --debug to commands that support it.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write a pending summary.")
    parser.add_argument("--strict", action="store_true", help="Fail on missing scripts or prerequisites.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        help="Experiment names to run, or 'all'.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip if the expected prediction file exists.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running after a failed experiment.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used for child commands.")
    parser.add_argument("--max-parallel", type=int, default=1, help="Maximum concurrent experiment processes.")
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--run-prefix", default="", help="Prefix added to every run_name.")
    parser.add_argument("--seed", type=int, default=362559)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--only-print-commands", action="store_true", help="Alias for command printing without running.")
    parser.add_argument("--selected-ft-checkpoint", type=Path, default=None)
    parser.add_argument("--selected-mlp-checkpoint", type=Path, default=None)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def as_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def log_runner(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{now_iso()}] {message}"
    print(line, flush=True)
    with as_repo_path(RUNNER_LOG).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def debug_run_base(base_name: str) -> str:
    if base_name.startswith("temporal_tabular_"):
        return base_name.replace("temporal_tabular_", "temporal_tabular_debug_", 1)
    return f"debug_{base_name}"


def run_name_for(base_name: str, args: argparse.Namespace) -> str:
    name = debug_run_base(base_name) if args.debug else base_name
    if args.run_prefix:
        return f"{args.run_prefix}_{name}"
    return name


def ensure_return_only_feature_groups() -> Path:
    source = as_repo_path(DEFAULT_FEATURE_GROUPS)
    target = as_repo_path(RETURN_ONLY_FEATURE_GROUPS)
    if not source.exists():
        return RETURN_ONLY_FEATURE_GROUPS
    with source.open(encoding="utf-8") as handle:
        groups = json.load(handle)
    groups["jkp_feature_columns"] = []
    groups["selected_compustat_feature_columns"] = []
    groups["selected_compustat_missing_indicator_columns"] = []
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
    return RETURN_ONLY_FEATURE_GROUPS


def ft_paths(run_name: str) -> dict[str, Path]:
    return {
        "prediction": PREDICTION_DIR / f"{run_name}_predictions.parquet",
        "metrics": TABLE_DIR / f"{run_name}_model_metrics.csv",
        "monthly": TABLE_DIR / f"{run_name}_monthly_rank_ic.csv",
        "model": DL_MODEL_DIR / f"{run_name}.pt",
        "training_log": LOG_DIR / f"{run_name}_training.log",
        "runner_log": LOG_DIR / f"{run_name}_runner.log",
    }


def temporal_paths(run_name: str) -> dict[str, Path]:
    return ft_paths(run_name)


def mlp_classifier_paths(run_name: str) -> dict[str, Path]:
    output_dir = DL_MODEL_DIR / run_name
    return {
        "output_dir": output_dir,
        "prediction": output_dir / "mlp_validation_predictions.parquet",
        "metrics": output_dir / "mlp_training_history.csv",
        "monthly": output_dir / "mlp_validation_rank_ic_cls.csv",
        "model": output_dir / "mlp_model.pt",
        "runner_log": LOG_DIR / f"{run_name}_runner.log",
    }


def mlp_regressor_paths(run_name: str) -> dict[str, Path]:
    output_dir = DL_MODEL_DIR / run_name
    return {
        "output_dir": output_dir,
        "prediction": output_dir / "mlp_reg_validation_predictions.parquet",
        "metrics": output_dir / "mlp_reg_training_history.csv",
        "monthly": output_dir / "mlp_reg_validation_rank_ic_er.csv",
        "model": output_dir / "mlp_regressor.pt",
        "runner_log": LOG_DIR / f"{run_name}_runner.log",
    }


def xgb_paths(run_name: str, model_name: str) -> dict[str, Path]:
    table_dir = TABLE_DIR / run_name
    model_dir = BASELINE_MODEL_DIR / run_name
    return {
        "table_dir": table_dir,
        "model_dir": model_dir,
        "prediction": PREDICTION_DIR / f"{run_name}_predictions.parquet",
        "metrics": table_dir / "boosting_model_metrics.csv",
        "monthly": table_dir / "boosting_monthly_rank_ic.csv",
        "model": model_dir / model_name,
        "training_log": LOG_DIR / f"{run_name}_training.log",
        "runner_log": LOG_DIR / f"{run_name}_runner.log",
    }


def make_mlp_experiment(
    experiment_name: str,
    run_name: str,
    script_path: Path,
    paths: dict[str, Path],
    feature_groups: Path | None,
) -> Experiment:
    command_args = ["--output-dir", str(paths["output_dir"])]
    prerequisites: list[tuple[Path, str]] = []
    if feature_groups is not None:
        command_args.extend(["--feature-groups", str(feature_groups)])
        prerequisites.append((feature_groups, "return-only feature-groups file is missing"))
    return Experiment(
        experiment_name=experiment_name,
        run_name=run_name,
        script_path=script_path,
        args=command_args,
        prediction_file=paths["prediction"],
        metrics_file=paths["metrics"],
        monthly_rank_ic_file=paths["monthly"],
        model_file=paths["model"],
        log_file=paths["runner_log"],
        prerequisites=prerequisites,
    )


def make_ft_experiment(
    experiment_name: str,
    run_name: str,
    extra_args: list[str],
    feature_groups: Path | None = None,
) -> Experiment:
    paths = ft_paths(run_name)
    command_args = [
        "--run-name",
        run_name,
        "--log-output",
        str(paths["training_log"]),
    ] + extra_args
    prerequisites: list[tuple[Path, str]] = []
    if feature_groups is not None:
        command_args.extend(["--feature-groups", str(feature_groups)])
        prerequisites.append((feature_groups, "return-only feature-groups file is missing"))
    return Experiment(
        experiment_name=experiment_name,
        run_name=run_name,
        script_path=Path("scripts/deep_learning/train_ft_transformer.py"),
        args=command_args,
        prediction_file=paths["prediction"],
        metrics_file=paths["metrics"],
        monthly_rank_ic_file=paths["monthly"],
        model_file=paths["model"],
        log_file=paths["runner_log"],
        supports_seed=True,
        prerequisites=prerequisites,
    )


def make_temporal_experiment(
    experiment_name: str,
    run_name: str,
    extra_args: list[str],
    prerequisites: list[tuple[Path, str]] | None = None,
) -> Experiment:
    paths = temporal_paths(run_name)
    command_args = [
        "--run-name",
        run_name,
        "--log-output",
        str(paths["training_log"]),
    ] + extra_args
    return Experiment(
        experiment_name=experiment_name,
        run_name=run_name,
        script_path=Path("scripts/deep_learning/train_temporal_tabular_transformer.py"),
        args=command_args,
        prediction_file=paths["prediction"],
        metrics_file=paths["metrics"],
        monthly_rank_ic_file=paths["monthly"],
        model_file=paths["model"],
        log_file=paths["runner_log"],
        supports_seed=True,
        prerequisites=prerequisites or [],
    )


def make_xgb_experiment(experiment_name: str, run_name: str, model_name: str) -> Experiment:
    paths = xgb_paths(run_name, model_name)
    return Experiment(
        experiment_name=experiment_name,
        run_name=run_name,
        script_path=Path("scripts/baselines/07_train_baselines.py"),
        args=[
            "--only-boosting",
            "--boosting-predictions-output",
            str(paths["prediction"]),
            "--table-dir",
            str(paths["table_dir"]),
            "--model-dir",
            str(paths["model_dir"]),
            "--boosting-log",
            str(paths["training_log"]),
        ],
        prediction_file=paths["prediction"],
        metrics_file=paths["metrics"],
        monthly_rank_ic_file=paths["monthly"],
        model_file=paths["model"],
        log_file=paths["runner_log"],
        supports_max_train_rows=True,
    )


def build_experiments(args: argparse.Namespace) -> dict[str, Experiment]:
    return_only_groups = ensure_return_only_feature_groups()
    experiments: dict[str, Experiment] = {}

    def add(experiment: Experiment) -> None:
        experiments[experiment.experiment_name] = experiment

    add(
        make_mlp_experiment(
            "mlp_classifier_full",
            run_name_for("mlp_classifier_full", args),
            Path("scripts/deep_learning/train_mlp.py"),
            mlp_classifier_paths(run_name_for("mlp_classifier_full", args)),
            None,
        )
    )
    add(
        make_mlp_experiment(
            "mlp_regressor_full",
            run_name_for("mlp_regressor_full", args),
            Path("scripts/deep_learning/train_mlp_regressor.py"),
            mlp_regressor_paths(run_name_for("mlp_regressor_full", args)),
            None,
        )
    )
    add(
        make_mlp_experiment(
            "mlp_classifier_return_only",
            run_name_for("mlp_classifier_return_only", args),
            Path("scripts/deep_learning/train_mlp.py"),
            mlp_classifier_paths(run_name_for("mlp_classifier_return_only", args)),
            return_only_groups,
        )
    )
    add(
        make_mlp_experiment(
            "mlp_regressor_return_only",
            run_name_for("mlp_regressor_return_only", args),
            Path("scripts/deep_learning/train_mlp_regressor.py"),
            mlp_regressor_paths(run_name_for("mlp_regressor_return_only", args)),
            return_only_groups,
        )
    )

    ft_small = [
        "--d-token",
        "32",
        "--n-layers",
        "2",
        "--n-heads",
        "4",
        "--attention-dropout",
        "0.2",
        "--ffn-dropout",
        "0.2",
        "--learning-rate",
        "1e-4",
        "--patience",
        "6",
    ]
    ft_tiny = [
        "--d-token",
        "16",
        "--n-layers",
        "1",
        "--n-heads",
        "2",
        "--attention-dropout",
        "0.2",
        "--ffn-dropout",
        "0.2",
        "--learning-rate",
        "1e-4",
        "--patience",
        "6",
    ]
    add(make_ft_experiment("ft_classifier_small_full", run_name_for("ft_classifier_small_full", args), ft_small))
    add(
        make_ft_experiment(
            "ft_regressor_small_full",
            run_name_for("ft_regressor_small_full", args),
            ft_small + ["--selection-metric", "er_train_rank_ic"],
        )
    )
    add(make_ft_experiment("ft_classifier_tiny_full", run_name_for("ft_classifier_tiny_full", args), ft_tiny))
    add(
        make_ft_experiment(
            "ft_classifier_return_only",
            run_name_for("ft_classifier_return_only", args),
            ft_small,
            feature_groups=return_only_groups,
        )
    )

    add(
        make_temporal_experiment(
            "temporal_tabular_small_random",
            run_name_for("temporal_tabular_small_random", args),
            [
                "--architecture-preset",
                "small",
                "--tabular-branch",
                "ft_transformer",
                "--learning-rate",
                "1e-4",
                "--patience",
                "6",
            ],
        )
    )
    add(
        make_temporal_experiment(
            "temporal_tabular_tiny_random",
            run_name_for("temporal_tabular_tiny_random", args),
            [
                "--architecture-preset",
                "tiny",
                "--tabular-branch",
                "ft_transformer",
                "--learning-rate",
                "1e-4",
                "--patience",
                "6",
            ],
        )
    )
    add(
        make_temporal_experiment(
            "temporal_tabular_small_mlp_branch",
            run_name_for("temporal_tabular_small_mlp_branch", args),
            [
                "--architecture-preset",
                "small",
                "--tabular-branch",
                "mlp",
                "--learning-rate",
                "1e-4",
                "--patience",
                "6",
            ],
        )
    )

    if args.selected_ft_checkpoint is not None:
        ft_checkpoint = args.selected_ft_checkpoint
        ft_prerequisites = [(ft_checkpoint, "selected FT checkpoint is missing")]
    else:
        ft_checkpoint = Path("PATH_TO_SELECTED_FT_CHECKPOINT")
        ft_prerequisites = [(ft_checkpoint, "selected FT checkpoint was not provided")]
    add(
        make_temporal_experiment(
            "temporal_tabular_small_ft_warmstart",
            run_name_for("temporal_tabular_small_ft_warmstart", args),
            [
                "--architecture-preset",
                "small",
                "--tabular-branch",
                "ft_transformer",
                "--init-tabular-from-ft-checkpoint",
                str(ft_checkpoint),
                "--freeze-tabular-epochs",
                "2",
                "--temporal-learning-rate",
                "1e-4",
                "--tabular-learning-rate",
                "1e-5",
                "--fusion-learning-rate",
                "1e-4",
                "--patience",
                "6",
            ],
            ft_prerequisites,
        )
    )

    if args.selected_mlp_checkpoint is not None:
        mlp_checkpoint = args.selected_mlp_checkpoint
        mlp_prerequisites = [(mlp_checkpoint, "selected MLP checkpoint is missing")]
    else:
        mlp_checkpoint = Path("PATH_TO_SELECTED_MLP_CHECKPOINT")
        mlp_prerequisites = [(mlp_checkpoint, "selected MLP checkpoint was not provided")]
    add(
        make_temporal_experiment(
            "temporal_tabular_small_mlp_warmstart",
            run_name_for("temporal_tabular_small_mlp_warmstart", args),
            [
                "--architecture-preset",
                "small",
                "--tabular-branch",
                "mlp",
                "--init-tabular-from-mlp-checkpoint",
                str(mlp_checkpoint),
                "--freeze-tabular-epochs",
                "2",
                "--temporal-learning-rate",
                "1e-4",
                "--tabular-learning-rate",
                "1e-5",
                "--fusion-learning-rate",
                "1e-4",
                "--patience",
                "6",
            ],
            mlp_prerequisites,
        )
    )

    add(
        make_xgb_experiment(
            "xgboost_classifier_full",
            run_name_for("xgboost_classifier_full", args),
            "gradient_boosting_classifier.joblib",
        )
    )
    add(
        make_xgb_experiment(
            "xgboost_regressor_full",
            run_name_for("xgboost_regressor_full", args),
            "gradient_boosting_regressor.joblib",
        )
    )
    return experiments


def select_experiments(all_experiments: dict[str, Experiment], requested: list[str]) -> list[Experiment]:
    if "all" in requested:
        return list(all_experiments.values())
    unknown = [name for name in requested if name not in all_experiments]
    if unknown:
        known = ", ".join(sorted(all_experiments))
        raise ValueError(f"Unknown experiment(s): {', '.join(unknown)}. Known experiments: {known}")
    return [all_experiments[name] for name in requested]


def row_for_experiment(experiment: Experiment, command: list[str]) -> dict[str, Any]:
    return {
        "experiment_name": experiment.experiment_name,
        "run_name": experiment.run_name,
        "command": shlex.join(command),
        "status": "pending",
        "start_time": "",
        "end_time": "",
        "elapsed_seconds": "",
        "return_code": "",
        "prediction_file": display_path(experiment.prediction_file),
        "metrics_file": display_path(experiment.metrics_file),
        "monthly_rank_ic_file": display_path(experiment.monthly_rank_ic_file),
        "model_file": display_path(experiment.model_file),
        "log_file": display_path(experiment.log_file),
        "error_message": "",
    }


def write_summary(rows: list[dict[str, Any]], output_summary: Path) -> None:
    path = as_repo_path(output_summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def preflight_issue(experiment: Experiment) -> tuple[str, str] | None:
    script = as_repo_path(experiment.script_path)
    if not script.exists():
        return ("script", f"script is missing: {experiment.script_path}")
    for path, reason in experiment.prerequisites:
        if not as_repo_path(path).exists():
            return ("prerequisite", f"{reason}: {path}")
    return None


def command_environment(device: str) -> dict[str, str]:
    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def run_one(
    experiment: Experiment,
    row: dict[str, Any],
    command: list[str],
    env: dict[str, str],
) -> dict[str, Any]:
    log_runner(f"Running {experiment.experiment_name}: {shlex.join(command)}")
    log_path = as_repo_path(experiment.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    row["start_time"] = now_iso()
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    end = time.monotonic()
    row["end_time"] = now_iso()
    row["elapsed_seconds"] = f"{end - start:.3f}"
    row["return_code"] = process.returncode
    if process.returncode == 0:
        row["status"] = "success"
        row["error_message"] = ""
        log_runner(f"Finished {experiment.experiment_name}: success")
    else:
        row["status"] = "failed"
        row["error_message"] = f"Command failed with return code {process.returncode}. See {experiment.log_file}."
        log_runner(f"Finished {experiment.experiment_name}: failed with return code {process.returncode}")
    return row


def run_parallel(
    runnable: list[tuple[Experiment, dict[str, Any], list[str]]],
    env: dict[str, str],
    max_parallel: int,
    continue_on_error: bool,
) -> bool:
    active: list[dict[str, Any]] = []
    index = 0
    any_failed = False

    while index < len(runnable) or active:
        while index < len(runnable) and len(active) < max_parallel and (continue_on_error or not any_failed):
            experiment, row, command = runnable[index]
            index += 1
            log_runner(f"Running {experiment.experiment_name}: {shlex.join(command)}")
            log_path = as_repo_path(experiment.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w", encoding="utf-8")
            row["start_time"] = now_iso()
            start = time.monotonic()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            active.append(
                {
                    "experiment": experiment,
                    "row": row,
                    "process": process,
                    "log_handle": log_handle,
                    "start": start,
                }
            )

        time.sleep(0.25)
        remaining: list[dict[str, Any]] = []
        for item in active:
            process = item["process"]
            return_code = process.poll()
            if return_code is None:
                remaining.append(item)
                continue
            item["log_handle"].close()
            row = item["row"]
            experiment = item["experiment"]
            row["end_time"] = now_iso()
            row["elapsed_seconds"] = f"{time.monotonic() - item['start']:.3f}"
            row["return_code"] = return_code
            if return_code == 0:
                row["status"] = "success"
                log_runner(f"Finished {experiment.experiment_name}: success")
            else:
                row["status"] = "failed"
                row["error_message"] = f"Command failed with return code {return_code}. See {experiment.log_file}."
                any_failed = True
                log_runner(f"Finished {experiment.experiment_name}: failed with return code {return_code}")
        active = remaining

    return any_failed


def prepare_rows_and_runnable(
    selected: list[Experiment],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[tuple[Experiment, dict[str, Any], list[str]]], bool]:
    rows: list[dict[str, Any]] = []
    runnable: list[tuple[Experiment, dict[str, Any], list[str]]] = []
    has_preflight_failure = False
    print_only = args.dry_run or args.only_print_commands

    for experiment in selected:
        command = experiment.command(args.python_bin, args.debug, args.seed, args.max_train_rows)
        row = row_for_experiment(experiment, command)
        log_runner(f"Command for {experiment.experiment_name}: {shlex.join(command)}")

        issue = preflight_issue(experiment)
        if issue is not None:
            issue_type, reason = issue
            row["status"] = "failed" if args.strict and issue_type == "script" else "skipped"
            row["error_message"] = reason
            has_preflight_failure = has_preflight_failure or (args.strict and issue_type == "script")
            log_runner(f"{row['status'].capitalize()} {experiment.experiment_name}: {reason}")
        elif args.skip_existing and as_repo_path(experiment.prediction_file).exists():
            row["status"] = "skipped"
            row["error_message"] = f"Prediction file already exists: {experiment.prediction_file}"
            log_runner(f"Skipped {experiment.experiment_name}: prediction file exists")
        elif print_only:
            row["status"] = "pending"
        else:
            runnable.append((experiment, row, command))

        rows.append(row)
    return rows, runnable, has_preflight_failure


def print_final_summary(rows: list[dict[str, Any]], output_summary: Path) -> None:
    requested = len(rows)
    succeeded = sum(row["status"] == "success" for row in rows)
    skipped = sum(row["status"] == "skipped" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    failed_names = [row["experiment_name"] for row in rows if row["status"] == "failed"]

    print()
    print(f"Experiments requested: {requested}")
    print(f"Succeeded: {succeeded}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Summary CSV: {output_summary}")
    if failed_names:
        print("Failed experiments: " + ", ".join(failed_names))


def main() -> None:
    args = parse_args()
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1.")

    as_repo_path(RUNNER_LOG).parent.mkdir(parents=True, exist_ok=True)
    as_repo_path(RUNNER_LOG).write_text("", encoding="utf-8")

    all_experiments = build_experiments(args)
    selected = select_experiments(all_experiments, args.experiments)
    rows, runnable, preflight_failed = prepare_rows_and_runnable(selected, args)
    write_summary(rows, args.output_summary)

    if preflight_failed and not args.continue_on_error:
        print_final_summary(rows, args.output_summary)
        raise SystemExit(1)

    if args.dry_run or args.only_print_commands:
        print_final_summary(rows, args.output_summary)
        return

    env = command_environment(args.device)
    any_failed = preflight_failed
    if args.max_parallel == 1:
        for experiment, row, command in runnable:
            run_one(experiment, row, command, env)
            write_summary(rows, args.output_summary)
            if row["status"] == "failed":
                any_failed = True
                if not args.continue_on_error:
                    break
    else:
        any_failed = run_parallel(runnable, env, args.max_parallel, args.continue_on_error) or any_failed
        write_summary(rows, args.output_summary)

    write_summary(rows, args.output_summary)
    print_final_summary(rows, args.output_summary)
    if any_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
