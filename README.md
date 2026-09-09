# Machine Learning for Cross-Sectional Stock Return Prediction

This repository contains the final, script-only submission for the ML for Finance project. The objective is to compare classical machine-learning, gradient-boosting, deep-learning, FT-Transformer, and Temporal-Tabular Transformer models for monthly cross-sectional stock-return prediction and portfolio construction.

The submitted repository is designed to reproduce the final reported tables and figures from saved prediction and model artifacts. Training is not required for grading.

## Project Deliverables

- [Final Report](./Final_Report.pdf)

## Repository Structure

```text
.
├── main.py
├── requirements.txt
├── scripts/
│   ├── backtesting/
│   │   ├── run_portfolio_analysis.py
│   │   └── run_drawdown_overlay_analysis.py
│   ├── baselines/
│   ├── data_processing/
│   ├── deep_learning/
│   └── forecasting_analysis.py
├── outputs/
│   ├── predictions/
│   ├── models/
│   ├── tables/
│   └── backtests/
│       └── figures/
├── Dataset/
│   └── README.md
├── Final_Report.pdf
└── Machine_Learning_in_Finance___Project_Instructions.pdf
```

There are no notebooks in the final repository. Exploratory notebook work has been replaced by reproducible Python scripts.

## Reproducing Final Results

Create a clean Python environment, install dependencies, and run the final evaluation pipeline:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

`main.py` performs only evaluation and reporting steps. It does not train any model. It:

1. verifies that required saved predictions and selected model artifacts exist;
2. regenerates forecasting summary tables;
3. reruns the 25 bps raw portfolio analysis;
4. reruns the drawdown-overlay robustness analysis;
5. writes final outputs under `outputs/tables/`, `outputs/backtests/`, and `outputs/backtests/figures/`.

Individual commands are also available:

```bash
python scripts/forecasting_analysis.py
python scripts/backtesting/run_portfolio_analysis.py --cost-bps 25
python scripts/backtesting/run_drawdown_overlay_analysis.py --cost-bps 25 --top-n 300
```

## Included Final Artifacts

The repository includes saved prediction files in `outputs/predictions/` for:

- baseline and boosted baseline scores;
- XGBoost, MLP classifier, and MLP regressor scores;
- FT-Transformer variants used in the forecasting comparison;
- Temporal-Tabular Transformer variants used in the forecasting comparison;
- the consolidated TTT backtest prediction file used by the final portfolio analysis.

The repository also includes selected trained model artifacts in `outputs/models/`, including baseline model objects, MLP checkpoints, FT-Transformer checkpoints, and selected TTT checkpoints. XGBoost is represented in the final reproducible evaluation by the saved prediction artifact and selected-hyperparameter tables. These files are kept for transparency and reproducibility of the submitted artifacts, but `main.py` does not retrain or reload them for evaluation.

The final evaluation scripts run from saved prediction artifacts, which already contain the identifiers, split labels, scores, and next-month returns needed for forecasting and backtesting.

For a full rebuild or retraining run, restore the original raw or parquet datasets under `Dataset/` using the layout described in `Dataset/README.md`. The lightweight parquet inputs are preferred for git submission, including `Dataset/Parquet/Targets/monthly_crsp.parquet`, `Dataset/Parquet/Predictors/CompFirmCharac_parquet/`, `Dataset/Parquet/Predictors/[usa]_[all_factors]_[monthly]_[vw_cap].parquet`, and `Dataset/Linking/ccm_links.parquet`. The scripts under `scripts/data_processing/`, `scripts/baselines/`, and `scripts/deep_learning/` implement the original pipeline from raw/parquet data to features, model training, predictions, and evaluation. That full rebuild requires access to the original CRSP/Compustat/WRDS-style data and optional WRDS credentials for the CCM link table.

Typical rebuild order from raw/parquet data:

```bash
# Optional only when starting from CSV/raw files instead of the submitted parquet inputs:
# python scripts/data_processing/convert_raw_to_parquet.py
python scripts/data_processing/01_deduplicate_crsp.py
python scripts/data_processing/build_linked_dataset.py
python scripts/data_processing/02_build_model_panel.py
python scripts/data_processing/03_build_return_features.py
python scripts/data_processing/04_add_jkp_features.py
python scripts/data_processing/05_select_compustat_features.py
python scripts/data_processing/06_create_splits.py
python scripts/baselines/07_train_baselines.py
python scripts/deep_learning/run_deep_learning_experiments.py
python scripts/deep_learning/run_ft_transformer_experiments.py
```

The default grading/reproduction workflow remains `python main.py`; it avoids retraining and uses the submitted saved artifacts.

## Main Final Outputs

Forecasting tables:

- `outputs/tables/forecasting_summary.csv`
- `outputs/tables/forecasting_family_top3_model_score_pairs.csv`
- `outputs/tables/forecasting_best_family_model_comparison.csv`
- `outputs/tables/forecasting_xgb_top3_model_score_pairs.csv`
- `outputs/tables/forecasting_mlp_top3_model_score_pairs.csv`
- `outputs/tables/forecasting_ft_transformer_top3_model_score_pairs.csv`
- `outputs/tables/forecasting_ttt_top3_model_score_pairs.csv`

Portfolio-analysis tables:

- `outputs/backtests/portfolio_full_grid_metrics_25bps.csv`
- `outputs/backtests/portfolio_validation_preselection_25bps.csv`
- `outputs/backtests/portfolio_best_specs_validation_25bps.csv`
- `outputs/backtests/portfolio_best_specs_test_25bps.csv`
- `outputs/backtests/portfolio_selected_model_comparison_25bps.csv`
- `outputs/backtests/portfolio_selected_annual_return_mdd_25bps.csv`

Drawdown-overlay robustness tables:

- `outputs/backtests/drawdown_overlay_average_changes_25bps.csv`
- `outputs/backtests/drawdown_overlay_average_changes_by_family_25bps.csv`
- `outputs/backtests/drawdown_overlay_strategy_results_25bps.csv`
- `outputs/backtests/drawdown_overlay_summary_25bps.csv`

Figures:

- `outputs/backtests/figures/test_cumulative_wealth_best_specs_25bps.png`
- `outputs/backtests/figures/test_drawdown_best_specs_25bps.png`
- `outputs/backtests/figures/selected_model_test_metric_bars_25bps.png`
- `outputs/backtests/figures/validation_net_sharpe_vs_max_dd_25bps.png`
- `outputs/backtests/figures/selected_strategies_annual_return_mdd_heatmap_25bps.png`
- `outputs/backtests/figures/drawdown_overlay_test_metric_bars_25bps.png`

## Notes on Large Artifacts

The final prediction artifacts and full monthly-return panel are large because they contain row-level scores and strategy returns for the full validation and test samples. They are required to rerun or inspect the final evaluation without retraining. If the repository is hosted on a service with a strict file-size limit, these files should be tracked with Git LFS or provided as a release artifact while preserving the same relative paths.

## No Notebooks

The final submission intentionally excludes `*.ipynb` files and notebook checkpoint folders. All final results are generated from Python scripts.
