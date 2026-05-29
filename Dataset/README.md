# Dataset Folder

This folder is intentionally kept in the repository layout because the training
and rebuild scripts expect the original data under `Dataset/`.

The large/raw datasets are not currently present in this working tree after the
cleanup. To rebuild the full pipeline from scratch, restore the original data or
parquet exports using the paths below.

Core lightweight parquet inputs for rebuilding the final numeric feature panel
include:

- `Dataset/Parquet/Targets/monthly_crsp.parquet`
- `Dataset/Parquet/Predictors/CompFirmCharac_parquet/`
- `Dataset/Parquet/Predictors/[usa]_[all_factors]_[monthly]_[vw_cap].parquet`
- `Dataset/Linking/ccm_links.parquet`, or generate it with
  `scripts/data_processing/download_ccm_links.py`

`CompFirmCharac_parquet/` is a partitioned parquet dataset split into files
below common 100 MB git hosting limits. The large CSV copies
`Dataset/Targets/monthly_crsp.csv` and
`Dataset/Predictors/CompFirmCharac.csv` are not needed when these parquet inputs
are present.

Optional text/linking inputs:

- `Dataset/Predictors/10K_fillings.parquet`
- `Dataset/Predictors/sm-calls_with_connectors.parquet`

The final reported models do not use 10-K or earnings-call text features. The
legacy linked-dataset script can link these files and write
`sec_10k_with_links.parquet` / `earnings_calls_with_gvkey.parquet`, but those
outputs are not consumed by the final numeric feature-selection or training
scripts.

Main processed outputs created by the rebuild pipeline:

- `Dataset/Processed/crsp_monthly_deduped.parquet`
- `Dataset/Processed/crsp_compustat_panel.parquet`
- `Dataset/Processed/model_panel_monthly_base.parquet`
- `Dataset/Processed/model_panel_with_return_features.parquet`
- `Dataset/Processed/model_panel_with_return_jkp_features.parquet`
- `Dataset/Processed/model_panel_full_features.parquet`
- `Dataset/Processed/model_panel_full_features_with_splits.parquet`
