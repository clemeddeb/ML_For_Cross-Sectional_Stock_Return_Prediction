# ML_For_Finance_Project-AxelTurinPlessia-362559-ClementMeddeb-346164

## CRSP Monthly Deduplication

Before linking or modeling, create a deduplicated monthly CRSP target file:

```bash
python scripts/01_deduplicate_crsp.py
```

This reads `Dataset/Targets/monthly_crsp.csv` without overwriting it and writes
`Dataset/Processed/crsp_monthly_deduped.parquet`. The script first removes exact
full-row duplicates. For any remaining duplicate `PERMNO`/`MthCalDt` keys, it
requires the core target/link fields (`MthRet`, `sprtrn`, `PERMCO`, `HdrCUSIP`)
to agree; if they do not, it stops and writes the conflicting rows under
`outputs/sanity_checks/tables/`. When core fields agree, only descriptive
metadata differs, so one row is selected deterministically by preferring valid
`CUSIP`, then `Ticker`, `TradingSymbol`, `SICCD`, and `NAICS`. This preserves
the return target while keeping the richest available metadata row.

## Linked Dataset Build

The project uses two identifier bridges:

- SEC 10-K filings -> Compustat: normalized SEC CIK.
- Compustat -> CRSP: WRDS CRSP/Compustat Merged link table (`crsp.ccmxpf_lnkhist`), matched by `PERMNO`/`GVKEY` and valid link date ranges.

I downloaded the WRDS CCM linking table and used it to attach Compustat `gvkey` identifiers to monthly CRSP observations through valid `PERMNO`-date links; I then matched Compustat accounting data with a three-month reporting lag, linked 10-K filings to Compustat through normalized SEC CIKs before mapping them to CRSP, and linked earnings-call transcripts directly through their existing `permno`. This produced usable linked parquet files, but the SEC 10-K match is incomplete because not every CIK maps cleanly to a public Compustat/CRSP security in the available period. Future improvements could include manually resolving unmatched CIKs, validating alternative CCM link types, handling multiple share classes more explicitly, and refining the accounting-data lag assumptions.

Credentials should not be committed. Set them only in your shell session:

```bash
source .venv/bin/activate
export WRDS_USERNAME="your_wrds_username"
export WRDS_PASSWORD="your_wrds_password"
```

Download the CCM link table:

```bash
python scripts/download_ccm_links.py
```

Build the linked project files:

```bash
python scripts/01_deduplicate_crsp.py
python scripts/build_linked_dataset.py
```

The generated outputs are written under `Dataset/Processed/`:

- `crsp_monthly_deduped.parquet`
- `crsp_with_gvkey.parquet`
- `crsp_compustat_panel.parquet`
- `sec_10k_with_links.parquet`
- `earnings_calls_with_gvkey.parquet`
- `link_summary.json`

The linking workflow uses `crsp_monthly_deduped.parquet` as the monthly CRSP
input, so the raw CSV is not consumed directly by downstream linked datasets.
By default, Compustat observations are made available three months after `datadate` to reduce look-ahead bias. To keep the linked text files smaller, add `--drop-text`.

## Monthly Modeling Panel

Build the base monthly modeling panel after the linked-data validation passes:

```bash
python scripts/02_build_model_panel.py
```

This reads `Dataset/Processed/crsp_compustat_panel.parquet` and writes
`Dataset/Processed/model_panel_monthly_base.parquet`. Each row is dated month
`t`: identifiers, raw month-`t` CRSP returns, and lagged Compustat fields are
kept as information available at or before `t`. The target `target_ret_1m` is
the next calendar month's `MthRet` for the same `PERMNO`; rows without a true
next-month return are dropped. Cross-sectional labels are assigned within each
month using realized `target_ret_1m`: quintile 1 is the bottom quintile,
quintile 5 is the top quintile, and `top_bottom_label` maps bottom/middle/top
to `0/1/2`. The target is not used to create predictors.

## Raw Parquet Conversion

To mirror the raw predictor and target datasets as parquet files, run:

```bash
python scripts/convert_raw_to_parquet.py
```

This writes CSV conversions and copies existing parquet inputs under `Dataset/Parquet/`, while skipping documentation files such as PDFs. Use `Dataset/Parquet/` when you need faster access to the unlinked raw data, and use `Dataset/Processed/` for the next modeling steps that require linked CRSP, Compustat, SEC filing, and earnings-call identifiers.
