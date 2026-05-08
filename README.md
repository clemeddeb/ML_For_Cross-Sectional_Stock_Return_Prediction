# ML_For_Finance_Project-AxelTurinPlessia-362559-ClementMeddeb-346164

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
python scripts/build_linked_dataset.py
```

The generated outputs are written under `Dataset/Processed/`:

- `crsp_with_gvkey.parquet`
- `crsp_compustat_panel.parquet`
- `sec_10k_with_links.parquet`
- `earnings_calls_with_gvkey.parquet`
- `link_summary.json`

By default, Compustat observations are made available three months after `datadate` to reduce look-ahead bias. To keep the linked text files smaller, add `--drop-text`.

## Raw Parquet Conversion

To mirror the raw predictor and target datasets as parquet files, run:

```bash
python scripts/convert_raw_to_parquet.py
```

This writes CSV conversions and copies existing parquet inputs under `Dataset/Parquet/`, while skipping documentation files such as PDFs. Use `Dataset/Parquet/` when you need faster access to the unlinked raw data, and use `Dataset/Processed/` for the next modeling steps that require linked CRSP, Compustat, SEC filing, and earnings-call identifiers.
