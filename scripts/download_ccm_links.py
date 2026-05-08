#!/usr/bin/env python3
"""Download the WRDS CRSP/Compustat Merged link history table.

The output is a compact parquet file used by ``build_linked_dataset.py``.
Credentials are read from ``WRDS_USERNAME`` and ``WRDS_PASSWORD`` or, if the
password is not set, from an interactive prompt.
"""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

import pandas as pd
import psycopg2


DEFAULT_OUTPUT = Path("Dataset/Linking/ccm_links.parquet")
DEFAULT_HOST = "wrds-pgdata.wharton.upenn.edu"
DEFAULT_PORT = 9737
DEFAULT_DB = "wrds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CCM links from WRDS crsp.ccmxpf_lnkhist."
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("WRDS_USERNAME"),
        help="WRDS username. Defaults to WRDS_USERNAME.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output parquet path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"WRDS PostgreSQL host. Defaults to {DEFAULT_HOST}.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"WRDS PostgreSQL port. Defaults to {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--dbname",
        default=DEFAULT_DB,
        help=f"WRDS PostgreSQL database. Defaults to {DEFAULT_DB}.",
    )
    parser.add_argument(
        "--link-types",
        nargs="+",
        default=["LC", "LU", "LS"],
        help="CCM link types to retain. Defaults to LC LU LS.",
    )
    parser.add_argument(
        "--link-prim",
        nargs="+",
        default=["P", "C"],
        help="CCM link primacy values to retain. Defaults to P C.",
    )
    return parser.parse_args()


def normalize_links(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    df["gvkey"] = df["gvkey"].astype("string").str.strip().str.zfill(6)
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    df["permco"] = pd.to_numeric(df["permco"], errors="coerce").astype("Int64")
    df["linkdt"] = pd.to_datetime(df["linkdt"], errors="coerce")
    df["linkenddt"] = pd.to_datetime(df["linkenddt"], errors="coerce").fillna(
        pd.Timestamp("2100-12-31")
    )
    df["linktype"] = df["linktype"].astype("string").str.strip()
    df["linkprim"] = df["linkprim"].astype("string").str.strip()
    df["liid"] = df["liid"].astype("string").str.strip()

    df = df[df["gvkey"].notna() & df["permno"].notna() & df["linkdt"].notna()]
    df = df.sort_values(["gvkey", "permno", "linkdt", "linkenddt"])
    return df.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if not args.username:
        raise SystemExit("Set WRDS_USERNAME or pass --username.")

    password = os.environ.get("WRDS_PASSWORD")
    if not password:
        password = getpass.getpass("WRDS password: ")

    query = """
        select
            gvkey,
            lpermno as permno,
            lpermco as permco,
            liid,
            linktype,
            linkprim,
            linkdt,
            linkenddt
        from crsp.ccmxpf_lnkhist
        where lpermno is not null
          and lpermno > 0
          and linktype = any(%s)
          and linkprim = any(%s)
    """

    with psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.username,
        password=password,
        sslmode="require",
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (list(args.link_types), list(args.link_prim)))
            links = pd.DataFrame.from_records(
                cursor.fetchall(),
                columns=[desc[0] for desc in cursor.description],
            )

    links = normalize_links(links)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    links.to_parquet(args.output, index=False)

    print(f"Wrote {len(links):,} CCM links to {args.output}")
    print(
        "Date range: "
        f"{links['linkdt'].min().date()} to {links['linkenddt'].max().date()}"
    )


if __name__ == "__main__":
    main()
