"""One-time fetch of real Bitcoin ledger data from the BigQuery public dataset.

This is the only online step of the dataset build. Everything downstream is offline.

Usage (run from C:\\IIITS\\SIH):
    python tools/fetch_bigquery.py auth    --project <gcp-project-id>
    python tools/fetch_bigquery.py track-r --project <gcp-project-id> [--dry-run]
    python tools/fetch_bigquery.py track-m --project <gcp-project-id> --start 2026-09-20T00:00:00 --hours 6 [--dry-run]

track-r: the ~202.8k real transactions behind the Elliptic dataset (2016-01 .. 2017-10),
         located via the public deanonymisation mapping (deanonyized99.5/Result.csv).
track-m: a contiguous recent window of mainnet transactions (modern-era background).

Values in crypto_bitcoin are integer satoshis. Output is Parquet under data/raw/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pydata_google_auth.auth as pga
from google.auth.transport.requests import Request
from google.cloud import bigquery
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

ROOT = Path(__file__).resolve().parents[1]
CREDS_CACHE = Path.home() / ".argus" / "bq_user_credentials.json"
AUTH_PORT = 8765
RAW = ROOT / "data" / "raw"
MAPPING_CSV = ROOT / "deanonyized99.5" / "Result.csv"
SCOPES = ["https://www.googleapis.com/auth/bigquery"]
PUBLIC_TX = "bigquery-public-data.crypto_bitcoin.transactions"
WORK_DATASET = "argus_work"

# Only the leaf columns we need are referenced, which keeps bytes scanned (= free-tier quota) low.
TX_SELECT = """
  t.hash, t.block_number, t.block_timestamp, t.size, t.virtual_size, t.version, t.lock_time,
  t.input_count, t.output_count, t.input_value, t.output_value, t.fee, t.is_coinbase,
  ARRAY(SELECT AS STRUCT i.index, i.spent_transaction_hash, i.spent_output_index,
                         i.sequence, i.type, i.addresses, i.value
        FROM UNNEST(t.inputs) i ORDER BY i.index) AS inputs,
  ARRAY(SELECT AS STRUCT o.index, o.type, o.addresses, o.value
        FROM UNNEST(t.outputs) o ORDER BY o.index) AS outputs
"""


def user_credentials() -> Credentials:
    """Browser sign-in on a fixed loopback port (8080 is taken on this machine), cached after the first run."""
    if CREDS_CACHE.exists():
        creds = Credentials.from_authorized_user_file(str(CREDS_CACHE), SCOPES)
        if creds.valid:
            return creds
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            CREDS_CACHE.write_text(creds.to_json())
            return creds
    flow = InstalledAppFlow.from_client_config(
        {"installed": {
            "client_id": pga.DESKTOP_CLIENT_ID,
            "client_secret": pga.DESKTOP_CLIENT_SECRET,
            "auth_uri": pga.GOOGLE_AUTH_URI,
            "token_uri": pga.GOOGLE_TOKEN_URI,
        }},
        SCOPES,
    )
    creds = flow.run_local_server(host="localhost", port=AUTH_PORT, open_browser=True,
                                  authorization_prompt_message="Open this URL to authorize: {url}")
    CREDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CREDS_CACHE.write_text(creds.to_json())
    return creds


def client(project: str) -> bigquery.Client:
    return bigquery.Client(project=project, credentials=user_credentials())


def run_query(bq: bigquery.Client, sql: str, dry_run: bool) -> pd.DataFrame | None:
    dry = bq.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    gb = dry.total_bytes_processed / 1e9
    print(f"[estimate] this query scans {gb:.1f} GB (free tier: 1000 GB / month)")
    if dry_run:
        return None
    t0 = time.time()
    job = bq.query(sql)
    job.result()
    print(f"[query] done in {time.time() - t0:.0f}s, downloading rows ...")
    # REST download (no Storage API) so it also works in the BigQuery sandbox.
    df = job.to_dataframe(create_bqstorage_client=False, progress_bar_type=None)
    print(f"[download] {len(df):,} rows in {time.time() - t0:.0f}s total")
    return df


def ensure_mapping_table(bq: bigquery.Client, project: str) -> str:
    table_id = f"{project}.{WORK_DATASET}.elliptic_map"
    bq.create_dataset(f"{project}.{WORK_DATASET}", exists_ok=True)
    try:
        n = bq.get_table(table_id).num_rows
        print(f"[mapping] {table_id} already present ({n:,} rows)")
        return table_id
    except Exception:
        pass
    m = pd.read_csv(MAPPING_CSV, dtype={"txId": "int64", "transaction": "string"})
    m = m.rename(columns={"txId": "elliptic_txid", "transaction": "hash"})
    job = bq.load_table_from_dataframe(
        m, table_id, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    )
    job.result()
    print(f"[mapping] uploaded {len(m):,} rows to {table_id}")
    return table_id


def save(df: pd.DataFrame, out: Path, meta: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    meta = {**meta, "rows": int(len(df)), "fetched_at_utc": pd.Timestamp.utcnow().isoformat()}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[saved] {out} ({out.stat().st_size / 1e6:.1f} MB)")


def cmd_track_r(args) -> None:
    bq = client(args.project)
    table_id = ensure_mapping_table(bq, args.project)
    # Mapped Elliptic window is 2016-01-01 .. 2017-10-04; one month of slack on each side.
    sql = f"""
    SELECT m.elliptic_txid, {TX_SELECT}
    FROM `{PUBLIC_TX}` t
    JOIN `{table_id}` m ON t.hash = m.hash
    WHERE t.block_timestamp_month BETWEEN '2015-12-01' AND '2017-11-01'
    """
    df = run_query(bq, sql, args.dry_run)
    if df is not None:
        save(df, RAW / "track_r" / "elliptic_real_txs.parquet",
             {"source": PUBLIC_TX, "mapping": str(MAPPING_CSV.name), "sql": sql})


def cmd_track_m(args) -> None:
    bq = client(args.project)
    start = pd.Timestamp(args.start, tz="UTC")
    end = start + pd.Timedelta(hours=args.hours)
    months = sorted({start.strftime("%Y-%m-01"), end.strftime("%Y-%m-01")})
    sql = f"""
    SELECT {TX_SELECT}
    FROM `{PUBLIC_TX}` t
    WHERE t.block_timestamp_month IN ({", ".join(f"'{m}'" for m in months)})
      AND t.block_timestamp >= '{start.isoformat()}' AND t.block_timestamp < '{end.isoformat()}'
    """
    df = run_query(bq, sql, args.dry_run)
    if df is not None:
        tag = start.strftime("%Y%m%dT%H") + f"_{args.hours}h"
        save(df, RAW / "track_m" / f"mainnet_{tag}.parquet",
             {"source": PUBLIC_TX, "start": start, "end": end, "sql": sql})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("auth", "track-r", "track-m"):
        s = sub.add_parser(name)
        s.add_argument("--project", required=True, help="GCP project id (BigQuery sandbox is fine)")
        s.add_argument("--dry-run", action="store_true", help="only estimate bytes scanned")
        if name == "track-m":
            s.add_argument("--start", required=True, help="UTC window start, e.g. 2026-09-20T00:00:00")
            s.add_argument("--hours", type=int, default=6)
    args = p.parse_args()
    if args.cmd == "auth":
        bq = client(args.project)
        print("[auth] ok; datasets visible:", [d.dataset_id for d in bq.list_datasets()])
    elif args.cmd == "track-r":
        cmd_track_r(args)
    else:
        cmd_track_m(args)


if __name__ == "__main__":
    sys.exit(main())
