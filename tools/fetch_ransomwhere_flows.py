"""One-time fetch: real ransomware money flows from Ransomwhere seeds (BigQuery public Bitcoin data).

hop 0: the ransom payments listed by Ransomwhere (victim -> ransom address)
hop 1: transactions spending from the ransom addresses (operator-initiated: aggregation / cash-out / layering)
hop 2: transactions spending outputs of hop-1 transactions (continued layering, or a counterparty such as an
       exchange sweeping a deposit - disambiguated later, see argusds/build_track_m.py)

Every query is dry-run first; the run aborts if the cumulative scan would exceed --budget-gb.

    python tools/fetch_ransomwhere_flows.py --project <gcp-project> [--start 2020-07-01 --end 2022-06-30] [--dry-run]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_bigquery import PUBLIC_TX, RAW, TX_SELECT, WORK_DATASET, client  # noqa: E402

OUT = RAW / "ransomwhere_flows"


class Budget:
    def __init__(self, bq, limit_gb: float, dry: bool):
        self.bq, self.limit, self.used, self.dry = bq, limit_gb, 0.0, dry

    def run(self, sql: str, label: str) -> pd.DataFrame | None:
        est = self.bq.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)).total_bytes_processed / 1e9
        print(f"[{label}] scans {est:.1f} GB (cumulative {self.used + est:.1f} / budget {self.limit:.0f})", flush=True)
        if self.used + est > self.limit:
            sys.exit(f"[{label}] budget exceeded; aborting")
        self.used += est
        if self.dry:
            return None
        return self.bq.query(sql).to_dataframe(create_bqstorage_client=False)


def upload(bq, project: str, name: str, df: pd.DataFrame) -> str:
    tid = f"{project}.{WORK_DATASET}.{name}"
    bq.create_dataset(f"{project}.{WORK_DATASET}", exists_ok=True)
    bq.load_table_from_dataframe(df, tid, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")).result()
    return tid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--start", default="2020-07-01")
    ap.add_argument("--end", default="2022-06-30")
    ap.add_argument("--min-usd", type=float, default=50.0)
    ap.add_argument("--budget-gb", type=float, default=450.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    bq = client(a.project)
    B = Budget(bq, a.budget_gb, a.dry_run)
    OUT.mkdir(parents=True, exist_ok=True)

    rw = pd.read_parquet(RAW / "labels" / "ransomwhere_payments.parquet")
    rw = rw[(rw["date"] >= a.start) & (rw["date"] <= a.end) & (rw["usd"] >= a.min_usd)]
    print(f"[seeds] {len(rw):,} payments to {rw['address'].nunique():,} ransom addresses, {rw['family'].nunique()} families")
    m0 = pd.Timestamp(a.start).strftime("%Y-%m-01")
    m1 = (pd.Timestamp(a.end) + pd.DateOffset(months=3)).strftime("%Y-%m-01")     # money moves after payment
    months = f"block_timestamp_month BETWEEN '{m0}' AND '{m1}'"

    addr_t = upload(bq, a.project, "rw_addresses", rw[["address", "family"]].drop_duplicates("address"))
    hop1 = B.run(f"""
        SELECT DISTINCT t.hash AS txh, r.family
        FROM `{PUBLIC_TX}` t, UNNEST(t.inputs) i, UNNEST(i.addresses) ad JOIN `{addr_t}` r ON ad = r.address
        WHERE t.{months}""", "hop1: spends from ransom addresses")
    if hop1 is not None:
        hop1 = hop1.drop_duplicates("txh")
        print(f"[hop1] {len(hop1):,} operator transactions")
        h1_t = upload(bq, a.project, "rw_hop1", hop1[["txh"]])
    else:
        h1_t = upload(bq, a.project, "rw_hop1", pd.DataFrame({"txh": ["0" * 64]}))
    hop2 = B.run(f"""
        SELECT DISTINCT t.hash AS txh
        FROM `{PUBLIC_TX}` t, UNNEST(t.inputs) i JOIN `{h1_t}` h ON i.spent_transaction_hash = h.txh
        WHERE t.{months}""", "hop2: spends of hop-1 outputs")
    if hop2 is not None:
        print(f"[hop2] {len(hop2):,} transactions")

    hashes = pd.concat([
        rw[["txid", "family"]].rename(columns={"txid": "txh"}).assign(hop=0),
        (hop1.assign(hop=1) if hop1 is not None else pd.DataFrame(columns=["txh", "family", "hop"])),
        (hop2.assign(hop=2, family=None) if hop2 is not None else pd.DataFrame(columns=["txh", "family", "hop"])),
    ]).sort_values("hop").drop_duplicates("txh")
    all_t = upload(bq, a.project, "rw_all", hashes[["txh"]] if len(hashes) else pd.DataFrame({"txh": ["0" * 64]}))
    txs = B.run(f"""
        SELECT {TX_SELECT}
        FROM `{PUBLIC_TX}` t JOIN `{all_t}` h ON t.hash = h.txh
        WHERE t.{months}""", "details for all flow transactions")
    if txs is not None:
        txs = txs.merge(hashes.rename(columns={"txh": "hash"}), on="hash", how="left")
        txs.to_parquet(OUT / "ransomwhere_flow_txs.parquet", index=False)
        rw.to_parquet(OUT / "seeds_used.parquet", index=False)
        (OUT / "meta.json").write_text(json.dumps({"start": a.start, "end": a.end, "min_usd": a.min_usd,
                                                  "gb_scanned": round(B.used, 1), "rows": len(txs),
                                                  "by_hop": txs["hop"].value_counts().to_dict()}, indent=2, default=str))
        print(f"[saved] {len(txs):,} transactions, {B.used:.0f} GB scanned")
    else:
        print(f"[dry-run] total estimate {B.used:.0f} GB")


if __name__ == "__main__":
    main()
