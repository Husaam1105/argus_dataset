"""Cut a sub-window out of a downloaded mainnet window (no new BigQuery scan).

    python tools/slice_window.py --src mainnet_20260923T00_48h.parquet --start 2026-09-24T00:00:00 --hours 24
"""
import argparse
import json
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "track_m"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--hours", type=int, required=True)
    a = ap.parse_args()
    df = pd.read_parquet(RAW / a.src)
    start = pd.Timestamp(a.start, tz="UTC")
    end = start + pd.Timedelta(hours=a.hours)
    part = df[(df["block_timestamp"] >= start) & (df["block_timestamp"] < end)]
    name = f"mainnet_{start.strftime('%Y%m%dT%H')}_{a.hours}h.parquet"
    part.to_parquet(RAW / name, index=False)
    (RAW / name).with_suffix(".meta.json").write_text(json.dumps(
        {"sliced_from": a.src, "start": str(start), "end": str(end), "rows": int(len(part)),
         "blocks": int(part["block_number"].nunique())}, indent=2))
    print(f"[slice] {name}: {len(part):,} txs in {part['block_number'].nunique()} blocks")


if __name__ == "__main__":
    main()
