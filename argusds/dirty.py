"""Dirty / drifted variants of a release's observable files, for testing the ingestion layer.

    python -m argusds.dirty --release data/releases/argus-dataset-vX [--track track_m] [--rate 0.005]

Writes observable/<track>/dirty/ with:
  transactions_dirty.{csv,jsonl,xml}  the clean table with ~rate of records faulted (exact list in the manifest)
  transactions_drift.csv              same clean data under schema drift: renamed columns, epoch-ms timestamps,
                                      integer-satoshi amounts, ';' array separator
and truth/<track>/faults.parquet: file, record_index (0-based data record), txid, fault_type.
Fault types: bad_checksum, negative_amount, nan_amount, length_mismatch, bad_timestamp, conflicting_duplicate_txid,
exact_duplicate, invalid_ip, invalid_port, missing_field, conservation_violation, truncated_record.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C, export

FAULTS = ["bad_checksum", "negative_amount", "nan_amount", "length_mismatch", "bad_timestamp",
          "conflicting_duplicate_txid", "exact_duplicate", "invalid_ip", "invalid_port", "missing_field",
          "conservation_violation", "truncated_record"]


def corrupt(rec: dict, fault: str, rng: np.random.Generator) -> dict:
    r = json.loads(json.dumps(rec))
    if fault == "bad_checksum":
        a = r["output_addresses"][0]
        i = len(a) - 1 - int(rng.integers(0, 4))
        alphabet = "qpzry9x8gf2tvdw0s3jn54khce6mua7l" if a.startswith("bc1") else "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
        r["output_addresses"][0] = a[:i] + next(c for c in alphabet if c != a[i]) + a[i + 1:]
    elif fault == "negative_amount":
        r["output_amounts"][0] = "-" + r["output_amounts"][0]
    elif fault == "nan_amount":
        r["input_amounts"][0] = "NaN"
    elif fault == "length_mismatch":
        r["output_amounts"] = r["output_amounts"][:-1] if len(r["output_amounts"]) > 1 else r["output_amounts"] + ["0.00010000"]
    elif fault == "bad_timestamp":
        r["timestamp"] = str(rng.choice(["2026-13-45T25:61:00.000Z", "not-a-time", "1970-01-01T00:00:00.000Z", ""]))
    elif fault == "invalid_ip":
        r["src_ip"] = str(rng.choice(["300.12.1.7", "240.1.2", "::zz", "localhost"]))
    elif fault == "invalid_port":
        r["src_port"] = int(rng.choice([0, 70000, -1]))
    elif fault == "missing_field":
        r[str(rng.choice(["src_ip", "fee", "script_type", "geo_country"]))] = ""
    elif fault == "conservation_violation":
        r["fee"] = "0.50000000"
    elif fault == "conflicting_duplicate_txid":
        r["output_amounts"][0] = export.btc(int(round(float(r["output_amounts"][0]) * 1e8)) + 1234)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--track", default="track_m")
    ap.add_argument("--rate", type=float, default=0.005)
    a = ap.parse_args()
    rel = Path(a.release)
    df = pd.read_parquet(rel / "observable" / a.track / "transactions.parquet")
    rng = np.random.default_rng(np.random.SeedSequence([C.MASTER_SEED, 7]))
    recs = list(export._records(df))
    n = len(recs)
    idx = np.sort(rng.choice(n, size=max(len(FAULTS), int(n * a.rate)), replace=False))
    kinds = [FAULTS[i % len(FAULTS)] for i in range(len(idx))]
    rng.shuffle(kinds)
    fault_at = dict(zip(idx.tolist(), kinds))

    out = rel / "observable" / a.track / "dirty"
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    # build the dirty record stream once; duplicates are inserted right after their source record
    stream = []                                                     # (record or raw-text marker, fault or None)
    for i, rec in enumerate(recs):
        f = fault_at.get(i)
        if f in ("exact_duplicate", "conflicting_duplicate_txid"):
            stream.append((rec, None))
            stream.append((rec if f == "exact_duplicate" else corrupt(rec, f, rng), f))
        elif f == "truncated_record":
            stream.append((rec, f))
        elif f:
            stream.append((corrupt(rec, f, rng), f))
        else:
            stream.append((rec, None))

    def text_of(rec, fmt):
        if fmt == "jsonl":
            return json.dumps(rec, separators=(",", ":"))
        if fmt == "csv":
            import io
            s = io.StringIO()
            csv.writer(s).writerow([("|".join(map(str, rec[k])) if isinstance(rec[k], list) else rec[k]) for k in export.FIELDS])
            return s.getvalue().rstrip("\r\n")
        ins = "".join(f'<input address="{x}" amount="{v}"/>' for x, v in zip(rec["input_addresses"], rec["input_amounts"]))
        outs = "".join(f'<output address="{x}" amount="{v}"/>' for x, v in zip(rec["output_addresses"], rec["output_amounts"]))
        return (f'  <tx txid="{rec["txid"]}" timestamp="{rec["timestamp"]}" fee="{rec["fee"]}" script_type="{rec["script_type"]}">'
                f'<network src_ip="{rec["src_ip"]}" src_port="{rec["src_port"]}" dst_ip="{rec["dst_ip"]}" dst_port="{rec["dst_port"]}" '
                f'geo_country="{rec["geo_country"]}" asn="{rec["asn"]}"/><inputs>{ins}</inputs><outputs>{outs}</outputs></tx>')

    for fmt, head, tail in [("csv", ",".join(export.FIELDS), None), ("jsonl", None, None),
                            ("xml", '<?xml version="1.0" encoding="UTF-8"?>\n<transactions>', "</transactions>")]:
        name = f"transactions_dirty.{fmt}"
        with (out / name).open("w", encoding="utf-8", newline="") as fh:
            if head:
                fh.write(head + "\n")
            for j, (rec, f) in enumerate(stream):
                t = text_of(rec, fmt)
                if f == "truncated_record":
                    t = t[: max(10, len(t) // 2)]
                fh.write(t + "\n")
                if f:
                    manifest.append((name, j, rec["txid"], f))
            if tail:
                fh.write(tail + "\n")

    # schema drift: same clean data, different conventions
    d = pd.DataFrame(recs)
    drift = pd.DataFrame({
        "tx_hash": d["txid"], "time_ms": pd.to_datetime(d["timestamp"]).astype("int64") // 10**6,
        "source_ip": d["src_ip"], "source_port": d["src_port"], "destination_ip": d["dst_ip"], "destination_port": d["dst_port"],
        "inputs": d["input_addresses"].map(";".join), "input_values_sat": df["input_sats"].map(lambda v: ";".join(map(str, v))),
        "outputs": d["output_addresses"].map(";".join), "output_values_sat": df["output_sats"].map(lambda v: ";".join(map(str, v))),
        "fee_sat": df["fee_sats"], "scriptType": d["script_type"], "country": d["geo_country"], "as_number": d["asn"],
    })
    drift.to_csv(out / "transactions_drift.csv", index=False)
    (out / "README.txt").write_text(
        "transactions_dirty.*: faulted copies of the clean data (see truth/<track>/faults.parquet for exact locations).\n"
        "transactions_drift.csv: clean data under schema drift. Column map: tx_hash=txid, time_ms=timestamp (epoch ms), "
        "source_*/destination_*=src_*/dst_*, inputs/outputs=';'-separated addresses, *_values_sat=integer satoshis, "
        "fee_sat=fee in satoshis, scriptType=script_type, country=geo_country, as_number=asn.\n", encoding="utf-8")
    fm = pd.DataFrame(manifest, columns=["file", "record_index", "txid", "fault_type"])
    fm.to_parquet(rel / "truth" / a.track / "faults.parquet", index=False)
    print(f"[dirty] {a.track}: {len(idx)} faulted source records ({len(idx) / n:.2%}), {len(fm)} manifest rows;",
          fm.drop_duplicates("record_index")["fault_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
