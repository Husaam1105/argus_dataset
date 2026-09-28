"""Observable-layer exporters. Every format carries exactly the brief's fields and decodes to the same table.

Fields: txid, timestamp, src_ip, src_port, dst_ip, dst_port, input_addresses, input_amounts,
        output_addresses, output_amounts, fee, script_type, geo_country, asn
Amounts and fee are BTC with 8 decimals (exact decimal strings from integer satoshis).
Timestamps are UTC ISO-8601 with millisecond precision.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import quoteattr

import pandas as pd

FIELDS = ["txid", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "input_addresses", "input_amounts",
          "output_addresses", "output_amounts", "fee", "script_type", "geo_country", "asn"]


def btc(sats: int) -> str:
    return f"{sats // 100_000_000}.{sats % 100_000_000:08d}"


def observable_table(ledger: pd.DataFrame, net: pd.DataFrame) -> pd.DataFrame:
    """Join ledger + network fields into the canonical observable table (integer satoshis, list columns)."""
    df = ledger[["txid", "in_addr", "in_sats", "out_addr", "out_sats", "fee_sats", "script_type"]].merge(net, on="txid")
    df = df.rename(columns={"in_addr": "input_addresses", "in_sats": "input_sats", "out_addr": "output_addresses",
                            "out_sats": "output_sats"})
    df = df.sort_values(["timestamp", "txid"]).reset_index(drop=True)
    return df[["txid", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "input_addresses", "input_sats",
               "output_addresses", "output_sats", "fee_sats", "script_type", "geo_country", "asn"]]


def _records(df: pd.DataFrame):
    ts = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
    for r, t in zip(df.itertuples(index=False), ts):
        yield dict(
            txid=r.txid, timestamp=t, src_ip=r.src_ip, src_port=int(r.src_port), dst_ip=r.dst_ip,
            dst_port=int(r.dst_port), input_addresses=list(r.input_addresses),
            input_amounts=[btc(int(v)) for v in r.input_sats], output_addresses=list(r.output_addresses),
            output_amounts=[btc(int(v)) for v in r.output_sats], fee=btc(int(r.fee_sats)),
            script_type=r.script_type, geo_country=r.geo_country, asn=int(r.asn),
        )


def _json_obj(rec: dict) -> str:
    # Amounts are emitted as JSON numbers with exactly 8 decimals (no float round-off).
    parts = []
    for k in FIELDS:
        v = rec[k]
        if k in ("input_amounts", "output_amounts"):
            parts.append(f'"{k}":[{",".join(v)}]')
        elif k == "fee":
            parts.append(f'"{k}":{v}')
        else:
            parts.append(f'"{k}":{json.dumps(v)}')
    return "{" + ",".join(parts) + "}"


def write_announcements(ann: pd.DataFrame, out_dir: Path) -> dict:
    """Traffic log: first N announcements per tx (txid, timestamp, src/dst ip+port, geo, asn, rank)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ann.to_parquet(out_dir / "announcements.parquet", index=False)
    a = ann.copy()
    a["timestamp"] = a["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
    a.to_csv(out_dir / "announcements.csv", index=False)
    return {"announcements_parquet": "announcements.parquet", "announcements_csv": "announcements.csv"}


def write_all(df: pd.DataFrame, out_dir: Path, stem: str = "transactions", formats=None) -> dict:
    formats = set(formats or ["parquet", "csv", "jsonl", "json", "xml"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    p = out_dir / f"{stem}.parquet"
    df.to_parquet(p, index=False)
    paths["parquet"] = p
    if formats == {"parquet"}:
        return {k: str(v.name) for k, v in paths.items()}

    p = out_dir / f"{stem}.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for rec in _records(df):
            w.writerow([("|".join(rec[k]) if isinstance(rec[k], list) else rec[k]) for k in FIELDS])
    paths["csv"] = p

    p = out_dir / f"{stem}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for rec in _records(df):
            f.write(_json_obj(rec) + "\n")
    paths["jsonl"] = p

    p = out_dir / f"{stem}.json"
    with p.open("w", encoding="utf-8") as f:
        f.write("[\n")
        for i, rec in enumerate(_records(df)):
            f.write(("," if i else "") + _json_obj(rec) + "\n")
        f.write("]\n")
    paths["json"] = p

    p = out_dir / f"{stem}.xml"
    with p.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<transactions>\n')
        for rec in _records(df):
            f.write(f'  <tx txid="{rec["txid"]}" timestamp="{rec["timestamp"]}" fee="{rec["fee"]}" '
                    f'script_type="{rec["script_type"]}">\n')
            f.write(f'    <network src_ip="{rec["src_ip"]}" src_port="{rec["src_port"]}" dst_ip="{rec["dst_ip"]}" '
                    f'dst_port="{rec["dst_port"]}" geo_country={quoteattr(str(rec["geo_country"]))} asn="{rec["asn"]}"/>\n')
            f.write("    <inputs>\n")
            for a, v in zip(rec["input_addresses"], rec["input_amounts"]):
                f.write(f'      <input address="{a}" amount="{v}"/>\n')
            f.write("    </inputs>\n    <outputs>\n")
            for a, v in zip(rec["output_addresses"], rec["output_amounts"]):
                f.write(f'      <output address="{a}" amount="{v}"/>\n')
            f.write("    </outputs>\n  </tx>\n")
        f.write("</transactions>\n")
    paths["xml"] = p
    return {k: str(v.name) for k, v in paths.items()}


def read_back(path: Path) -> pd.DataFrame:
    """Reference reader: decode any exported format back to the canonical table (used by the equivalence test)."""
    from decimal import Decimal
    import defusedxml.ElementTree as ET  # safe against XXE / entity expansion on untrusted input

    def sats(x: str) -> int:
        return int(Decimal(x) * 100_000_000)

    suf = path.suffix
    rows = []
    if suf == ".csv":
        d = pd.read_csv(path, dtype=str, keep_default_na=False)
        for r in d.itertuples(index=False):
            rows.append((r.txid, r.timestamp, r.src_ip, int(r.src_port), r.dst_ip, int(r.dst_port),
                         tuple(r.input_addresses.split("|")), tuple(sats(v) for v in r.input_amounts.split("|")),
                         tuple(r.output_addresses.split("|")), tuple(sats(v) for v in r.output_amounts.split("|")),
                         sats(r.fee), r.script_type, r.geo_country, int(r.asn)))
    elif suf in (".jsonl", ".json"):
        text = path.read_text(encoding="utf-8")
        recs = json.loads(text, parse_float=Decimal) if suf == ".json" else \
            [json.loads(line, parse_float=Decimal) for line in text.splitlines() if line]
        for r in recs:
            rows.append((r["txid"], r["timestamp"], r["src_ip"], r["src_port"], r["dst_ip"], r["dst_port"],
                         tuple(r["input_addresses"]), tuple(int(v * 100_000_000) for v in r["input_amounts"]),
                         tuple(r["output_addresses"]), tuple(int(v * 100_000_000) for v in r["output_amounts"]),
                         int(r["fee"] * 100_000_000), r["script_type"], r["geo_country"], r["asn"]))
    elif suf == ".xml":
        for tx in ET.parse(path).getroot():
            n = tx.find("network")
            ins, outs = tx.find("inputs"), tx.find("outputs")
            rows.append((tx.get("txid"), tx.get("timestamp"), n.get("src_ip"), int(n.get("src_port")), n.get("dst_ip"),
                         int(n.get("dst_port")), tuple(i.get("address") for i in ins),
                         tuple(sats(i.get("amount")) for i in ins), tuple(o.get("address") for o in outs),
                         tuple(sats(o.get("amount")) for o in outs), sats(tx.get("fee")), tx.get("script_type"),
                         n.get("geo_country"), int(n.get("asn"))))
    elif suf == ".parquet":
        d = pd.read_parquet(path)
        ts = d["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
        for r, t in zip(d.itertuples(index=False), ts):
            rows.append((r.txid, t, r.src_ip, int(r.src_port), r.dst_ip, int(r.dst_port), tuple(r.input_addresses),
                         tuple(int(v) for v in r.input_sats), tuple(r.output_addresses),
                         tuple(int(v) for v in r.output_sats), int(r.fee_sats), r.script_type, r.geo_country, int(r.asn)))
    return pd.DataFrame(rows, columns=FIELDS)


def canonical_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for row in df.sort_values("txid").itertuples(index=False):
        h.update(repr(tuple(row)).encode())
    return h.hexdigest()
