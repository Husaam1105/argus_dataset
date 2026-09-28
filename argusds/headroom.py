"""Dual-layer headroom: does network-layer information help entity resolution, and where?

Evaluated on Track M injected entities with exact ownership truth. Linkers:
  cioh         common-input ownership over all observable transactions (CoinJoin-shaped txs skipped)
  ip_hour      addresses whose transactions were first seen from the same src_ip in the same hour
  fingerprint  Biryukov-style: per (address, day) the set of peers in the first announcements of its transactions;
               two addresses are linked when their fingerprints overlap strongly (>= MIN_SHARED peers, Jaccard >= J)
  dual         cioh  U  fingerprint
  *_hubaware   fingerprint links through shared broadcasters (matching > HUB CIOH clusters) dropped
Scores are pairwise precision / recall / F1, overall and restricted to entities of each broadcast path.

    python -m argusds.headroom --release data/releases/argus-dataset-vX [--out file.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .cluster import UnionFind, coinjoin_rule

MIN_SHARED, J = 3, 0.3
HUB = 10         # a fingerprint matching more than HUB distinct CIOH clusters is treated as a shared broadcaster


def pair_scores(true: pd.Series, pred: pd.Series) -> dict:
    def pairs(s):
        g = s.groupby(s).size()
        return int((g * (g - 1) // 2).sum())
    tp = pairs(pd.Series(list(zip(true, pred)), index=true.index).astype(str))
    pp, tt = pairs(pred.astype(str)), pairs(true.astype(str))
    p, r = tp / max(pp, 1), tp / max(tt, 1)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-12), 4), "true_pairs": tt}


def evaluate(rel: Path) -> dict:
    tm = rel / "observable" / "track_m"
    obs = pd.read_parquet(tm / "transactions.parquet")
    ann = pd.read_parquet(tm / "announcements.parquet", columns=["txid", "src_ip", "rank", "timestamp"])
    lab = pd.read_parquet(rel / "truth" / "track_m" / "tx_labels.parquet").set_index("txid")
    wal = pd.read_parquet(rel / "truth" / "track_m" / "wallets_injected.parquet").set_index("address")["entity"]
    ents = pd.read_parquet(rel / "truth" / "track_m" / "entities.parquet").set_index("entity")
    exact = set(ents.index[ents["truth_quality"] == "exact"])

    inj = obs["txid"].map(lab["origin"]).to_numpy() == "injected"
    rows = [(a, t) for t, ins in zip(obs["txid"][inj], obs["input_addresses"][inj]) for a in ins
            if a in wal.index and wal[a] in exact]
    A = pd.DataFrame(rows, columns=["addr", "txid"]).drop_duplicates()
    addrs = A["addr"].unique()
    true = pd.Series(wal.loc[addrs].to_numpy(), index=addrs)
    path_of_entity = lab.loc[lab["origin"] == "injected"].groupby("entity")["broadcast_path"].first()

    # CIOH
    uf = UnionFind()
    for ins, outs in zip(obs["input_addresses"], obs["output_sats"]):
        for a in ins:
            uf.find(a)
        if coinjoin_rule(list(outs), len(ins)) is None:
            for x in ins[1:]:
                uf.union(ins[0], x)
    cioh = pd.Series({a: uf.find(a) for a in addrs})

    # same first-seen IP within the same hour
    fs = obs.set_index("txid")[["src_ip", "timestamp"]]
    Af = A.join(fs, on="txid")
    uf2 = UnionFind()
    for a in addrs:
        uf2.find(a)
    for _, g in Af.groupby(["src_ip", Af["timestamp"].dt.floor("1h")]):
        l = g["addr"].unique()
        for x in l[1:]:
            uf2.union(l[0], x)
    ip_hour = pd.Series({a: uf2.find(a) for a in addrs})

    # fingerprint = announcing peers of the address's transactions, per UTC day
    ann = ann[ann["txid"].isin(set(A["txid"]))]
    ann["day"] = ann["timestamp"].dt.floor("1D")
    tx_fp = ann.groupby("txid")["src_ip"].agg(set)
    tx_day = ann.groupby("txid")["day"].first()
    fp = defaultdict(set)
    for a, t in zip(A["addr"], A["txid"]):
        if t in tx_fp.index:
            fp[(a, tx_day[t])] |= tx_fp[t]
    inv = defaultdict(list)
    for key, s in fp.items():
        for ip in s:
            inv[(key[1], ip)].append(key)
    uf3 = UnionFind()
    for a in addrs:
        uf3.find(a)
    shared = defaultdict(int)
    for (day, ip), keys in inv.items():
        if len(keys) > 200:                       # a peer seen with hundreds of wallets that day carries no identity
            continue
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if keys[i][0] != keys[j][0]:
                    shared[(keys[i], keys[j])] += 1
    links = [(k1, k2) for (k1, k2), c in shared.items() if c >= MIN_SHARED and c / len(fp[k1] | fp[k2]) >= J]
    for k1, k2 in links:
        uf3.union(k1[0], k2[0])
    fingerprint = pd.Series({a: uf3.find(a) for a in addrs})

    # hub-aware: a fingerprint that matches many different CIOH clusters is a shared broadcaster
    # (light-wallet server, proxy, exchange); its links carry no ownership information and are dropped.
    partners = defaultdict(set)
    for k1, k2 in links:
        partners[k1].add(cioh[k2[0]])
        partners[k2].add(cioh[k1[0]])
    uf5 = UnionFind()
    for a in addrs:
        uf5.find(a)
    for k1, k2 in links:
        if len(partners[k1]) <= HUB and len(partners[k2]) <= HUB:
            uf5.union(k1[0], k2[0])
    fp_hub = pd.Series({a: uf5.find(a) for a in addrs})

    def union_of(*parts):
        u = UnionFind()
        for a in addrs:
            for i, prt in enumerate(parts):
                u.union(f"{i}:{prt[a]}", a)
        return pd.Series({a: u.find(a) for a in addrs})
    dual = union_of(cioh, fingerprint)
    dual_hub = union_of(cioh, fp_hub)

    out = {"addresses": int(len(addrs)), "entities": int(true.nunique()), "overall": {}, "by_path": {}}
    for name, pred in [("cioh", cioh), ("ip_hour", ip_hour), ("fingerprint", fingerprint), ("dual", dual),
                       ("fingerprint_hubaware", fp_hub), ("dual_hubaware", dual_hub)]:
        out["overall"][name] = pair_scores(true, pred)
    paths = true.map(path_of_entity)
    for p in sorted(paths.dropna().unique()):
        m = (paths == p).to_numpy()
        if pair_scores(true[m], true[m])["true_pairs"] < 10:
            continue
        out["by_path"][p] = {name: pair_scores(true[m], pred[addrs][m])["f1"]
                             for name, pred in [("cioh", cioh), ("dual", dual), ("dual_hubaware", dual_hub)]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = evaluate(Path(a.release))
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
