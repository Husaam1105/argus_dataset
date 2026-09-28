"""Track R: real Elliptic transactions (2016-01 .. 2017-10) with Elliptic's real labels + simulated network layer.

Ledger layer : real (BigQuery crypto_bitcoin, located via the public 99.5% deanonymisation mapping)
Labels       : real (Elliptic: illicit / licit / unknown, time steps 1-49)
Network layer: simulated (argusds.network), label-blind: we do not know real operators' OPSEC, so illicit and
               licit transactions get exactly the same network model. No network signal is planted here.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import cluster, config as C, export, network
from .normalize import normalize


def elliptic_labels() -> pd.DataFrame:
    cls = pd.read_csv(C.ELLIPTIC_DIR / "elliptic_txs_classes.csv")
    ts = pd.read_csv(C.ELLIPTIC_DIR / "elliptic_txs_features.csv", header=None, usecols=[0, 1])
    ts.columns = ["txId", "elliptic_timestep"]
    m = pd.read_csv(C.ELLIPTIC_MAP)
    d = cls.merge(ts, on="txId").merge(m, on="txId", how="inner")
    d["elliptic_class"] = d["class"].astype(str).map({"1": "illicit", "2": "licit", "unknown": "unknown"})
    return d.rename(columns={"txId": "elliptic_txid", "transaction": "txid"})[
        ["txid", "elliptic_txid", "elliptic_class", "elliptic_timestep"]]


def build(space, seed: np.random.SeedSequence, out_obs, out_truth, coverage=None, formats=None) -> dict:
    raw = pd.read_parquet(C.RAW / "track_r" / "elliptic_real_txs.parquet")
    ledger, excluded = normalize(raw)
    labels = elliptic_labels()
    ledger = ledger.merge(labels, on="txid", how="left")
    excluded = excluded.merge(labels, on="txid", how="left")
    missing = labels[~labels["txid"].isin(raw["hash"])]

    entity, cj = cluster.cioh_clusters(ledger)
    ledger["entity"] = entity.to_numpy()
    svc = cluster.service_flags(ledger, entity)
    ents = pd.DataFrame({"entity": np.unique(entity)})
    ents["role"] = np.where(ents["entity"].map(svc.groupby(entity).max()), "service", "user")
    ents["path"] = None

    net, net_truth, ip_truth, stats, ann = network.simulate(ledger, ents, space, seed, coverage=coverage,
                                                              path_mix=C.NETWORK["user_broadcast_path_mix_2016"])
    obs = export.observable_table(ledger, net)
    files = export.write_all(obs, out_obs, formats=formats)
    files.update(export.write_announcements(ann, out_obs))

    tx_labels = ledger[["txid", "elliptic_txid", "elliptic_class", "elliptic_timestep", "block_height", "block_time",
                        "n_in", "n_out", "feerate", "dropped_opreturn"]].copy()
    tx_labels["is_illicit"] = tx_labels["elliptic_class"].map({"illicit": True, "licit": False})  # unknown -> NA
    tx_labels["label_source"] = "elliptic_2019"
    tx_labels["coinjoin_rule"] = cj.to_numpy()
    tx_labels["entity_proxy"] = ledger["entity"].to_numpy()
    tx_labels["entity_proxy_is_service"] = svc.to_numpy()
    tx_labels["input_outpoints"] = [[f"{t}:{v}" for t, v in zip(a, b)]
                                    for a, b in zip(ledger["in_prev_txid"], ledger["in_prev_vout"])]
    tx_labels = tx_labels.merge(net_truth, on="txid")
    # Temporal split as in Weber et al. 2019 (train 1-34, test 35-49). Time steps are ~13.4 days apart and the
    # tx graph is a set of per-time-step components, so a time split is also graph-disjoint.
    tx_labels["split"] = np.where(tx_labels["elliptic_timestep"] <= 34, "train", "test")

    out_truth.mkdir(parents=True, exist_ok=True)
    tx_labels.to_parquet(out_truth / "tx_labels.parquet", index=False)
    ip_truth.to_parquet(out_truth / "ip_truth.parquet", index=False)
    ents.assign(note="CIOH proxy cluster, not ground truth").to_parquet(out_truth / "entities_proxy.parquet", index=False)
    excluded.to_parquet(out_truth / "excluded.parquet", index=False)
    missing.to_parquet(out_truth / "unfetched.parquet", index=False)

    stats.update({
        "track": "R",
        "elliptic_mapped": int(len(labels)), "fetched": int(len(raw)), "not_found_in_bigquery": int(len(missing)),
        "kept": int(len(ledger)), "excluded": int(len(excluded)),
        "excluded_by_reason": excluded["reason"].value_counts().to_dict(),
        "excluded_by_class": excluded["elliptic_class"].value_counts().to_dict(),
        "kept_by_class": ledger["elliptic_class"].value_counts().to_dict(),
        "coinjoin_rule_hits": cj.value_counts().to_dict(),
        "time_range": [str(ledger["block_time"].min()), str(ledger["block_time"].max())],
        "files": files,
    })
    (out_truth / "stats_realized.json").write_text(json.dumps(stats, indent=2, default=str))
    return stats
