"""Score an ARGUS prototype run against the ARGUS dataset's ground truth.

    python tools/eval_argus_run.py --run ARGUS_prototype/data_argusds_m \
        --truth data/releases/argus-dataset-v1.0.1-6h/truth/track_m [--label A]

Reports (seed addresses excluded from every hit count — ARGUS is given the seeds, so flagging them is not a
discovery):
  alerts        size of the alert queue
  precision@k   share of the top-k alerts that are illicit wallets or illicit transactions
  recall        share of all illicit wallets + transactions that appear anywhere in the queue
  AP            average precision of the queue's ranking
  ER            pairwise precision / recall / F1 of artifacts/entities.parquet vs ground_truth/entities.parquet
                (inner join on wallet_id: only wallets with a known owner are scored)
  anomaly AUC   ROC-AUC of the anomaly head's score alone, illicit vs licit, over wallets with a known owner
                and all labelled transactions
Prints one JSON object; --label names the run in it.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def pairs(s: pd.Series) -> int:
    g = s.value_counts()
    return int((g * (g - 1) // 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--truth", required=True, type=Path)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    art, gtd = a.run / "artifacts", a.run / "ground_truth"

    seeds = set(pd.read_parquet(gtd / "seeds.parquet")["wallet_id"])
    wal = pd.read_parquet(a.truth / "wallets_injected.parquet")
    ents = pd.read_parquet(a.truth / "entities.parquet")
    lab = pd.read_parquet(a.truth / "tx_labels.parquet", columns=["txid", "is_illicit"])
    ill_w = set(wal.loc[wal["entity"].isin(set(ents.loc[ents["is_illicit"] == True, "entity"])), "address"]) - seeds  # noqa: E712
    ill_tx = set(lab.loc[lab["is_illicit"] == True, "txid"])                                                       # noqa: E712
    targets = ill_w | ill_tx
    out = {"label": a.label}

    alerts = pd.DataFrame(json.load(open(art / "alerts.json", encoding="utf-8")))
    alerts = alerts[~alerts["node_id"].isin(seeds)].sort_values("final_score", ascending=False).reset_index(drop=True)
    hit = alerts["node_id"].isin(targets)
    out["alerts"] = int(len(alerts))
    for k in (10, 50, 100, 500):
        out[f"precision@{k}"] = round(float(hit.head(k).mean()), 3) if len(alerts) else None
    out["recall_in_queue"] = round(float(hit.sum() / len(targets)), 4)
    out["AP"] = round(float(average_precision_score(hit, alerts["final_score"])), 3) if hit.any() else 0.0

    pred = pd.read_parquet(art / "entities.parquet")[["wallet_id", "entity_id"]]
    gt = pd.read_parquet(gtd / "entities.parquet")[["wallet_id", "entity_id"]]
    m = pred.merge(gt, on="wallet_id", suffixes=("_pred", "_true"))
    tp = pairs(m["entity_id_pred"] + "|" + m["entity_id_true"])
    pp, tt = pairs(m["entity_id_pred"]), pairs(m["entity_id_true"])
    p, r = (tp / pp if pp else 1.0), (tp / tt if tt else 1.0)
    out["ER_precision"], out["ER_recall"] = round(p, 3), round(r, 3)
    out["ER_F1"] = round(2 * p * r / (p + r), 3) if p + r else 0.0

    an = pd.read_parquet(art / "scores_anomaly.parquet")
    gtw = pd.read_parquet(gtd / "entities.parquet")
    lab_w = pd.Series(gtw["entity_type"].isin(["ransomware", "darknet", "mixer"]).to_numpy(), index=gtw["wallet_id"])
    lab_t = pd.Series(lab["is_illicit"].fillna(False).astype(bool).to_numpy(), index=lab["txid"])
    y = pd.concat([lab_w, lab_t[~lab_t.index.isin(lab_w.index)]])
    s = an.set_index("node_id")["score"]
    common = y.index.intersection(s.index)
    yy = y.loc[common]
    if yy.nunique() == 2:
        out["anomaly_AUC"] = round(float(roc_auc_score(yy, s.loc[common])), 3)
        out["anomaly_score_p50_licit"] = round(float(s.loc[common][~yy.to_numpy()].median()), 4)
        out["anomaly_score_p50_illicit"] = round(float(s.loc[common][yy.to_numpy()].median()), 4)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
