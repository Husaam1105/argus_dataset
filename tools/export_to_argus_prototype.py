"""Export one track of an ARGUS dataset release into the ARGUS prototype's data-dir layout.

    python tools/export_to_argus_prototype.py --release data/releases/argus-dataset-v1.0.1 --track track_m \
        --out ../ARGUS_prototype/data_argusds_m

Produces <out>/raw/transactions.csv (the prototype's CSV convention: '|' array separator, BTC amounts, ISO
timestamps — identical to our export), <out>/ground_truth/{entities,seeds,patterns}.parquet in the prototype's
schemas (docs/contracts.md), and <out>/geoip/*.mmdb (our shadow GeoIP databases).

Ground-truth mapping (prototype entity types: licit / ransomware / darknet / mixer / exchange):
  ransomware_operator, ransomware_operator_real -> ransomware
  launderer                                     -> mixer
  elliptic_replay_actor                         -> darknet   (Elliptic's illicit class is a mix of scams, markets,
                                                              ransomware; 'darknet' is the closest generic bucket)
  exchange                                      -> exchange
  every other injected licit entity             -> licit
Only wallets whose owner is known are exported (truth quality exact / inherited_cioh / traced_heuristic);
real-background wallets and replay counterparties have unknown owners and are left out, so the prototype's
pairwise ER evaluation (an inner join on wallet_id) scores exactly the wallets with real ground truth.
Patterns: our peeling_chain -> 'peeling', coinjoin_whirlpool -> 'coinjoin'. Hard negatives (exchange peeling
chains, batch payouts) are not exported as patterns: flagging them counts as a false positive, as it should.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

TYPE_MAP = {"ransomware_operator": "ransomware", "ransomware_operator_real": "ransomware", "launderer": "mixer",
            "elliptic_replay_actor": "darknet", "exchange": "exchange"}
KNOWN = {"exact", "inherited_cioh", "traced_heuristic"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--track", default="track_m")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    obs, truth = a.release / "observable" / a.track, a.release / "truth" / a.track
    for sub in ("raw", "ground_truth", "geoip"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(obs / "transactions.csv", a.out / "raw" / "transactions.csv")
    for f in (a.release / "observable" / "geoip").glob("*.mmdb"):
        shutil.copy(f, a.out / "geoip" / f.name)

    if a.track == "track_m":
        ents = pd.read_parquet(truth / "entities.parquet")
        wal = pd.read_parquet(truth / "wallets_injected.parquet")
        ents = ents[ents["truth_quality"].isin(KNOWN)]
        ents["entity_type_proto"] = ents["entity_type"].map(TYPE_MAP).fillna("licit")
        ents.loc[ents["is_illicit"] == True, "entity_type_proto"] = \
            ents.loc[ents["is_illicit"] == True, "entity_type"].map(TYPE_MAP).fillna("darknet")   # noqa: E712
        gt = wal.merge(ents[["entity", "entity_type_proto"]], on="entity")
        gt = pd.DataFrame({"wallet_id": gt["address"], "entity_id": "E" + gt["entity"].astype(str),
                           "entity_type": gt["entity_type_proto"]})
        # a wallet must appear in the observable data to be scorable
        seen = pd.read_parquet(obs / "transactions.parquet", columns=["input_addresses", "output_addresses"])
        seen_addrs = set(pd.Series([x for col in ("input_addresses", "output_addresses") for l in seen[col] for x in l]))
        gt = gt[gt["wallet_id"].isin(seen_addrs)].reset_index(drop=True)
        seeds = pd.read_parquet(truth / "seeds.parquet").rename(columns={"address": "wallet_id"})[["wallet_id"]]
        pat = pd.read_parquet(truth / "patterns.parquet")
        pmap = {"peeling_chain": "peeling", "coinjoin_whirlpool": "coinjoin"}
        pat = pat[pat["type"].isin(pmap)].assign(type=lambda d: d["type"].map(pmap))
        pat = pat[["pattern_id", "type", "txids", "wallets"]].reset_index(drop=True)
    else:
        # Track R has real labels per transaction but no entity truth: seeds = inputs of a sample of illicit txs
        lab = pd.read_parquet(truth / "tx_labels.parquet", columns=["txid", "elliptic_class", "split"])
        o = pd.read_parquet(obs / "transactions.parquet", columns=["txid", "input_addresses"]).merge(lab, on="txid")
        ill = o[(o["elliptic_class"] == "illicit") & (o["split"] == "train")]
        seeds = pd.DataFrame({"wallet_id": sorted({x for l in ill["input_addresses"] for x in l})})
        seeds = seeds.sample(frac=0.1, random_state=0).reset_index(drop=True)
        gt = pd.DataFrame(columns=["wallet_id", "entity_id", "entity_type"])
        pat = pd.DataFrame(columns=["pattern_id", "type", "txids", "wallets"])
    gt.to_parquet(a.out / "ground_truth" / "entities.parquet", index=False)
    seeds.to_parquet(a.out / "ground_truth" / "seeds.parquet", index=False)
    pat.to_parquet(a.out / "ground_truth" / "patterns.parquet", index=False)
    print(f"[export] {a.track} -> {a.out}: entities={len(gt):,} wallets "
          f"({gt['entity_type'].value_counts().to_dict() if len(gt) else {}}), seeds={len(seeds):,}, patterns={len(pat)}")


if __name__ == "__main__":
    main()
