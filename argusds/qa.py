"""Release QA: hard invariants, format equivalence, leakage & provenance probes, external validity vs Elliptic.

    python -m argusds.qa --release data/releases/argus-dataset-v0.2.0

Writes <release>/qa/qa_report.json and qa_report.md. Hard invariants failing -> exit code 1.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import sys
from pathlib import Path

import maxminddb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config as C, export
from .addresses import validate_address

V4 = ipaddress.ip_network(C.SHADOW_V4)
V6 = ipaddress.ip_network(C.SHADOW_V6)


# ------------------------------------------------------------------------------------------------ invariants
def invariants(rel: Path, track: str) -> dict:
    obs = pd.read_parquet(rel / "observable" / track / "transactions.parquet")
    lab = pd.read_parquet(rel / "truth" / track / "tx_labels.parquet")
    r = {}
    tin = obs["input_sats"].map(sum)
    tout = obs["output_sats"].map(sum)
    r["conservation_sum_in_eq_sum_out_plus_fee"] = bool((tin == tout + obs["fee_sats"]).all())
    r["no_negative_or_zero_amounts"] = bool(obs["input_sats"].map(min).gt(0).all() and obs["output_sats"].map(min).gt(0).all()
                                            and obs["fee_sats"].ge(0).all())
    r["txid_unique"] = bool(obs["txid"].is_unique)
    r["txid_format_64hex"] = bool(obs["txid"].str.fullmatch(r"[0-9a-f]{64}").all())
    r["arrays_index_aligned"] = bool((obs["input_addresses"].str.len() == obs["input_sats"].str.len()).all() and
                                     (obs["output_addresses"].str.len() == obs["output_sats"].str.len()).all())
    addrs = pd.unique(np.concatenate([np.concatenate(obs["input_addresses"].to_numpy()),
                                      np.concatenate(obs["output_addresses"].to_numpy())]))
    bad = [a for a in addrs if validate_address(a) is None]
    r["addresses_valid_checksum"] = len(bad) == 0
    r["addresses_checked"] = int(len(addrs))
    # script_type field == type of the largest output
    st = [validate_address(a[int(np.argmax(v))]) for a, v in zip(obs["output_addresses"], obs["output_sats"])]
    r["script_type_consistent"] = bool((pd.Series(st) == obs["script_type"].to_numpy()).all())
    # no double spend: every outpoint spent at most once
    ops = pd.Series(np.concatenate(lab["input_outpoints"].to_numpy()))
    r["no_double_spend"] = bool(ops.is_unique)
    # causality: in-dataset parent created before child
    tc = lab.set_index("txid")["t_create"]
    par = [(c, o.split(":")[0]) for c, outs in zip(lab["txid"], lab["input_outpoints"]) for o in outs]
    par = pd.DataFrame(par, columns=["child", "parent"])
    par = par[par["parent"].isin(tc.index)]
    r["causality_parent_created_before_child"] = bool((tc.loc[par["parent"]].to_numpy() <= tc.loc[par["child"]].to_numpy()).all())
    r["in_dataset_spend_edges"] = int(len(par))
    ts = obs.merge(lab[["txid", "t_create"]], on="txid")
    r["observed_after_creation_within_skew"] = bool(((ts["timestamp"] - ts["t_create"]).dt.total_seconds() > -0.5).all())
    # IPs: all inside shadow ranges and resolvable in the shadow GeoIP DBs, ASN/country fields consistent
    geo = rel / "observable" / "geoip"
    asn_db = maxminddb.open_database(str(geo / "ARGUS-Shadow-ASN.mmdb"))
    cc_db = maxminddb.open_database(str(geo / "ARGUS-Shadow-Country.mmdb"))
    ips = pd.unique(pd.concat([obs["src_ip"], obs["dst_ip"]]))
    inside = [ipaddress.ip_address(i) in (V4 if ":" not in i else V6) for i in ips]
    r["all_ips_in_shadow_ranges"] = bool(all(inside))
    sample = obs.sample(min(20000, len(obs)), random_state=0)
    ok = [(asn_db.get(i) or {}).get("autonomous_system_number") == a and
          ((cc_db.get(i) or {}).get("country") or {}).get("iso_code") == c
          for i, a, c in zip(sample["src_ip"], sample["asn"], sample["geo_country"])]
    r["geoip_resolves_and_matches_fields"] = bool(all(ok))
    r["ports_in_range"] = bool(obs["src_port"].between(1, 65535).all() and obs["dst_port"].between(1, 65535).all())
    r["rows"] = int(len(obs))
    r["_hard_fail"] = [k for k, v in r.items() if v is False]
    return r


def format_equivalence(rel: Path, track: str) -> dict:
    d = rel / "observable" / track
    hashes = {p.suffix[1:]: export.canonical_hash(export.read_back(p))
              for p in sorted(d.glob("transactions.*"))}
    return {"hashes": {k: v[:16] for k, v in hashes.items()}, "all_equal": len(set(hashes.values())) == 1}


# ------------------------------------------------------------------------------------------------ features
def ledger_features(obs: pd.DataFrame, usd_per_btc) -> pd.DataFrame:
    """Observable-only per-tx features (identical definitions on every track). `usd_per_btc`: scalar or per-row."""
    px = np.broadcast_to(np.asarray(usd_per_btc, float), (len(obs),))
    ins, outs = obs["input_sats"], obs["output_sats"]
    n_in, n_out = obs["input_addresses"].str.len(), obs["output_addresses"].str.len()
    tot = outs.map(sum).astype(float)
    approx_vsize = 10.5 + 68 * n_in + 31 * n_out                 # observable has no vsize; P2WPKH-equivalent size

    def btc_zeros(v):
        return max(len(str(x)) - len(str(x).rstrip("0")) for x in v)

    def fiat_round(v, p):
        u = np.asarray(v, float) / 1e8 * p
        return float(np.any(np.abs(u - np.round(u / 5) * 5) < 0.005 * np.maximum(u, 1)))

    return pd.DataFrame({
        "n_in": n_in.to_numpy(), "n_out": n_out.to_numpy(),
        "log_usd_total": np.log10(tot.to_numpy() / 1e8 * px + 1e-9),
        "log_feerate_approx": np.log10(obs["fee_sats"].to_numpy() / approx_vsize.to_numpy() + 1e-3),
        "max_out_share": (outs.map(max) / tot).to_numpy(),
        "equal_out_max": outs.map(lambda v: pd.Series(v).value_counts().iloc[0]).to_numpy(),
        "btc_trailing_zeros": outs.map(btc_zeros).to_numpy(),
        "fiat_round_output": [fiat_round(v, p) for v, p in zip(outs, px)],
        "n_in_types": obs["input_addresses"].map(lambda a: len({validate_address(x) for x in a})).to_numpy(),
        "share_p2wpkh_out": obs["output_addresses"].map(lambda a: np.mean([x.startswith("bc1q") and len(x) == 42 for x in a])).to_numpy(),
        "share_p2tr_out": obs["output_addresses"].map(lambda a: np.mean([x.startswith("bc1p") and len(x) == 62 for x in a])).to_numpy(),
        "share_legacy_out": obs["output_addresses"].map(lambda a: np.mean([x[0] in "13" for x in a])).to_numpy(),
        "addr_reuse_in_out": [len(set(a) & set(b)) for a, b in zip(obs["input_addresses"], obs["output_addresses"])],
    })


def graph_features(obs: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """In-dataset spend-graph degrees (computed from address/amount linkage in truth outpoints)."""
    txids = set(obs["txid"])
    parents = lab.set_index("txid")["input_outpoints"].map(lambda o: len({x.split(':')[0] for x in o} & txids))
    child_count = pd.Series([x.split(":")[0] for o in lab["input_outpoints"] for x in set(o)]).value_counts()
    return pd.DataFrame({"in_deg": obs["txid"].map(parents).fillna(0).to_numpy(),
                         "out_deg": obs["txid"].map(child_count).fillna(0).to_numpy()})


def cv_auc(X: pd.DataFrame, y: np.ndarray, seed=0, groups=None, min_groups: int = 5) -> tuple[float, list]:
    """Cross-validated AUC. With `groups` (the owning entity of each row), every entity lands entirely in one fold,
    so near-duplicate transactions of one actor (e.g. the hops of a peeling chain) cannot leak across folds."""
    from sklearn.model_selection import StratifiedGroupKFold
    if groups is not None and len(np.unique(groups[np.asarray(y, bool)])) < min_groups:
        return float("nan"), []                  # too few distinct entities for a stable grouped estimate
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed) if groups is not None else         StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    clf.fit(X, y)
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(clf, X, y, n_repeats=3, random_state=seed, scoring="roc_auc")
    imp = sorted(zip(X.columns, pi.importances_mean), key=lambda t: -t[1])[:6]
    return float(roc_auc_score(y, p)), [(k, round(float(v), 4)) for k, v in imp]


# ------------------------------------------------------------------------------------------------ probes
def leakage_probe(rel: Path) -> dict:
    """Raw-string artifacts only (txid bytes, address string lengths, ports, timestamp sub-second, row position)
    must not predict injected-vs-real or illicit. AUC ~ 0.5 expected."""
    obs = pd.read_parquet(rel / "observable" / "track_m" / "transactions.parquet")
    lab = pd.read_parquet(rel / "truth" / "track_m" / "tx_labels.parquet").set_index("txid")
    X = pd.DataFrame({
        "txid_b0": obs["txid"].str[:2].map(lambda h: int(h, 16)), "txid_b1": obs["txid"].str[2:4].map(lambda h: int(h, 16)),
        "txid_last": obs["txid"].str[-2:].map(lambda h: int(h, 16)),
        "ms": obs["timestamp"].dt.microsecond // 1000, "row_pos": np.arange(len(obs)) / len(obs),
        "addr0_len": obs["output_addresses"].map(lambda a: len(a[0])),
        "addr0_last_char": obs["output_addresses"].map(lambda a: ord(a[0][-1])),
        "src_port": obs["src_port"], "dst_port": obs["dst_port"],
        "ip_last_octet": obs["src_ip"].map(lambda i: int(i.split(".")[-1]) if ":" not in i else int(i.split(":")[-1] or "0", 16) % 256),
    })
    # compare only against real *payment* traffic, balanced by construction of the metric (AUC)
    pay = (lab.loc[obs["txid"], "scenario"] != "real_data_carrier").to_numpy()
    inj = (lab.loc[obs["txid"], "origin"] == "injected").to_numpy()
    ill = (lab.loc[obs["txid"], "is_illicit"] == True).to_numpy()            # noqa: E712
    out = {}
    g = lab.loc[obs["txid"], "entity"].to_numpy()
    out["injected_vs_real_auc"], out["injected_vs_real_top"] = cv_auc(X[pay].drop(columns=["addr0_len"]), inj[pay], groups=g[pay])
    out["illicit_vs_rest_auc"], out["illicit_vs_rest_top"] = cv_auc(X[pay].drop(columns=["addr0_len"]), ill[pay], groups=g[pay])
    return out


def provenance_probe(rel: Path) -> dict:
    """Realism test (classifier two-sample test): can observable ledger features tell injected LICIT cover
    transactions from real payment transactions in the same window? Lower AUC = more realistic injections."""
    obs = pd.read_parquet(rel / "observable" / "track_m" / "transactions.parquet")
    lab = pd.read_parquet(rel / "truth" / "track_m" / "tx_labels.parquet").set_index("txid").loc[obs["txid"]]
    stats = json.loads((rel / "truth" / "track_m" / "stats_realized.json").read_text())
    F = ledger_features(obs, stats["btc_usd"])
    # the real transactions that served as bootstrap templates are removed from the real class: otherwise each
    # injected tx would be compared against its own near-twin, which makes injections look more realistic than they are
    templates = set(lab["template_of"].dropna()) if "template_of" in lab else set()
    real_pay = ((lab["scenario"] == "real_background") & ~lab.index.isin(templates)).to_numpy()
    inj_licit = ((lab["origin"] == "injected") & (lab["is_illicit"] == False)).to_numpy()   # noqa: E712
    inj_ill = ((lab["origin"] == "injected") & (lab["is_illicit"] == True)).to_numpy()      # noqa: E712
    g = lab["entity"].to_numpy()
    out = {"cv": "grouped by owning entity (StratifiedGroupKFold)"}
    m = real_pay | inj_licit
    out["injected_licit_vs_real_payment_auc"], out["top_giveaways"] = cv_auc(F[m], inj_licit[m], groups=g[m])
    out["injected_licit_vs_real_payment_auc_rowcv"] = round(cv_auc(F[m], inj_licit[m])[0], 4)
    m2 = real_pay | inj_ill
    out["injected_illicit_vs_real_payment_auc"], out["illicit_top_features"] = cv_auc(F[m2], inj_ill[m2], groups=g[m2])
    # per-scenario: how separable is each injected scenario from real payments?
    per = {}
    for sc in lab.loc[lab["origin"] == "injected", "scenario"].value_counts().index:
        ms = (lab["scenario"] == sc).to_numpy()
        if ms.sum() >= 20:
            mm = real_pay | ms
            auc = cv_auc(F[mm], ms[mm], groups=g[mm], min_groups=50)[0]
            per[sc] = round(auc, 3) if auc == auc else "n/a (<50 entities)"
    out["per_scenario_auc_vs_real_payments"] = per
    return out


def elliptic_transfer(rel: Path) -> dict:
    """External validity. Structure-only features computed identically on both tracks.
    floor  : 0.5
    ceiling: train on Elliptic time steps 1-34, test on 35-49 (real labels, same features)
    transfer: train on Track M (injected illicit vs everything else), test on Elliptic 35-49."""
    price = pd.read_csv(C.RAW / "price" / "btc_usd_daily_bitstamp.csv", parse_dates=["date"]).set_index("date")["btc_usd_close"]
    def feats(track):
        obs = pd.read_parquet(rel / "observable" / track / "transactions.parquet")
        lab = pd.read_parquet(rel / "truth" / track / "tx_labels.parquet").set_index("txid").loc[obs["txid"]].reset_index()
        px = price.reindex(lab["block_time"].dt.tz_localize(None).dt.normalize()).to_numpy()
        F = ledger_features(obs, px)
        F = pd.concat([F.reset_index(drop=True), graph_features(obs, lab)], axis=1)
        # drop script-type shares: era-dependent (2016 vs 2026), not behaviour
        F = F.drop(columns=["share_p2wpkh_out", "share_p2tr_out", "share_legacy_out", "log_feerate_approx"])
        return F, lab
    Fr, lr = feats("track_r")
    Fm, lm = feats("track_m")
    lab_r = lr["elliptic_class"].isin(["illicit", "licit"]).to_numpy()
    y_r = (lr["elliptic_class"] == "illicit").to_numpy()
    tr = lab_r & (lr["elliptic_timestep"] <= 34).to_numpy()
    te = lab_r & (lr["elliptic_timestep"] > 34).to_numpy()
    y_m = (lm["is_illicit"] == True).to_numpy()                                         # noqa: E712
    keep_m = (lm["scenario"] != "real_data_carrier").to_numpy()
    out = {"elliptic_test_n": int(te.sum()), "elliptic_test_illicit_share": round(float(y_r[te].mean()), 4)}
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    clf.fit(Fr[tr], y_r[tr])
    out["ceiling_auc_elliptic_train_to_test"] = round(float(roc_auc_score(y_r[te], clf.predict_proba(Fr[te])[:, 1])), 4)
    clf2 = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    w = np.where(y_m[keep_m], (~y_m[keep_m]).sum() / max(y_m[keep_m].sum(), 1), 1.0)
    clf2.fit(Fm[keep_m], y_m[keep_m], sample_weight=w)
    out["transfer_auc_trackM_to_elliptic_test"] = round(float(roc_auc_score(y_r[te], clf2.predict_proba(Fr[te])[:, 1])), 4)
    # structural comparison of illicit transactions: real (Elliptic) vs injected (Track M)
    comp = {}
    for col in ["n_in", "n_out", "in_deg", "out_deg", "equal_out_max", "max_out_share"]:
        a = Fr.loc[y_r & lab_r, col]
        b = Fm.loc[y_m, col]
        from scipy.stats import ks_2samp
        comp[col] = {"elliptic_illicit_mean": round(float(a.mean()), 3), "injected_illicit_mean": round(float(b.mean()), 3),
                     "ks": round(float(ks_2samp(a, b).statistic), 3)}
    out["illicit_structure_elliptic_vs_injected"] = comp
    return out


def baselines(rel: Path) -> dict:
    """Headroom checks on Track M injected entities (exact truth)."""
    obs = pd.read_parquet(rel / "observable" / "track_m" / "transactions.parquet")
    lab = pd.read_parquet(rel / "truth" / "track_m" / "tx_labels.parquet").set_index("txid")
    wal = pd.read_parquet(rel / "truth" / "track_m" / "wallets_injected.parquet").set_index("address")["entity"]
    ents = pd.read_parquet(rel / "truth" / "track_m" / "entities.parquet").set_index("entity")
    exact = set(ents.index[ents["truth_quality"] == "exact"])
    # CIOH over all observable txs, evaluated on injected addresses with exact owners
    from .cluster import UnionFind, coinjoin_rule
    uf = UnionFind()
    for a, v in zip(obs["input_addresses"], obs["output_sats"]):
        if coinjoin_rule(list(v), len(a)) is None:
            for x in a[1:]:
                uf.union(a[0], x)
    addrs = [a for a in pd.unique(np.concatenate(obs["input_addresses"].to_numpy())) if a in wal.index and wal[a] in exact]
    true = pd.Series({a: wal[a] for a in addrs})
    pred = pd.Series({a: uf.find(a) for a in addrs})
    def pairs(s):
        g = s.groupby(s).size()
        return int((g * (g - 1) // 2).sum())
    tp = pairs(pd.Series(list(zip(true, pred))).astype(str))
    out = {"cioh_addresses_evaluated": len(addrs),
           "cioh_pairwise_precision": round(tp / max(pairs(pred), 1), 4),
           "cioh_pairwise_recall": round(tp / max(pairs(true), 1), 4)}
    # naive peeling rule: 1 input, 2 outputs, one output < 10% of total, chained >= 3 hops
    pat = pd.read_parquet(rel / "truth" / "track_m" / "patterns.parquet")
    peel_tx = {t for tx in pat.loc[pat["type"] == "peeling_chain", "txids"] for t in tx}
    neg_tx = {t for tx in pat.loc[pat["type"] == "exchange_peel_negative", "txids"] for t in tx}
    o = obs.set_index("txid")
    small = o["output_sats"].map(lambda v: len(v) == 2 and min(v) / sum(v) < 0.10) & (o["input_addresses"].str.len() == 1)
    flagged = set(o.index[small])
    txids = set(o.index)
    parent_of = lab["input_outpoints"].map(lambda x: {y.split(":")[0] for y in x} & txids)
    chained = {t for t in flagged if parent_of.get(t) and (parent_of[t] & flagged)}
    tp_p = len(chained & peel_tx)
    out["naive_peel_rule_flagged"] = len(chained)
    out["naive_peel_rule_recall_on_illicit_peels"] = round(tp_p / max(len(peel_tx), 1), 4)
    out["naive_peel_rule_hits_on_exchange_negatives"] = len(chained & neg_tx)
    out["naive_peel_rule_precision_vs_illicit_peels"] = round(tp_p / max(len(chained), 1), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    args = ap.parse_args()
    rel = Path(args.release)
    rep = {}
    hard = []
    for track in ("track_r", "track_m"):
        if (rel / "observable" / track).exists():
            rep[f"invariants_{track}"] = inv = invariants(rel, track)
            hard += [f"{track}:{k}" for k in inv["_hard_fail"]]
            rep[f"format_equivalence_{track}"] = fe = format_equivalence(rel, track)
            if not fe["all_equal"]:
                hard.append(f"{track}:format_equivalence")
            print(f"[qa] {track} invariants + formats done; failures: {inv['_hard_fail']}", flush=True)
    rep["leakage_probe"] = leakage_probe(rel)
    print("[qa] leakage probe", rep["leakage_probe"], flush=True)
    rep["provenance_probe"] = provenance_probe(rel)
    print("[qa] provenance probe", rep["provenance_probe"], flush=True)
    rep["elliptic_transfer"] = elliptic_transfer(rel)
    print("[qa] elliptic transfer", rep["elliptic_transfer"], flush=True)
    rep["baselines"] = baselines(rel)
    print("[qa] baselines", rep["baselines"], flush=True)
    rep["hard_failures"] = hard
    out = rel / "qa"
    out.mkdir(exist_ok=True)
    (out / "qa_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print("[qa] hard failures:", hard)
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
