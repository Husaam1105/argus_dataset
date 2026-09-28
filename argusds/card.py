"""Render DATASET_CARD.md and DATA_DICTIONARY.md for a release from its build stats and QA report.

    python -m argusds.card --release data/releases/argus-dataset-v0.3.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config as C


def pct(x) -> str:
    return f"{100 * float(x):.2f}%"


def card(rel: Path) -> str:
    b = json.loads((rel / "truth" / "build_stats.json").read_text())
    qa = json.loads((rel / "qa" / "qa_report.json").read_text()) if (rel / "qa" / "qa_report.json").exists() else {}
    r, m = b.get("track_r", {}), b.get("track_m", {})
    bn = json.loads(C.BITNODES_SUMMARY.read_text())
    lk, pv, el, bl = (qa.get(k, {}) for k in ("leakage_probe", "provenance_probe", "elliptic_transfer", "baselines"))
    inv_r, inv_m = qa.get("invariants_track_r", {}), qa.get("invariants_track_m", {})
    hard = qa.get("hard_failures", ["(QA not run)"])
    per = pv.get("per_scenario_auc_vs_real_payments", {})
    comp = el.get("illicit_structure_elliptic_vs_injected", {})
    L = []
    a = L.append
    a(f"# ARGUS Bitcoin Transaction-Traffic Dataset — v{rel.name.split('-v')[-1]}")
    a("")
    a("Team Doomsbyte · SIH 2026 · PS SIH26146 (AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic)")
    a("")
    a("## 1. What this dataset is")
    a("")
    a("Two tracks, each delivered with exactly the brief's fields (timestamp, src/dst IP & port, TXID, input/output "
      "addresses & amounts, fee, script type, geo country, ASN) in CSV, JSON, JSONL, XML and Parquet, plus a "
      "physically separate ground-truth folder.")
    a("")
    a("| Layer | Track R (real labels) | Track M (modern, exact truth) |")
    a("|---|---|---|")
    a(f"| Ledger | **Real** Bitcoin mainnet: the {r.get('kept', 0):,} transactions behind the Elliptic dataset "
      f"({r.get('time_range', ['?', '?'])[0][:10]} → {r.get('time_range', ['?', '?'])[1][:10]}) | **Real** mainnet "
      f"background: {m.get('real', 0):,} txs, window `{m.get('source_window', '?')}` + {m.get('injected', 0):,} injected txs |")
    a(f"| Labels | **Real**: Elliptic (Weber et al. 2019) — illicit {r.get('kept_by_class', {}).get('illicit', 0):,}, licit "
      f"{r.get('kept_by_class', {}).get('licit', 0):,}, unknown {r.get('kept_by_class', {}).get('unknown', 0):,} | "
      f"**Exact** for injected campaigns ({m.get('illicit_tx', 0)} illicit txs); real background = unknown |")
    a("| Network layer | Simulated (label-blind) | Simulated (per-operator OPSEC for injected illicit actors) |")
    a("| Entity truth | none (CIOH proxy only, marked as such) | exact for scripted injected entities |")
    a("")
    a("**Only the network layer is simulated**, because real peer IPs are personal data and are not published by anyone. "
      "Every IP in the dataset lies in reserved, non-routable space (`240.0.0.0/4`, `2001:db8::/32`) and resolves, via "
      "the shipped `observable/geoip/*.mmdb` (GeoLite2-compatible), to **real** ASN / organisation / country metadata. "
      "No IP address in this dataset belongs to any real party.")
    a("")
    a("## 2. How it was built")
    a("")
    a("1. **Real ledger.** One-time fetch from the Google BigQuery public dataset `bigquery-public-data.crypto_bitcoin` "
      "(`tools/fetch_bigquery.py`). Track R transactions were located through the public deanonymisation of Elliptic "
      f"(Kaggle `alexbenzik/deanonymized-995-pct-of-elliptic-transactions`, {r.get('elliptic_mapped', 0):,} of 203,769 mapped; "
      "every illicit transaction is mapped). Track M is a contiguous mainnet window "
      f"(`{m.get('source_window', '?')}`) chosen near the 5-week median of mainnet transaction shape "
      "(`calibration/mainnet_daily_shape_*.csv`); about two-thirds of 2026 mainnet transactions carry an OP_RETURN "
      "data payload and are kept as `real_data_carrier`.")
    a("2. **Normalisation.** Integer satoshis; coinbase excluded; zero-value OP_RETURN outputs dropped from output "
      "arrays (counted in truth); transactions with non-address inputs or valued non-address outputs excluded and "
      f"listed in `truth/*/excluded.parquet` (Track R: {r.get('excluded', 0)}, Track M: "
      f"{sum(m.get('excluded_by_reason', {}).values())}).")
    a(f"3. **Injection (Track M only).** {m.get('injected', 0):,} transactions, timed to follow the real daily "
      "activity curve of the window. Illicit: **replays of real ransomware money flows** traced from Ransomwhere "
      "seeds (2020–22; victim payments → operator spends → layering, 38 families, `tools/fetch_ransomwhere_flows.py`) "
      "and **replays of real Elliptic illicit flow components** — both re-keyed, re-priced at USD parity with the "
      "Bitstamp daily BTC/USD series, script types modernised and time-compressed into the window — plus scripted "
      "ransomware lifecycles with exact pattern truth (ransom sizes drawn from real Ransomwhere payments ≥ $50; "
      "victim payments built from real payment templates; aggregation → peeling or Whirlpool-style CoinJoin → "
      "cash-out to *real* exchange deposit addresses from the window) and smurfing launderers. **Licit cover** is "
      "built with the same machinery: retail payments and exchange batch withdrawals generated by an *empirical "
      "shape bootstrap* of real transactions in the same window, PayJoins (break common-input ownership), exchange "
      "hot-wallet peeling withdrawals with real service withdrawal sizes and equal-amount batch payouts (hard "
      "negatives), and licit-only CoinJoin rounds. Injected fee rates come from real payments in the same block.")
    a("4. **Network layer.** Broadcast path per entity (own node clearnet / own node over Tor / shared light-wallet "
      "server / service infrastructure / VPS / proxy pool); stable per-day peer sets (the entry-node effect of "
      "Biryukov et al. 2014); Bitcoin Core Poisson trickle relay (mean 2 s outbound, 5 s inbound); three sensors "
      f"connected to {pct(m.get('sensor_peer_coverage', 0.3))} of reachable relay nodes; TCP-direction-aware ports; "
      f"relay-node ASN/country and IPv4/IPv6 mix from a Bitnodes snapshot ({bn['total_nodes']:,} reachable nodes, "
      f"{pct(bn['net_share'].get('onion', 0))} onion, {pct(bn['net_share'].get('ipv4', 0))} IPv4, "
      f"{pct(bn['net_share'].get('ipv6', 0))} IPv6; clearnet port 8333 share {pct(bn['clearnet_port_8333_share'])}; "
      "raw IPs discarded after aggregation). Besides the one-row-per-transaction file, each track ships "
      "`announcements.*`: the first 8 (peer, sensor) announcements of every transaction — the traffic log a real "
      "monitoring node records, and the raw material for entry-node fingerprinting.")
    a("")
    a("## 3. Validation (from `qa/qa_report.json`)")
    a("")
    a(f"**Hard invariants:** {'all pass' if not hard else 'FAILURES: ' + ', '.join(hard)} — value conservation, positive "
      "amounts, unique 64-hex txids, index-aligned arrays, valid Base58Check/Bech32/Bech32m checksums "
      f"({inv_r.get('addresses_checked', 0):,} + {inv_m.get('addresses_checked', 0):,} addresses), no double spend, "
      "parent-before-child creation times, observed ≥ created (within sensor clock skew), every IP in shadow space and "
      "resolving to the row's ASN/country, ports in range, and byte-identical content across all five formats.")
    a("")
    a("| Check | Result | Reading |")
    a("|---|---|---|")
    a(f"| Leakage probe: raw-string artifacts → injected vs real | AUC {lk.get('injected_vs_real_auc', float('nan')):.3f} | "
      "≈0.5 means txids, row order, sub-second timestamps and address strings carry no provenance signal |")
    a(f"| Leakage probe: raw-string artifacts → illicit | AUC {lk.get('illicit_vs_rest_auc', float('nan')):.3f} | "
      "residual lift comes mainly from row position (campaigns are bursty and cluster in time within the window) and from "
      "ports/IPs (the intended per-operator OPSEC signal); txid bytes and address strings contribute nothing |")
    a(f"| Realism (classifier two-sample test, entity-grouped CV, bootstrap templates excluded): injected **licit** vs real payments | AUC "
      f"{pv.get('injected_licit_vs_real_payment_auc', float('nan')):.3f} | lower = harder to tell injected cover from real traffic |")
    a(f"| Injected **illicit** vs real payments | AUC {pv.get('injected_illicit_vs_real_payment_auc', float('nan')):.3f} | "
      "illicit behaviour is meant to differ (aggregation, peeling, mixing) |")
    a(f"| Sim-to-real: train on Track M, test on real Elliptic (steps 35–49) | AUC "
      f"{el.get('transfer_auc_trackM_to_elliptic_test', float('nan')):.3f} | floor 0.5; ceiling (train on Elliptic 1–34) "
      f"{el.get('ceiling_auc_elliptic_train_to_test', float('nan')):.3f} — injected crime carries real illicit structure |")
    a(f"| Common-input-ownership on injected entities | precision {bl.get('cioh_pairwise_precision', float('nan')):.3f}, "
      f"recall {bl.get('cioh_pairwise_recall', float('nan')):.3f} | imperfect by design (PayJoin, fresh change) — ER pass 2 has real work |")
    a(f"| Naive peeling rule | recall {bl.get('naive_peel_rule_recall_on_illicit_peels', float('nan')):.3f}, precision "
      f"{bl.get('naive_peel_rule_precision_vs_illicit_peels', float('nan')):.3f}, "
      f"{bl.get('naive_peel_rule_hits_on_exchange_negatives', 0)} hits on exchange negatives | the hard negatives work |")
    a("")
    if per:
        a("Per-scenario separability from real payments (AUC, template-excluded; scenarios with < 50 distinct entities are not estimated): " +
          ", ".join(f"{k} {v:.2f}" for k, v in per.items() if isinstance(v, (int, float))) + ". "
          "Whirlpool premix (tx0) transactions are trivially separable (1 input -> several identical outputs plus a pool-fee "
          "output), exactly as real tx0s are; they are licit and exist so that 'mixing-shaped' does not imply 'illicit'.")
        a("")
    if comp:
        a("Illicit transaction structure, real Elliptic vs injected (KS statistic): " +
          ", ".join(f"{k} {v['ks']:.2f}" for k, v in comp.items()) + ".")
        a("")
    grid_p = rel / "qa" / "headroom_grid.json"
    if grid_p.exists():
        g = json.loads(grid_p.read_text())
        a("### Network-layer headroom (does fusing the network layer help entity resolution?)")
        a("")
        a("Pairwise F1 on Track M injected entities with exact ownership, for datasets rebuilt at different sensor "
          "coverages (`argusds/headroom.py`). *dual* = common-input ownership ∪ entry-node fingerprints from the "
          "announcement log, ignoring fingerprints shared by many unrelated wallets (light-wallet servers, proxies).")
        a("")
        a("| Sensor coverage | CIOH only | same first-seen IP (naive) | dual (hub-aware) | gain |")
        a("|---|---|---|---|---|")
        for row in g["grid"]:
            a(f"| {pct(row['coverage'])} | {row['cioh']:.3f} | {row['ip_hour']:.3f} | {row['dual_hubaware']:.3f} | "
              f"{(row['dual_hubaware'] / row['cioh'] - 1) * 100:+.0f}% |")
        a("")
        a("By broadcast path (dual / CIOH F1 at the release's coverage): " +
          ", ".join(f"{k} {v['dual_hubaware']:.2f}/{v['cioh']:.2f}" for k, v in g.get("by_path", {}).items()) + ".")
        a("")
        gains = [r["dual_hubaware"] / r["cioh"] - 1 for r in g["grid"]]
        mono = all(x <= y + 0.01 for x, y in zip(gains, gains[1:]))
        low, high = min(gains[1:]) * 100, max(gains[1:]) * 100
        a("Reading: the network layer adds information only for actors who broadcast from their own infrastructure "
          "(self-hosted nodes, VPS, services); light-wallet, Tor and proxy users gain nothing, and naive same-IP "
          "linking is worse than no network data — consistent with Neudecker & Hartenstein (2017). "
          + (f"The gain grows with monitoring coverage ({low:+.0f}% to {high:+.0f}% above the lowest setting)."
             if mono else
             f"Above the lowest coverage the gain is {low:+.0f}% to {high:+.0f}%, but it does not grow monotonically: "
             "this simple fixed-threshold linker (shared-peer count, Jaccard and hub cut-offs) discards more links as "
             "fingerprints get richer at high coverage, so it under-uses the extra signal — headroom for a learned "
             "model. Note: the hub cut-off (10) was chosen by a sweep on a 30%-coverage development build, which "
             "favours the 30% setting; the other thresholds were fixed a priori."))
        a("")
    a("## 4. Prevalence")
    a("")
    a(f"Track R keeps Elliptic's labelled prevalence (illicit {pct(r.get('kept_by_class', {}).get('illicit', 0) / max(r.get('kept', 1), 1))} "
      f"of all transactions). Track M is closer to reality: illicit {pct(m.get('illicit_share_all', 0))} of all "
      f"transactions, {pct(m.get('illicit_share_non_data_carrier', 0))} excluding OP_RETURN data carriers. "
      "Report precision / PR-AUC with the prevalence stated.")
    a("")
    a("## 4b. Robustness variants and reproducibility")
    a("")
    a("`observable/<track>/dirty/` holds faulted copies of the clean data (~0.5% of records: bad checksums, "
      "negative / NaN amounts, array-length mismatch, bad timestamps, conflicting and exact duplicate txids, invalid "
      "IPs and ports, missing fields, conservation violations, truncated records — every fault located in "
      "`truth/<track>/faults.parquet`) and a schema-drift CSV (renamed columns, epoch-ms timestamps, integer-satoshi "
      "amounts). Rebuilding with the same seed is byte-identical (verified under different Python hash seeds), and "
      "`manifest.json` lists the SHA-256 of every file.")
    a("")
    a("## 5. Label semantics")
    a("")
    a("- `is_illicit` (tx) — the transaction is initiated by an illicit entity (Elliptic-compatible). Victims' ransom "
      "payments are **not** illicit-initiated (`pays_illicit=True`, `scenario=ransom_payment`).")
    a("- `taint_hops` — hops from the nearest illicit-initiated injected transaction along injected spends.")
    a("- `coinjoin_rule` — rule-based *silver* label on real transactions (equal-output / Whirlpool 5×5 shape).")
    a("- `entity_truth_quality` — `exact` (scripted injected), `inherited_cioh` (replayed Elliptic actors and "
      "real-ransomware victims), `traced_heuristic` (real-ransomware operators: ransom-address spenders and hop-2 "
      "layering funded ≥75% by them), `cioh_proxy` (real background; **not** ground truth).")
    a("- Splits: Track R temporal (Elliptic steps 1–34 train / 35–49 test, graph-disjoint by construction); Track M "
      "campaign-disjoint.")
    a("")
    a("## 6. Assumptions (all in `argusds/config.py` and `argusds/build_track_m.py`, tagged [src]/[cal]/[assume])")
    a("")
    a("Broadcast-path mix for users (Track M: 15% own clearnet node, 15% own Tor node, 70% light-wallet server; "
      "Track R, 2016-17: 25% / 5% / 70%) · sensor peer "
      "coverage 30% (see the headroom grid for other values) · listening share of self-hosted nodes 35% · IP churn "
      "2%/day, 60% within the same /24 · mempool waiting time by fee-rate percentile · per-operator OPSEC (35% VPS, "
      "35% proxy pool, 30% Tor) · structuring threshold $10,000 · mixer fraction 30% · seed coverage 70% · "
      "Whirlpool pools 0.01 / 0.05 BTC with a 300-sat premix buffer · hosting-vs-residential ASN classification by "
      "organisation-name keywords.")
    a("")
    a("## 7. Known limitations")
    a("")
    a("- The network layer is a model, not a capture; its realism rests on the cited mechanisms and parameters above.")
    a("- Track R ledger is 2016–2017 (no Taproot, different fee market); its network layer uses an era-adjusted "
      "broadcast mix but today's relay model and node population (2016 topology is not recoverable).")
    a("- Track M is a finite slice; real campaign lifecycles (weeks of dormancy, cash-out) are compressed into it.")
    a("- Entity-resolution accuracy can only be scored on injected entities; real ownership is unknown.")
    a("- Replayed Elliptic flows inherit Elliptic's label quality (see DFRWS 2026, 'The Enemy of Reproducibility is "
      "Opacity'); we use only Elliptic's labels and hashes, never its anonymised features.")
    a("- Out of scope: Lightning, OP_RETURN payloads, coinbase, bare multisig / P2PK, mempool-only (never-confirmed) "
      "transactions, RBF replacements.")
    a("- Track R real addresses carry Elliptic's published labels. Keep the release inside the hackathon / evaluation "
      "context; no real address is ever labelled illicit by us.")
    a("")
    a("## 8. Sources")
    a("")
    a("Weber et al., KDD-W 2019 (Elliptic) · Elliptic deanonymisation (Kaggle, A. Benzik) · Google BigQuery "
      "`crypto_bitcoin` (blockchain-etl) · Bitnodes snapshot (resolved with DB-IP Lite, CC BY 4.0) · Bitstamp daily "
      "BTC/USD · Meiklejohn et al., IMC 2013 · Biryukov, Khovratovich, Pustogarov, CCS 2014 · Fanti & Viswanath, "
      "NeurIPS 2017 · Bitcoin Core net_processing (trickle constants) · Kappos et al., USENIX Security 2022 · "
      "Neudecker & Hartenstein 2017 · Ransomwhere (Cable et al. 2022, api.ransomwhe.re) · OFAC SDN list (XBT "
      "addresses) · BIP-78 (PayJoin), BIP-173/350 (Bech32/Bech32m).")
    return "\n".join(L) + "\n"


DICT = """# Data dictionary

## observable/<track>/transactions.{csv,json,jsonl,xml,parquet}
One row per transaction, ordered by first-seen time. All formats decode to the same table (hash-checked in QA).

| Field | Type | Meaning |
|---|---|---|
| txid | 64-hex | transaction id |
| timestamp | ISO-8601 UTC, ms | first time any sensor saw the transaction announced (not block time) |
| src_ip | IPv4/IPv6 (shadow space) | remote peer that announced it first (orientation is always peer → sensor) |
| src_port | int | sensor-outbound link: the peer's listen port; inbound link: the peer's ephemeral port (sticky per session) |
| dst_ip | IPv4 (shadow space) | the sensor that saw it first |
| dst_port | int | inbound link: 8333; sensor-outbound link: the sensor's ephemeral port |
| input_addresses / input_amounts | list / list of BTC (8 dp) | index-aligned; CSV uses `|` separators |
| output_addresses / output_amounts | list / list of BTC (8 dp) | address-bearing outputs only (OP_RETURN dropped) |
| fee | BTC (8 dp) | sum(inputs) − sum(outputs) |
| script_type | P2PKH/P2SH/P2WPKH/P2WSH/P2TR/P2A | type of the largest output |
| geo_country | ISO-3166 alpha-2 | country of src_ip (real metadata via shadow GeoIP) |
| asn | int | ASN of src_ip (real metadata via shadow GeoIP) |

Parquet stores amounts as integer satoshis (`input_sats`, `output_sats`, `fee_sats`).

## observable/<track>/announcements.{csv,parquet}
Traffic log: the first 8 (peer, sensor) announcements of every transaction. Columns: txid, timestamp, src_ip,
src_port, dst_ip, dst_port, geo_country, asn, rank (0 = the first-seen record in transactions.*).

## observable/<track>/dirty/
`transactions_dirty.{csv,jsonl,xml}` (faulted copies; locations in truth/<track>/faults.parquet) and
`transactions_drift.csv` (schema drift; column map in README.txt).

## observable/geoip/
`ARGUS-Shadow-ASN.mmdb`, `ARGUS-Shadow-Country.mmdb` — MaxMind-format databases covering every shadow prefix.

## truth/<track>/ (evaluation only — never give to the pipeline as input)
| File | Content |
|---|---|
| tx_labels.parquet | per tx: labels (Track R: elliptic_class, elliptic_timestep; Track M: origin, scenario, is_illicit, campaign_id, pattern_id, hop_index, taint_hops, pays_illicit, is_coinjoin, is_payjoin, replay_of), coinjoin_rule (silver), entity / entity_proxy, input_outpoints (exact spend links), network truth (t_create, broadcast_path, origin_ip, first_seen_is_origin, sensor_id, conn_direction, net_delay_s), split |
| entities.parquet / entities_proxy.parquet | entity type, role, is_illicit, campaign, truth quality |
| wallets_injected.parquet | address → entity for every injected address (Track M) |
| patterns.parquet | planted patterns and hard negatives with their txids and wallets (Track M) |
| seeds.parquet | known-illicit seed addresses for risk propagation (Track M; ~70% of campaigns seeded) |
| ip_truth.parquet | every IP: node, node kind, entity, validity start, real ASN/org/country, hosting vs residential |
| excluded.parquet | real transactions left out of the observable layer, with reason |
| stats_realized.json | realised parameters (e.g. measured ip_noise per broadcast path) |
| faults.parquet | every injected fault in the dirty variant: file, record_index, txid, fault_type |
| ../shadow_prefix_map.parquet | shadow prefix → real ASN/org/country |
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    rel = Path(ap.parse_args().release)
    (rel / "DATASET_CARD.md").write_text(card(rel), encoding="utf-8")
    (rel / "DATA_DICTIONARY.md").write_text(DICT, encoding="utf-8")
    print("[card] written")


if __name__ == "__main__":
    main()
