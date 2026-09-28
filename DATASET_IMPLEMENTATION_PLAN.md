# ARGUS — Dataset: design and implementation (v3, as built)

*Team Doomsbyte · SIH 2026 · PS SIH26146. This document describes what was built and why. The earlier fully-synthetic
plan (v2) is archived in `docs/archive/`. Final numbers live in the release's `DATASET_CARD.md` and `qa/qa_report.json`.*

## 1. Goal and the key decision

The problem statement ships no dataset and no labels, yet every ARGUS claim (dual-layer fusion, entity resolution,
pattern detection, risk propagation) must be demonstrated on data a judge can trust. The deciding observation:

| Layer of the brief's data | Real source available? | Decision |
|---|---|---|
| Ledger (txid, addresses, amounts, fee, script type) | **Yes** — the Bitcoin blockchain is public | **Use real transactions** |
| Network (src/dst IP, port, ASN, geo, first-seen time) | No — peer IPs are personal data, nobody publishes them | **Simulate**, from cited mechanisms, in non-routable IP space |
| Labels (who owns what, what is illicit) | Sparse (Elliptic, Ransomwhere, OFAC) | **Use real labels where they exist; inject campaigns with exact truth where they do not** |

So the dataset is a **hybrid**: real ledger + simulated network layer + real and injected labels. Nothing that is
publicly real is simulated, and the one simulated layer is measured against what the literature predicts.

## 2. Two tracks

| | Track R — real labels | Track M — modern, exact truth |
|---|---|---|
| Ledger | the ~203k real transactions behind the Elliptic dataset (2016-17), located via the public 99.5% deanonymisation mapping | a real 24-hour mainnet window (24 Sep 2026, ~634k txs) |
| Illicit labels | Elliptic's real labels (4,545 illicit) | injected campaigns with exact truth (~0.2–0.3% prevalence) |
| What it proves | detectors work on real criminal activity | entity resolution, patterns and risk propagation can be scored exactly on 2026-era data |

## 3. Pipeline (code in `argusds/`, one-time downloads in `tools/`)

1. **One-time downloads** (the only online step): BigQuery `crypto_bitcoin` (Track R ledger via the Elliptic hash
   mapping; the Track M window; real ransomware flows traced from Ransomwhere seeds), a Bitnodes snapshot (aggregated,
   raw IPs discarded), DB-IP Lite ASN/country databases, Bitstamp daily BTC/USD, Ransomwhere export, OFAC SDN XBT list.
2. **Normalise** (`normalize.py`): integer satoshis; coinbase and non-address outputs handled by documented rules.
3. **Inject** (Track M, `build_track_m.py`):
   - illicit: replays of **real ransomware money flows** (Ransomwhere seeds → operator spends → layering, 38
     families) and of **real Elliptic illicit flows**, re-keyed, re-priced at USD parity, script types modernised;
     scripted ransomware lifecycles and smurfing launderers for exact pattern truth (ransom sizes from real payments);
   - licit cover built with the same machinery: retail payments and exchange batches bootstrapped from real
     transactions of the same window, PayJoins, exchange hot-wallet peeling chains and equal-amount batch payouts
     (hard negatives), licit CoinJoin rounds;
   - campaigns may start before the window and are cut by it, as in a real capture; injected activity follows the
     real daily curve.
4. **Network layer** (`network.py`, `geo.py`): broadcast paths (own node / Tor / light-wallet server / service / VPS /
   proxy), stable daily entry-node sets, Bitcoin Core Poisson trickle relay (2 s / 5 s), three sensors, TCP-direction
   ports; shadow IP space (`240.0.0.0/4`, `2001:db8::/32`) with GeoLite2-format databases mapping to real ASN/country.
   Outputs both the brief's one-row-per-tx table and an **announcement log** (first 8 announcements per tx).
5. **Export** (`export.py`): CSV, JSON, JSONL, XML, Parquet — identical content (hash-checked); dirty/drift variants
   with an exact fault manifest (`dirty.py`).
6. **Validate** (`qa.py`, `headroom.py`) and **document** (`card.py`). `python -m argusds.release --version X` runs
   everything; rebuilds are byte-identical.

## 4. How legitimacy is established (the checks and why each exists)

| Check | Question it answers |
|---|---|
| Hard invariants (conservation, no double spend, parent-before-child, valid checksums, IPs resolve, 5-format equivalence, determinism) | Is the data internally consistent and reproducible? |
| Leakage probe (txid bytes, row order, sub-second time, address strings, ports) | Can labels or provenance be read from artifacts? (target ≈ 0.5) |
| Classifier two-sample test, entity-grouped CV, bootstrap templates excluded | How distinguishable are injected licit transactions from real payments? |
| Sim-to-real transfer: train on Track M, test on real Elliptic labels | Does the injected crime carry real illicit structure? |
| Coverage-grid headroom (`headroom.py`) | Does the network layer help entity resolution, for whom, and how much? |
| Baselines (common-input ownership, naive peel rule vs hard negatives) | Are tasks neither trivial nor impossible? |

## 5. Main findings (see the release card for final values)

Release v1.0.1 (Track M = 24 Sep 2026, 633,330 real + 6,035 injected txs, 801 illicit = 0.13%):
- All hard invariants pass on both tracks; rebuilds are byte-identical.
- Leakage probe ≈ chance for provenance (AUC 0.518); 0.610 for illicit, mostly from the intended network-layer
  (OPSEC) signal and campaign burstiness.
- Injected licit cover vs real payments: AUC 0.697 (entity-grouped CV, bootstrap templates excluded) — partly
  distinguishable; the card lists the giveaway features (USD size, fee rate, output shares).
- A model trained only on Track M injected crime reaches AUC 0.718 on real Elliptic labels (floor 0.5, ceiling 0.904).
- The network layer helps entity resolution **only** for actors broadcasting from their own infrastructure: combined
  F1 is −1% vs common-input ownership at 10% sensor coverage and +19–30% at 30–100%. Light-wallet, Tor and proxy
  users gain nothing and naive same-IP linking is harmful — consistent with Neudecker & Hartenstein (2017). The gain
  is not monotonic in coverage because the simple reference linker's fixed thresholds under-use richer fingerprints
  (its hub cut-off was tuned at 30% coverage) — headroom for ARGUS's learned model.

## 6. Known limitations and what would improve them

- Network layer is modelled, not captured (a real capture would need a synced node and raises privacy/legal issues).
- Entity ground truth exists only for injected entities.
- Real laundering (weeks long) is compressed into a 24-hour window; a 48-hour variant can be built from the existing
  download (`--window mainnet_20260923T00_48h.parquet`).
- Track R's network layer uses an era-adjusted broadcast mix but today's relay model and node population.
