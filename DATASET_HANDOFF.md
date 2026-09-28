# ARGUS dataset — hand-off for the pipeline team

Release folder: `data/releases/argus-dataset-v<version>/`. Read `DATASET_CARD.md` (what/how/validation) and
`DATA_DICTIONARY.md` (every field) first. This note is the short list of things that will bite you otherwise.

## 1. Things you must change in the pipeline

1. **IP addresses are in reserved space.** Every IP is in `240.0.0.0/4` (IPv4) or `2001:db8::/32` (IPv6) on purpose
   (no real person's IP is in the data). Python's `ipaddress.ip_address(x).is_global` is **False** for all of them and
   `is_reserved` is True for the IPv4 ones. If ingestion drops "private/reserved/bogon" IPs, it will drop every row.
   Add a `synthetic_mode` flag that skips that filter.
2. **Use the shipped GeoIP databases** for this data: `observable/geoip/ARGUS-Shadow-ASN.mmdb` and
   `ARGUS-Shadow-Country.mmdb`. They have the same record layout as MaxMind GeoLite2 (`autonomous_system_number`,
   `autonomous_system_organization`, `country.iso_code`), so only the file path changes. The real GeoLite2 files
   will not resolve these IPs. Keep real GeoLite2 for organiser-provided data.
3. **Never feed `truth/` into the pipeline.** It holds labels, entity ids, exact spend links (`input_outpoints`) and
   the true origin IPs. Evaluation only.

## 2. What to load

| Purpose | File |
|---|---|
| Main input (brief's fields, one row per tx) | `observable/<track>/transactions.{csv,json,jsonl,xml,parquet}` — identical content, pick one |
| Traffic log for network analysis (first 8 announcements per tx) | `observable/<track>/announcements.{csv,parquet}` |
| Ingestion robustness tests | `observable/<track>/dirty/` (+ `truth/<track>/faults.parquet` to score your validator) |
| Address validation | `argusds/addresses.py: validate_address(addr) -> script type or None` (Base58Check, Bech32, Bech32m, P2A) |

Parquet stores amounts as integer satoshis; text formats use BTC with 8 decimals.

## 3. Which track for what

- **Track R** (real Elliptic transactions + real Elliptic labels): train / evaluate the illicit-transaction classifier and
  risk propagation against real labels. Use the temporal split in `truth/track_r/tx_labels.parquet` (`split`:
  time steps 1–34 train, 35–49 test). No entity ground truth here.
- **Track M** (real 2026 mainnet + injected campaigns with exact truth): entity resolution (score only on entities with
  `entity_truth_quality` in {exact, inherited_cioh, traced_heuristic}), pattern detection (`patterns.parquet`: peeling
  chains, CoinJoin, smurfing, real ransomware replays, and hard negatives you must NOT flag), risk propagation from
  `seeds.parquet`, anomaly detection. Prevalence is realistic (~0.2–0.3% illicit) — report precision / PR-AUC.

## 4. What to expect from the network layer (so nobody over-claims)

Network features help entity resolution **only** for actors broadcasting from their own infrastructure (self-hosted
nodes, VPS, services) and more so at higher sensor coverage; they carry no ownership information for light-wallet,
Tor and proxy users, and naive "same IP ⇒ same owner" linking is worse than no network data. The measured curve is in
the dataset card ("Network-layer headroom"). Use `announcements.*` (entry-node fingerprints), not just `src_ip`.

## 5. Regenerating

`python -m argusds.release --version X` rebuilds everything from the raw downloads in `data/raw/` (byte-identical for
the same seed). Raw downloads are one-time (`tools/fetch_bigquery.py`, `tools/fetch_ransomwhere_flows.py`) and need
a Google Cloud (BigQuery sandbox) login.
