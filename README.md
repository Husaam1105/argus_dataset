# ARGUS dataset builder

Builds the evaluation dataset for **ARGUS** (Team Doomsbyte, SIH 2026, PS SIH26146 — AI-powered monitoring and
analysis of Bitcoin transaction traffic): a **hybrid** of real Bitcoin ledger data, a simulated network layer, and
real + injected labels.

| Layer | Source |
|---|---|
| Ledger (txid, addresses, amounts, fee, script type) | **Real** Bitcoin mainnet (Google BigQuery `crypto_bitcoin`) |
| Network (src/dst IP & port, ASN, geo, first-seen time) | **Simulated** from cited relay mechanisms, in non-routable IP space with real ASN/country metadata |
| Labels | **Real** (Elliptic, Ransomwhere-traced ransomware flows) + **injected** campaigns with exact truth |

- **Track R** — the ~203k real transactions behind the Elliptic dataset (2016–17) with Elliptic's real labels.
- **Track M** — a real 24-hour 2026 mainnet window (~634k txs) with injected illicit campaigns (replays of real
  ransomware and Elliptic flows, scripted campaigns) and licit cover, all with exact ground truth.

Design, validation method and findings: [`DATASET_IMPLEMENTATION_PLAN.md`](DATASET_IMPLEMENTATION_PLAN.md).
Notes for pipeline developers: [`DATASET_HANDOFF.md`](DATASET_HANDOFF.md). Each built release also contains
`DATASET_CARD.md` (with the QA results) and `DATA_DICTIONARY.md`.

## Rebuild

```bash
pip install -r requirements-dataset.txt

# 1. third-party inputs (not redistributed here)
#    - Elliptic Data Set (Kaggle: ellipticco/elliptic-data-set)            -> elliptic_bitcoin_dataset/
#    - Elliptic deanonymisation (Kaggle: alexbenzik/deanonymized-995-pct-of-elliptic-transactions) -> deanonyized99.5/Result.csv
#    - DB-IP Lite ASN + Country mmdb (db-ip.com, CC BY 4.0)                 -> data/raw/geoip/
#    - Bitnodes snapshot aggregates, Bitstamp daily BTC/USD, Ransomwhere export, OFAC SDN XBT -> data/raw/...

# 2. one-time BigQuery downloads (free BigQuery sandbox project)
python tools/fetch_bigquery.py auth    --project <gcp-project>
python tools/fetch_bigquery.py track-r --project <gcp-project>
python tools/fetch_bigquery.py track-m --project <gcp-project> --start 2026-09-23T00:00:00 --hours 48
python tools/slice_window.py --src mainnet_20260923T00_48h.parquet --start 2026-09-24T00:00:00 --hours 24
python tools/fetch_ransomwhere_flows.py --project <gcp-project>

# 3. build + dirty variants + coverage grid + QA + dataset card (offline; byte-identical for the same seed)
python -m argusds.release --version 1.0.1
```

## Layout

```
argusds/     build pipeline: normalize, cluster, geo (shadow GeoIP), network (relay model), build_track_r/m,
             export, dirty, headroom, qa, card, release
tools/       one-time online fetchers and the window slicer
calibration/ real-data calibration evidence (mainnet daily transaction shape)
docs/        archived earlier plans
```
