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

## Download

**[data.zip on Google Drive](https://drive.google.com/file/d/1LQNZ5RlhYghxoxukF3BnE-0Q-glB2xRt/view?usp=drive_link)**
(3.1 GB). Unzip into the repository root so it becomes `data/`. Shared for evaluation (SIH judges and the team);
please do not redistribute — Track R pairs real Bitcoin addresses with Elliptic's published labels.

| Folder in `data/releases/` | Contents | Use |
|---|---|---|
| `argus-dataset-v1.0.1` | Track R (Elliptic, 2016–17) + Track M (24 Sep 2026, 24 h, 639k txs), dirty variants, QA report, dataset card | Main release |
| `argus-dataset-v1.0.1-6h` | Track M, 24 Sep 00:00–06:00 | Sealed test for ARGUS |
| `argus-dataset-vdev-6h`, `argus-dataset-vdev2-6h` | Track M, 23 Sep 00:00–06:00 and 06:00–12:00 | Dev sets: tune only on these |
| `argus-dataset-vfresh-6h` | Track M, 24 Sep 12:00–18:00 | Fresh check, never used for any choice |

Dev and test sets share no replayed criminal flows (`python -m argusds.build ... --exclude-replays-from <releases>`).
Each release holds `observable/` (the pipeline's input: `transactions.*` in CSV/JSON/JSONL/XML/Parquet,
`announcements.*`, shadow GeoIP `.mmdb` files) and `truth/` (evaluation only — never feed it to a model).
`data/raw/` holds the one-time downloads the releases were built from (BigQuery windows, Bitnodes aggregates,
DB-IP Lite GeoIP (CC BY 4.0, © DB-IP.com), Ransomwhere export, OFAC SDN XBT list, BTC/USD prices).

## Running the ARGUS prototype on this data

```bash
# 1. convert one release into the prototype's data-dir layout (CSV + ground truth in its schema + GeoIP files)
python tools/export_to_argus_prototype.py --release data/releases/argus-dataset-v1.0.1-6h --track track_m \
    --out ../ARGUS_prototype/data_argusds_m
# 2. in the prototype: point GeoIP at the shipped databases, then run the stages with --data-dir data_argusds_m
#    ARGUS_GEOIP_ASN_MMDB=data_argusds_m/geoip/ARGUS-Shadow-ASN.mmdb
#    ARGUS_GEOIP_COUNTRY_MMDB=data_argusds_m/geoip/ARGUS-Shadow-Country.mmdb
# 3. score the run (precision@k with seeds excluded, ER precision/recall/F1, anomaly-score AUC)
python tools/eval_argus_run.py --run ../ARGUS_prototype/data_argusds_m --truth data/releases/argus-dataset-v1.0.1-6h/truth/track_m
```

## Documentation

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
tools/       one-time online fetchers, the window slicer, the ARGUS exporter and the ARGUS run evaluator
calibration/ real-data calibration evidence (mainnet daily transaction shape)
docs/        archived earlier plans
```
