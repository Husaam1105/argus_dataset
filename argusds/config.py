"""Central configuration. Every realism parameter is tagged:
  [src]    taken from a cited source
  [cal]    measured from our calibration data (Bitnodes snapshot, real ledger windows)
  [assume] explicit assumption, documented in the dataset card and swept where it matters
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BUILD = ROOT / "data" / "build"
RELEASES = ROOT / "data" / "releases"

ELLIPTIC_DIR = ROOT / "elliptic_bitcoin_dataset"
ELLIPTIC_MAP = ROOT / "deanonyized99.5" / "Result.csv"
BITNODES_ASN = RAW / "bitnodes" / "bitnodes_asn_country.csv"
BITNODES_SUMMARY = RAW / "bitnodes" / "bitnodes_summary.json"
DBIP_ASN = RAW / "geoip" / "dbip-asn-lite.mmdb"
DBIP_COUNTRY = RAW / "geoip" / "dbip-country-lite.mmdb"

MASTER_SEED = 1337
TRACK_M_WINDOW = "mainnet_20260924T00_24h.parquet"   # main Track M window: 24 Sep 2026, full day (dev subset: mainnet_20260924T00_6h.parquet)
MMDB_BUILD_EPOCH = 1790467200     # fixed build_epoch written into the shadow .mmdb files (2026-09-27), for byte-identical rebuilds

# --- Shadow address space (no real, routable IP ever appears in a release) ---------------
SHADOW_V4 = "240.0.0.0/4"        # IANA reserved ("future use"), not routable
SHADOW_V6 = "2001:db8::/32"      # RFC 3849 documentation prefix
SHADOW_V4_BLOCK = 16             # each (ASN, country) gets one /16 of shadow space (256 /24s)
SHADOW_V6_BLOCK = 48

# --- Network layer -----------------------------------------------------------------------
NETWORK = {
    # Relay: Bitcoin Core announces txs to each peer on a Poisson timer,
    # mean 2 s to outbound peers and 5 s to inbound peers. [src: Bitcoin Core net_processing]
    "trickle_mean_s_outbound": 2.0,
    "trickle_mean_s_inbound": 5.0,
    "hop_latency_ms_median": 60.0,         # [assume] one-way latency, log-normal
    "hop_latency_sigma": 0.6,              # [assume]
    "peers_per_node": 10,                  # 8 outbound full-relay + ~2 inbound relaying [src: Core defaults; assume inbound]
    "sensors": 3,                          # [assume] K monitoring sensors, one operator
    "announcements_per_tx": 8,             # first A (peer, sensor) announcements logged per tx (traffic log)
    "sensor_country": "IN",                # [assume] sensors hosted in India if Bitnodes has IN hosting ASNs
    "sensor_peer_coverage": 0.30,          # [assume] share of reachable nodes each sensor is connected to; swept
    "p_origin_dials_sensor": 0.02,         # [assume] P(an origin node has a sensor among its outbound peers)
    "sensor_clock_skew_ms_sd": 40.0,       # [assume] fixed per-sensor offset
    "listening_share_own_node": 0.35,      # [assume] share of self-hosted user nodes that accept inbound (reachable)
    "relay_pool_size": 4000,               # [assume] reachable relay nodes modelled (scaled-down network)
    # Broadcast path for *user* entities (non-services). No public measurement exists. [assume]
    "user_broadcast_path_mix": {"own_node_clearnet": 0.15, "own_node_tor": 0.15, "light_wallet_server": 0.70},
    # Track R (2016-17): Tor use among wallet users was far lower and self-hosted full nodes more common. [assume]
    "user_broadcast_path_mix_2016": {"own_node_clearnet": 0.25, "own_node_tor": 0.05, "light_wallet_server": 0.70},
    "light_wallet_servers": 14,            # [assume] shared Electrum/Esplora-like broadcasters (Zipf popularity)
    "light_wallet_zipf_a": 1.3,            # [assume]
    "service_ips_min_max": (2, 6),         # [assume] exchange/pool/service hosting IP cluster size
    "ip_churn_per_day": 0.02,              # [assume] residential IP change rate
    "churn_same_24_prob": 0.6,             # [assume] new IP stays in the same /24
    "session_mean_days": 1.5,              # [assume] (peer, sensor) TCP session length -> sticky ephemeral ports
    "sensor_inbound_ephemeral": (32768, 60999),   # sensors run Linux [src: Linux ip_local_port_range default]
    # Peer-side ephemeral ports for inbound connections: OS / NAT mix [assume]
    "peer_ephemeral_mix": {"windows": (0.45, 49152, 65535), "linux": (0.35, 32768, 60999), "nat": (0.20, 1024, 65535)},
    # Mempool waiting time before confirmation: mean minutes as a function of the tx's fee-rate percentile p
    # within its block: mean = base + span * (1 - p)^3 [assume; calibrate against mempool data if available]
    "wait_base_min": 4.0,
    "wait_span_min": 90.0,
}

# Share of reachable-node networks [cal: Bitnodes snapshot 2026-09-27] is read from BITNODES_SUMMARY at runtime.

# Keywords used to classify an ASN's organisation as hosting/cloud (everything else = residential/ISP). [assume]
HOSTING_KEYWORDS = [
    "hetzner", "ovh", "amazon", "aws", "google", "microsoft", "azure", "digitalocean", "contabo", "akamai",
    "linode", "ionos", "oracle", "vultr", "choopa", "constant company", "alibaba", "tencent", "scaleway",
    "leaseweb", "m247", "datacamp", "hostinger", "netcup", "hosting", "host", "cloud", "data center",
    "datacenter", "colo", "server", "vps", "online s.a.s", "online sas", "psychz", "quadranet", "servers",
    "cogent", "g-core", "gcore", "stark industries", "hostkey", "selectel", "timeweb", "aeza", "worldstream",
    "serverius", "nforce", "ikoula", "upcloud", "kamatera", "rackspace", "softlayer", "ibm", "equinix",
    "cloudflare", "fastly", "zenlayer", "hivelocity", "tzulo", "interserver", "ramnode", "buyvm", "frantech",
    "bluehost", "godaddy", "a2 hosting", "dreamhost", "liquid web", "inmotion", "phoenixnap", "servercentral",
    "ctrls", "hurricane electric", "tier 4 cloud", "yotta", "sify", "netmagic", "e2e networks",
]
