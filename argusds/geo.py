"""Shadow GeoIP: real ASN / organisation / country metadata attached to synthetic, non-routable prefixes.

The geo pool is the Bitnodes snapshot (reachable nodes, resolved with DB-IP Lite) aggregated to
(ASN, org, country, node count). Each pool entry is given its own blocks of shadow address space
(240.0.0.0/4 for IPv4, 2001:db8::/32 for IPv6) in a *shuffled* order, so neither the numeric order of
shadow prefixes nor their position reveals anything about the ASN behind them.
"""
from __future__ import annotations

import ipaddress
import json

import numpy as np
import pandas as pd

from . import config as C


def load_pool() -> pd.DataFrame:
    pool = pd.read_csv(C.BITNODES_ASN).dropna(subset=["asn", "country"])
    pool["asn"] = pool["asn"].astype(int)
    pool["org"] = pool["org"].fillna("Unknown")
    org = pool["org"].str.lower()
    pool["asn_class"] = np.where(org.apply(lambda o: any(k in o for k in C.HOSTING_KEYWORDS)), "hosting", "residential")
    pool = pool.groupby(["asn", "org", "country", "asn_class"], as_index=False)["nodes"].sum()
    return pool.sort_values("nodes", ascending=False).reset_index(drop=True)


class ShadowSpace:
    """Allocates shadow prefixes per pool entry and hands out host addresses inside them."""

    def __init__(self, pool: pd.DataFrame, rng: np.random.Generator):
        self.pool = pool.copy()
        n = len(pool)
        v4 = ipaddress.ip_network(C.SHADOW_V4)
        v6 = ipaddress.ip_network(C.SHADOW_V6)
        v4_blocks = v4.subnets(new_prefix=C.SHADOW_V4_BLOCK)
        v6_blocks = v6.subnets(new_prefix=C.SHADOW_V6_BLOCK)
        # One IPv4 block (/16 = 256 /24s) and one IPv6 /48 per entry, assigned in shuffled order.
        order = rng.permutation(n)
        v4_list = [next(v4_blocks) for _ in range(n)]
        v6_list = [next(v6_blocks) for _ in range(n)]
        self.pool["v4_blocks"] = [[str(v4_list[i])] for i in order]
        self.pool["v6_block"] = [str(v6_list[i]) for i in order]
        self._used: set[str] = set()
        self.rng = rng

    def entry(self, idx: int) -> pd.Series:
        return self.pool.iloc[idx]

    def pick_entry(self, asn_class: str | None = None, country: str | None = None) -> int:
        p = self.pool
        mask = np.ones(len(p), bool)
        if asn_class:
            mask &= (p["asn_class"] == asn_class).to_numpy()
        if country:
            m2 = mask & (p["country"] == country).to_numpy()
            if m2.any():
                mask = m2
        idx = np.flatnonzero(mask)
        w = p["nodes"].to_numpy()[idx].astype(float)
        return int(self.rng.choice(idx, p=w / w.sum()))

    def new_ip(self, idx: int, v6: bool = False, near: str | None = None) -> str:
        """A fresh host address in entry `idx`. With `near`, stay in the same /24 (IPv4) or /64 (IPv6)."""
        e = self.pool.iloc[idx]
        for attempt in range(1000):
            if attempt == 30:
                near = None          # that /24 (or /64) is full: move to a fresh one in the same ASN
            if v6:
                base = ipaddress.ip_network(e["v6_block"])
                if near:
                    net = ipaddress.ip_network(f"{near}/64", strict=False)
                else:
                    sub = int(self.rng.integers(0, 2 ** 16))
                    net = ipaddress.ip_network((int(base.network_address) + (sub << 64), 64))
                host = int(self.rng.integers(1, 2 ** 32))
                ip = str(ipaddress.ip_address(int(net.network_address) + host))
            else:
                if near:
                    net = ipaddress.ip_network(f"{near}/24", strict=False)
                else:
                    blk = ipaddress.ip_network(e["v4_blocks"][0])
                    n24 = 2 ** (24 - blk.prefixlen)
                    net = ipaddress.ip_network((int(blk.network_address) + (int(self.rng.integers(0, n24)) << 8), 24))
                ip = str(ipaddress.ip_address(int(net.network_address) + int(self.rng.integers(1, 255))))
            if ip not in self._used:
                self._used.add(ip)
                return ip
        raise RuntimeError("shadow space exhausted")

    def prefix_table(self) -> pd.DataFrame:
        rows = []
        for _, e in self.pool.iterrows():
            for b in e["v4_blocks"]:
                rows.append(dict(shadow_prefix=b, asn=e["asn"], org=e["org"], country=e["country"], asn_class=e["asn_class"]))
            rows.append(dict(shadow_prefix=e["v6_block"], asn=e["asn"], org=e["org"], country=e["country"], asn_class=e["asn_class"]))
        return pd.DataFrame(rows)


def write_shadow_mmdb(prefixes: pd.DataFrame, out_dir) -> dict:
    """Write GeoLite2-compatible ASN and Country databases covering every shadow prefix."""
    from mmdb_writer import MMDBWriter
    from netaddr import IPSet

    out_dir.mkdir(parents=True, exist_ok=True)
    asn_w = MMDBWriter(ip_version=6, ipv4_compatible=True, database_type="GeoLite2-ASN",
                       languages=["en"], description={"en": "ARGUS shadow ASN database (synthetic prefixes, real ASN metadata)"})
    cc_w = MMDBWriter(ip_version=6, ipv4_compatible=True, database_type="GeoLite2-Country",
                      languages=["en"], description={"en": "ARGUS shadow Country database (synthetic prefixes)"})
    for _, r in prefixes.iterrows():
        s = IPSet([r["shadow_prefix"]])
        asn_w.insert_network(s, {"autonomous_system_number": int(r["asn"]), "autonomous_system_organization": str(r["org"])})
        cc_w.insert_network(s, {"country": {"iso_code": str(r["country"])}, "registered_country": {"iso_code": str(r["country"])}})
    asn_path, cc_path = out_dir / "ARGUS-Shadow-ASN.mmdb", out_dir / "ARGUS-Shadow-Country.mmdb"
    from unittest import mock
    with mock.patch("time.time", return_value=float(C.MMDB_BUILD_EPOCH)):   # mmdb_writer stamps time.time()
        asn_w.to_db_file(str(asn_path))
        cc_w.to_db_file(str(cc_path))
    return {"asn_mmdb": asn_path.name, "country_mmdb": cc_path.name, "prefixes": int(len(prefixes))}


def net_share() -> dict:
    return json.loads(C.BITNODES_SUMMARY.read_text())["net_share"]
