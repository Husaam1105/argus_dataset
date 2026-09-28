"""Simulated network layer: who a monitoring sensor first hears each transaction from, and when.

Model (all parameters in config.NETWORK, tagged [src]/[cal]/[assume]):
  1. t_create: the transaction enters the network some time before its block. Waiting time depends on its fee-rate
     percentile inside its block; parents are always created before children (in-dataset spends).
  2. Broadcast path per entity: own node (clearnet), own node over Tor, a shared light-wallet server,
     or a service's own hosting infrastructure. Injected entities may force a path (e.g. per-operator OPSEC).
  3. Relay race: the origin announces to its peers (a stable peer set per node per day: the "entry nodes"
     exploited by Biryukov et al.) after Poisson trickle delays (2 s outbound / 5 s inbound mean), peers announce
     onward to the sensors they are connected to. The first announcement to reach any sensor wins.
     Direct origin->sensor links exist only if the sensor is connected to the origin (listening origins) or the
     origin dialled the sensor. Tor origins are never directly visible.
  4. Ports follow the TCP connection direction: sensor-outbound links carry the peer's listen port as src_port;
     inbound links carry a sticky ephemeral src_port (OS/NAT mix) and dst_port 8333.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .geo import ShadowSpace

NET = C.NETWORK
DAY = 86400.0


def _splitmix(x: np.ndarray) -> np.ndarray:
    x = (x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return x ^ (x >> np.uint64(31))


def _u01(x: np.ndarray) -> np.ndarray:
    return (_splitmix(x) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


class Nodes:
    """Registry of network nodes. Each node has a (possibly churning) IP timeline, a class, and port behaviour."""

    def __init__(self, space: ShadowSpace, rng: np.random.Generator, t0: float, t1: float, v6_share: float,
                 port_dist: dict):
        self.space, self.rng, self.t0, self.t1 = space, rng, t0, t1
        self.v6_share = v6_share
        ports = {int(k): v for k, v in port_dist.items()}
        self.port_vals = np.array(list(ports.keys()))
        self.port_p = np.array(list(ports.values()), float)
        self.port_p /= self.port_p.sum()
        self.kind, self.listening, self.hidden, self.listen_port, self.os = [], [], [], [], []
        self.entry, self.entity, self.epoch_t, self.epoch_ip = [], [], [], []

    def add(self, kind: str, entry: int | None, listening: bool, hidden: bool = False, entity: int = -1,
            churn: bool = False, near: str | None = None, v6: bool | None = None) -> int:
        nid = len(self.kind)
        self.kind.append(kind)
        self.listening.append(listening)
        self.hidden.append(hidden)
        self.entity.append(entity)
        self.entry.append(-1 if entry is None else entry)
        self.listen_port.append(int(self.rng.choice(self.port_vals, p=self.port_p)) if listening else 0)
        self.os.append(int(self.rng.choice(3, p=[v[0] for v in NET["peer_ephemeral_mix"].values()])))
        if hidden:
            self.epoch_t.append(np.array([self.t0]))
            self.epoch_ip.append([""])
            return nid
        if v6 is None:
            v6 = bool(self.rng.random() < self.v6_share)
        ip = self.space.new_ip(entry, v6=v6, near=near)
        times, ips = [self.t0], [ip]
        if churn:
            n = self.rng.poisson(NET["ip_churn_per_day"] * (self.t1 - self.t0) / DAY)
            for t in np.sort(self.rng.uniform(self.t0, self.t1, n)):
                same = self.rng.random() < NET["churn_same_24_prob"]
                ips.append(self.space.new_ip(entry, v6=v6, near=ips[-1] if same else None))
                times.append(t)
        self.epoch_t.append(np.array(times))
        self.epoch_ip.append(ips)
        return nid

    def ip_at(self, node: np.ndarray, t: np.ndarray) -> np.ndarray:
        out = np.empty(len(node), dtype=object)
        order = np.argsort(node, kind="stable")
        sn = node[order]
        starts = np.flatnonzero(np.r_[True, sn[1:] != sn[:-1]])
        ends = np.r_[starts[1:], len(sn)]
        for s, e in zip(starts, ends):
            n, rows = sn[s], order[s:e]
            ips = self.epoch_ip[n]
            if len(ips) == 1:
                out[rows] = ips[0]
            else:
                k = np.searchsorted(self.epoch_t[n], t[rows], side="right") - 1
                out[rows] = np.asarray(ips, dtype=object)[np.clip(k, 0, None)]
        return out

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(dict(node=np.arange(len(self.kind)), kind=self.kind, listening=self.listening,
                                 hidden=self.hidden, listen_port=self.listen_port, entry=self.entry, entity=self.entity))


def epoch_s(ts: pd.Series) -> np.ndarray:
    return (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def creation_times(ledger: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Seconds since epoch at which each tx entered the network."""
    block_t = epoch_s(ledger["block_time"])
    p = ledger.groupby("block_height")["feerate"].rank(pct=True).to_numpy()
    mean_s = 60.0 * (NET["wait_base_min"] + NET["wait_span_min"] * (1.0 - p) ** 3)
    t = block_t - np.maximum(rng.exponential(mean_s), 1.0)
    # In-dataset parent -> child edges; children never precede parents.
    idx = pd.Series(np.arange(len(ledger)), index=ledger["txid"].to_numpy())
    child, parent = [], []
    for c, prevs in enumerate(ledger["in_prev_txid"]):
        for pt in sorted(set(prevs)):          # sorted: set order varies between runs (hash randomisation)
            j = idx.get(pt)
            if j is not None:
                child.append(c)
                parent.append(j)
    child, parent = np.array(child, int), np.array(parent, int)
    if len(child):
        gap = 1.0 + rng.exponential(20.0, len(child))
        # Relax along parent->child edges until nothing moves (each pass settles at least one more chain level).
        for _ in range(100_000):
            want = np.maximum(t[parent] + np.minimum(gap, np.maximum(block_t[child] - 0.5 - t[parent], 0.05)), t[parent] + 0.05)
            new = t.copy()
            np.maximum.at(new, child, want)
            if np.array_equal(new, t):
                break
            t = new
        else:
            raise RuntimeError("creation-time causality did not converge")
    return t


def simulate(ledger: pd.DataFrame, entities: pd.DataFrame, space: ShadowSpace, seed: np.random.SeedSequence,
             coverage: float | None = None, path_mix: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """ledger: needs txid, block_time, block_height, feerate, in_prev_txid, entity.
    entities: entity, role ('user'|'service'), path (None = draw from the configured mix).
    Returns (observable network fields per tx, per-tx network truth, ip_truth, realised stats, announcement log)."""
    s_nodes, s_time, s_race, s_port = seed.spawn(4)
    rng = np.random.default_rng(s_nodes)
    coverage = NET["sensor_peer_coverage"] if coverage is None else coverage
    summary = json.loads(C.BITNODES_SUMMARY.read_text())
    ns = summary["net_share"]
    v6_share = ns.get("ipv6", 0) / (ns.get("ipv6", 0) + ns.get("ipv4", 1))

    t_create = creation_times(ledger, np.random.default_rng(s_time))
    t0, t1 = float(t_create.min()) - 3600, float(t_create.max()) + 3600
    nodes = Nodes(space, rng, t0, t1, v6_share, summary["clearnet_port_top"])
    K = NET["sensors"]

    # --- infrastructure -------------------------------------------------------------------
    sensors = [nodes.add("sensor", space.pick_entry("hosting", NET["sensor_country"]), True, v6=False) for _ in range(K)]
    R = NET["relay_pool_size"]
    relays = np.array([nodes.add("relay", space.pick_entry(), True) for _ in range(R)])
    rel_conn = rng.random((R, K)) < coverage                       # sensor k connected to relay r
    rel_dir_in = rng.random((R, K)) < 0.10                          # [assume] relay dialled the sensor (inbound to sensor)
    conn_by_sensor = [np.flatnonzero(rel_conn[:, k]) for k in range(K)]
    L = NET["light_wallet_servers"]
    lws = np.array([nodes.add("light_wallet_server", space.pick_entry("hosting"), True) for _ in range(L)])
    zipf = 1.0 / np.arange(1, L + 1) ** NET["light_wallet_zipf_a"]
    zipf /= zipf.sum()

    # --- entity nodes -----------------------------------------------------------------------
    mix = path_mix or NET["user_broadcast_path_mix"]
    ent = entities.set_index("entity")
    path = ent["path"].copy() if "path" in ent else pd.Series(None, index=ent.index, dtype=object)
    users = ent.index[(ent["role"] == "user") & path.isna()]
    path.loc[users] = rng.choice(list(mix.keys()), size=len(users), p=list(mix.values()))
    path.loc[(ent["role"] == "service") & path.isna()] = "service_infra"
    ent_nodes: dict[int, np.ndarray] = {}
    for e, pth in path.items():
        if pth == "own_node_clearnet":
            ent_nodes[e] = np.array([nodes.add("home", space.pick_entry("residential"),
                                               bool(rng.random() < NET["listening_share_own_node"]),
                                               entity=e, churn=True)])
        elif pth == "own_node_tor":
            ent_nodes[e] = np.array([nodes.add("tor_client", None, False, hidden=True, entity=e)])
        elif pth == "light_wallet_server":
            ent_nodes[e] = np.array([lws[rng.choice(L, p=zipf)]])
        elif pth in ("service_infra", "vps"):
            entry = space.pick_entry("hosting")
            first = nodes.add("service" if pth == "service_infra" else "vps", entry, True, entity=e)
            k = rng.integers(NET["service_ips_min_max"][0], NET["service_ips_min_max"][1] + 1) if pth == "service_infra" else 1
            near = nodes.epoch_ip[first][0]
            more = [nodes.add("service", entry, True, entity=e, near=near, v6=":" in near) for _ in range(k - 1)]
            ent_nodes[e] = np.array([first] + more)
        elif pth == "proxy_pool":
            ent_nodes[e] = np.array([-2])            # resolved per tx to a shared proxy node below
        else:
            raise ValueError(f"unknown path {pth}")
    proxies = np.array([nodes.add("proxy", space.pick_entry("hosting"), False) for _ in range(40)])

    # --- per-tx origin -----------------------------------------------------------------------
    rr = np.random.default_rng(s_race)
    n = len(ledger)
    e_of = ledger["entity"].to_numpy()
    origin = np.empty(n, int)
    for e in np.unique(e_of):
        m = np.flatnonzero(e_of == e)
        cand = ent_nodes[e]
        origin[m] = proxies[rr.integers(0, len(proxies), len(m))] if cand[0] == -2 else cand[rr.integers(0, len(cand), len(m))]
    tx_path = path.reindex(e_of).to_numpy()
    kind = np.asarray(nodes.kind, dtype=object)
    hidden = np.asarray(nodes.hidden)[origin]
    lag = np.where(tx_path == "light_wallet_server", rr.lognormal(np.log(0.3), 0.5, n), 0.0)
    lag += np.where(hidden, rr.lognormal(np.log(0.8), 0.5, n), 0.0)       # [assume] Tor circuit latency
    t_net = t_create + lag

    def lat(shape):
        return rr.lognormal(np.log(NET["hop_latency_ms_median"] / 1000.0), NET["hop_latency_sigma"], shape)

    D, H = NET["peers_per_node"], 4
    A = NET["announcements_per_tx"]
    OUT_S, IN_S = NET["trickle_mean_s_outbound"], NET["trickle_mean_s_inbound"]
    listening_all = np.asarray(nodes.listening)

    def race(lo: int, hi: int):
        """First announcements to reach the sensors for txs lo..hi-1 (chunked to bound memory)."""
        o, tn, hid = origin[lo:hi], t_net[lo:hi], hidden[lo:hi]
        m = hi - lo
        day = np.floor(tn / DAY).astype(np.int64)
        key = o.astype(np.int64) * 100000 + day
        peers = relays[(_splitmix(key[:, None] * 64 + np.arange(D)[None, :]) % np.uint64(R)).astype(int)]  # stable entry nodes per (node, day)
        tor = np.flatnonzero(hid)
        peers[tor] = relays[rr.integers(0, R, (len(tor), D))]                   # Tor: fresh circuits each time
        t_first = tn[:, None] + rr.exponential(OUT_S, (m, D)) + lat((m, D))
        Tm, Nm, Km, Dm = [], [], [], []                                          # candidate time, node, sensor, dir_in
        peer_idx = peers - relays[0]
        listening = listening_all[o]
        is_relay_like = np.isin(kind[o], ["service", "light_wallet_server", "vps"])
        for k in range(K):
            c = rel_conn[peer_idx, k]
            din = rel_dir_in[peer_idx, k]
            tk = np.where(c, t_first + rr.exponential(np.where(din, OUT_S, IN_S)) + lat((m, D)), np.inf)
            Tm.append(tk); Nm.append(peers); Km.append(np.full((m, D), k)); Dm.append(din)
            # second hop: a random first-hop peer forwards to a relay that is connected to sensor k
            j = rr.integers(0, D, (m, H))
            r2 = conn_by_sensor[k][rr.integers(0, len(conn_by_sensor[k]), (m, H))]
            d2 = rel_dir_in[r2, k]
            t2 = (np.take_along_axis(t_first, j, 1) + rr.exponential(OUT_S, (m, H)) + lat((m, H))
                  + rr.exponential(np.where(d2, OUT_S, IN_S)) + lat((m, H)))
            Tm.append(t2); Nm.append(relays[r2]); Km.append(np.full((m, H), k)); Dm.append(d2)
            # direct origin links (fixed per node & sensor)
            node_u = _u01(o.astype(np.int64) * 16 + k)
            node_v = _u01(o.astype(np.int64) * 16 + 8 + k)
            sensor_dials = listening & ~hid & (node_u < coverage)
            # [assume] well-connected infrastructure (services, light-wallet servers) dials sensors more often
            origin_dials = (~hid & ~is_relay_like & (node_v < NET["p_origin_dials_sensor"])) |                 (~hid & is_relay_like & (node_v < 0.10))
            t_out = np.where(sensor_dials, tn + rr.exponential(IN_S, m) + lat(m), np.inf)
            t_in = np.where(origin_dials, tn + rr.exponential(OUT_S, m) + lat(m), np.inf)
            Tm.append(np.minimum(t_out, t_in)[:, None]); Nm.append(o[:, None]); Km.append(np.full((m, 1), k))
            Dm.append((t_in < t_out)[:, None])
        T, N, Kk, Din = (np.concatenate(a, axis=1) for a in (Tm, Nm, Km, Dm))
        rows_l = np.arange(m)
        w = np.argmin(T, axis=1)
        win = (T[rows_l, w], N[rows_l, w], Kk[rows_l, w], Din[rows_l, w].astype(bool))
        M = min(T.shape[1], 3 * A)
        part = np.argpartition(T, M - 1, axis=1)[:, :M]
        tt = np.take_along_axis(T, part, 1)
        order = np.argsort(tt, axis=1)
        part, tt = np.take_along_axis(part, order, 1), np.take_along_axis(tt, order, 1)
        a = pd.DataFrame(dict(row=np.repeat(rows_l + lo, M), t=tt.ravel(), node=np.take_along_axis(N, part, 1).ravel(),
                              sensor=np.take_along_axis(Kk, part, 1).ravel(),
                              inbound=np.take_along_axis(Din, part, 1).ravel().astype(bool)))
        a = a[np.isfinite(a["t"])].drop_duplicates(["row", "node", "sensor"])
        a["rank"] = a.groupby("row").cumcount()
        return win, a[a["rank"] < A]

    CHUNK = 75_000
    chunks = [race(i, min(i + CHUNK, n)) for i in range(0, n, CHUNK)]
    t_win, src, sens, d_in = (np.concatenate([c[0][i] for c in chunks]) for i in range(4))
    ann = pd.concat([c[1] for c in chunks], ignore_index=True)
    del chunks
    assert np.isfinite(t_win).all(), "some txs never reached a sensor"
    skew = np.random.default_rng(s_port).normal(0, NET["sensor_clock_skew_ms_sd"] / 1000.0, K)
    t_obs = t_win + skew[sens] + rr.uniform(0, 0.001, n)

    # --- ports -------------------------------------------------------------------------------------
    lo_s, hi_s = NET["sensor_inbound_ephemeral"]
    os_lo = np.array([v[1] for v in NET["peer_ephemeral_mix"].values()])
    os_hi = np.array([v[2] for v in NET["peer_ephemeral_mix"].values()])
    node_os, node_lport = np.asarray(nodes.os), np.asarray(nodes.listen_port)

    def ports(src_n, sens_k, inbound, t):
        """Direction-aware ports; ephemeral ports are sticky per (peer, sensor) TCP session."""
        lid = src_n.astype(np.int64) * 8 + sens_k
        L_s = -np.log(np.maximum(_u01(lid * 3 + 1), 1e-12)) * NET["session_mean_days"] * DAY + 600.0
        session = np.floor((t + _u01(lid * 3 + 2) * L_s) / L_s).astype(np.int64)
        h = _splitmix(lid * 1_000_003 + session)
        po = node_os[src_n]
        eph_peer = os_lo[po] + (h % (os_hi[po] - os_lo[po] + 1).astype(np.uint64)).astype(int)
        eph_sensor = lo_s + (h % np.uint64(hi_s - lo_s + 1)).astype(int)
        return np.where(inbound, eph_peer, node_lport[src_n]), np.where(inbound, 8333, eph_sensor)

    src_port, dst_port = ports(src, sens, d_in, t_obs)

    # --- announcement log: the first A announcements of each tx across all sensors -------------------
    ann = ann.reset_index(drop=True)
    a_row = ann["row"].to_numpy()
    a_sens = ann["sensor"].to_numpy()
    a_t = ann["t"].to_numpy() + skew[a_sens] + rr.uniform(0, 0.001, len(ann))
    first = ann["rank"].to_numpy() == 0
    a_t[first] = t_obs[a_row[first]]                       # rank 0 is exactly the first-seen record
    a_sp, a_dp = ports(ann["node"].to_numpy(), a_sens, ann["inbound"].to_numpy(), a_t)
    a_entry = np.asarray(nodes.entry)[ann["node"].to_numpy()]
    announcements = pd.DataFrame(dict(
        txid=ledger["txid"].to_numpy()[a_row],
        timestamp=pd.to_datetime(np.round(a_t * 1000).astype(np.int64), unit="ms", utc=True),
        src_ip=nodes.ip_at(ann["node"].to_numpy(), a_t), src_port=a_sp,
        dst_ip=nodes.ip_at(np.asarray(sensors)[a_sens], a_t), dst_port=a_dp,
        geo_country=space.pool["country"].to_numpy()[a_entry], asn=space.pool["asn"].to_numpy()[a_entry],
        rank=ann["rank"].to_numpy(),
    )).sort_values(["timestamp", "txid", "rank"]).reset_index(drop=True)

    src_ip = nodes.ip_at(src, t_obs)
    dst_ip = nodes.ip_at(np.asarray(sensors)[sens], t_obs)
    entry = np.asarray(nodes.entry)[src]
    pool = space.pool
    obs = pd.DataFrame(dict(
        txid=ledger["txid"].to_numpy(),
        timestamp=pd.to_datetime(np.round(t_obs * 1000).astype(np.int64), unit="ms", utc=True),
        src_ip=src_ip, src_port=src_port, dst_ip=dst_ip, dst_port=dst_port,
        geo_country=pool["country"].to_numpy()[entry], asn=pool["asn"].to_numpy()[entry],
    ))
    origin_ip = np.where(hidden, "", nodes.ip_at(origin, t_net))
    truth = pd.DataFrame(dict(
        txid=ledger["txid"].to_numpy(),
        t_create=pd.to_datetime(np.round(t_create * 1000).astype(np.int64), unit="ms", utc=True),
        broadcast_path=tx_path, origin_node=origin, origin_kind=kind[origin], origin_ip=origin_ip,
        first_seen_node=src, first_seen_kind=kind[src], first_seen_is_origin=src == origin,
        sensor_id=sens, conn_direction=np.where(d_in, "inbound_to_sensor", "sensor_outbound"),
        net_delay_s=t_win - t_create,
    ))
    ip_rows = []
    nt = nodes.table()
    for nid in range(len(nodes.kind)):
        if nodes.hidden[nid]:
            continue
        e = pool.iloc[nodes.entry[nid]]
        for i, ip in enumerate(nodes.epoch_ip[nid]):
            ip_rows.append((ip, nid, nodes.kind[nid], nodes.entity[nid], float(nodes.epoch_t[nid][i]),
                            int(e["asn"]), e["org"], e["country"], e["asn_class"]))
    ip_truth = pd.DataFrame(ip_rows, columns=["ip", "node", "node_kind", "entity", "valid_from", "asn", "org",
                                              "country", "asn_class"])
    stats = {
        "sensor_peer_coverage": coverage,
        "user_broadcast_path_mix": mix,
        "realised_ip_noise_by_path": (1 - truth.groupby("broadcast_path")["first_seen_is_origin"].mean()).round(4).to_dict(),
        "realised_ip_noise_overall": round(float(1 - truth["first_seen_is_origin"].mean()), 4),
        "broadcast_path_share": truth["broadcast_path"].value_counts(normalize=True).round(4).to_dict(),
        "net_delay_s_quantiles": truth["net_delay_s"].quantile([.05, .25, .5, .75, .95]).round(3).to_dict(),
        "conn_direction_share": truth["conn_direction"].value_counts(normalize=True).round(4).to_dict(),
        "dst_port_8333_share": round(float((obs["dst_port"] == 8333).mean()), 4),
        "src_is_ipv6_share": round(float(obs["src_ip"].str.contains(":").mean()), 4),
        "nodes_by_kind": nt["kind"].value_counts().to_dict(),
    }
    stats["announcements"] = int(len(announcements))
    return obs, truth, ip_truth, stats, announcements
