"""Track M: real modern mainnet background (contiguous window, 2026) + injected campaigns with exact truth + simulated
network layer.

Real background     : unchanged real transactions (label 'unknown'; OP_RETURN data carriers tagged).
Injected illicit    : replays of REAL ransomware money flows traced from Ransomwhere seeds (victim payments ->
                      operator spends -> layering), replays of real Elliptic illicit flows (both re-keyed, re-priced
                      at USD parity, script types modernised, time-compressed into the window), scripted ransomware
                      campaigns (exact pattern truth; ransom sizes drawn from real Ransomwhere payments), smurfing
                      launderers.
Injected licit cover: exchange hot-wallet peeling withdrawals, equal-amount batch payouts, retail payments,
                      Whirlpool-like CoinJoin rounds. Built with the same machinery as the illicit injections, so
                      'injected' carries no information about 'illicit'.
Injected transactions take their fee rate from real payment transactions in the same block, their script types
from the real window's payment mix, and cash out to real exchange deposit addresses seen in the window.
No real address is ever labelled illicit.
"""
from __future__ import annotations

import json
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd

from . import cluster, config as C, export, network
from .addresses import AddressFactory, txid_from, vsize
from .build_track_r import elliptic_labels
from .normalize import normalize

INJ_ENTITY_BASE = 10_000_000
DUST = 1_000                      # [assume] minimum injected output value (sats), above every dust limit
PREMIX_BUF = 300                  # [assume] Whirlpool premix miner-fee buffer per input (sats) -> ~3 sat/vB mix
INJ = {
    # counts below are per 6 h of window and scale linearly with the window length
    "ransomware_campaigns": 6, "victims": (8, 30),     # ransom USD: empirical Ransomwhere payments >= $50 [src]
    "replay_components": 25, "replay_size": (3, 80),
    "rw_replay_components": 4, "rw_replay_size": (3, 150), "rw_hop2_min_share": 0.75, "rw_hop2_max_in": 20,
    "max_replay_span_frac": 0.8,                # real flows are time-compressed to <= 80% of the window
    "launderers": 4, "smurf_split": (10, 30), "smurf_threshold_usd": 10_000,              # [assume] structuring
    "exchange_peel_chains": 6, "exchange_peel_len": (15, 40),
    "batch_payouts": 10, "batch_outputs": (10, 60),
    "retail_users": 800, "merchants": 40,
    "whirlpool_pools_btc": [0.01, 0.05], "whirlpool_licit_rounds": 6,                    # [assume; verify pools]
    "privacy_users": 20,                  # [assume] Whirlpool (Ashigaru) is a small community in 2026
    "mixer_fraction": 0.3,                      # share of ransomware layering routed through CoinJoin [assume]
    "peel_len": (5, 20), "peel_frac": (0.01, 0.10),
    "cashout_to_real_exchange": 0.6,            # [assume] share of cash-outs that land on real deposit addresses
    "seed_coverage": 0.7,                       # [assume] campaigns with >= 1 known seed address
    "illicit_opsec": {"vps": 0.35, "proxy_pool": 0.35, "own_node_tor": 0.30},   # [assume] per-operator OPSEC
    "retail_addr_reuse": 0.30,            # [cal] ~33% of real payment txs in the window reuse an input address
    "change_type_mismatch": 0.10,         # [assume]
    "payjoins": 40,                       # [assume] ownership-heuristic breakers
}


class World:
    """Holds injected wallets, UTXOs, transactions and truth while the scenarios run."""

    def __init__(self, rng, real: pd.DataFrame, price_usd: float, deposit_addrs: np.ndarray, ransom_usd: np.ndarray):
        self.rng = rng
        self.ransom_usd = ransom_usd
        self.af = AddressFactory(rng)
        self.price = price_usd
        self.deposit_addrs = deposit_addrs
        pay = real[~real["data_carrier"]]
        types = pd.Series(np.concatenate(pay["out_type"].to_numpy()))
        types = types[types != "P2A"].value_counts(normalize=True)
        self.type_vals, self.type_p = types.index.to_numpy(), types.to_numpy()
        self.blocks = real.groupby("block_height")["block_time"].min().sort_index()
        self.block_t = network.epoch_s(self.blocks.reset_index()["block_time"])
        self.fee_by_block = {h: g.to_numpy() for h, g in pay.groupby("block_height")["feerate"]}
        self.fee_all = pay["feerate"].to_numpy()
        # templates for the empirical shape bootstrap: real payment txs of the window (no P2A, modest size)
        tp = real[(real["scenario"] == "real_background") & (real["n_in"] <= 6) & (real["n_out"] <= 8)]
        tp = tp[~tp["in_type"].map(lambda x: "P2A" in x) & ~tp["out_type"].map(lambda x: "P2A" in x)]
        self.templates = list(zip(tp["in_addr"], tp["in_sats"], tp["in_type"], tp["out_addr"], tp["out_sats"],
                                  tp["out_type"], tp["fee_sats"], tp["txid"]))
        self.templates_1in = [t for t in self.templates if len(t[0]) == 1]
        tb = real[(real["scenario"] == "real_background") & (real["n_in"] <= 10) & (real["n_out"] >= 10) & (real["n_out"] <= 80)]
        tb = tb[~tb["in_type"].map(lambda x: "P2A" in x) & ~tb["out_type"].map(lambda x: "P2A" in x)]
        self.templates_batch = list(zip(tb["in_addr"], tb["in_sats"], tb["in_type"], tb["out_addr"], tb["out_sats"],
                                        tb["out_type"], tb["fee_sats"], tb["txid"]))
        # real service 1-in-2-out spends: withdrawal (smaller output) and remaining balance (larger output), in USD
        sv = real[real["is_service"] & (real["scenario"] == "real_background") & (real["n_in"] == 1) & (real["n_out"] == 2)]
        self.svc_pay_usd = sv["out_sats"].map(min).to_numpy() / 1e8 * price_usd
        self.svc_bal_usd = sv["out_sats"].map(max).to_numpy() / 1e8 * price_usd
        self.t0, self.t1 = float(self.block_t[0]) - 1800, float(self.block_t[-1]) - 60
        self.W = self.t1 - self.t0
        self.scale = self.W / (6 * 3600.0)                                  # injected counts are per 6 h
        # activity clock: block times of real payment txs (sorted) -> injected times follow the real daily rhythm
        self.act_t = np.sort(network.epoch_s(pay["block_time"]))
        self.txs: list[dict] = []
        self.entities: dict[int, dict] = {}
        self.addr_owner: dict[str, int] = {}
        self.patterns: list[dict] = []
        self.seeds: list[tuple[str, str]] = []
        self.utxo_value: dict[tuple[str, int], int] = {}
        self._next_entity = INJ_ENTITY_BASE

    # --- entities & addresses -----------------------------------------------------------------------------
    def entity(self, etype: str, illicit: bool | None, role: str = "user", path: str | None = None, campaign: str | None = None,
               quality: str = "exact") -> int:
        e = self._next_entity
        self._next_entity += 1
        self.entities[e] = dict(entity=e, entity_type=etype, is_illicit=illicit, role=role, path=path,
                                campaign_id=campaign, truth_quality=quality,
                                pref_type=str(self.rng.choice(self.type_vals, p=self.type_p)))
        return e

    def opsec(self) -> str:
        o = INJ["illicit_opsec"]
        return str(self.rng.choice(list(o), p=list(o.values())))

    def addr(self, e: int, stype: str | None = None) -> str:
        a = self.af.new(stype or self.entities[e]["pref_type"])
        self.addr_owner[a] = e
        return a

    def usd(self, usd: float) -> int:
        return int(round(usd / self.price * 1e8))

    def fiat_round(self, usd: float) -> int:
        step = self.rng.choice([1, 5, 10, 50, 100], p=[0.15, 0.2, 0.3, 0.2, 0.15])
        return self.usd(max(step, round(usd / step) * step))

    # --- time / blocks / fees ------------------------------------------------------------------------------
    def block_for(self, t: float) -> int:
        i = int(np.clip(np.searchsorted(self.block_t, t + 1.0), 0, len(self.block_t) - 1))
        return int(self.blocks.index[i])

    def feerate(self, height: int) -> float:
        pool = self.fee_by_block.get(height, self.fee_all)
        return float(max(self.rng.choice(pool), 0.1))

    # --- transactions ----------------------------------------------------------------------------------------
    def fund(self, e: int, sats: int, stype: str | None = None) -> tuple:
        """A pre-window UTXO (its funding transaction is outside the dataset, as for many real inputs)."""
        a = self.addr(e, stype)
        u = (txid_from(self.rng), int(self.rng.integers(0, 3)), a, int(sats), self.af_type(a))
        self.utxo_value[(u[0], u[1])] = u[3]
        return u

    @staticmethod
    def af_type(a: str) -> str:
        from .addresses import validate_address
        return validate_address(a)

    def tx(self, t: float, initiator: int, inputs: list[tuple], outputs: list[tuple[str, int]], scenario: str,
           change_to: str | None = None, shuffle: bool = True, fee_sats: int | None = None, **truth) -> tuple[dict, list]:
        """inputs: UTXO tuples (txid, vout, addr, sats, type). outputs: (addr, sats); change (if any) is added to
        `change_to` so that sum(in) = sum(out) + fee. Returns (tx record, list of created UTXO tuples)."""
        h = self.block_for(t)
        in_types = [u[4] for u in inputs]
        outs = [(a, int(v)) for a, v in outputs]
        out_types = [self.af_type(a) for a, _ in outs] + ([self.af_type(change_to)] if change_to else [])
        total_in = sum(u[3] for u in inputs)
        fee = fee_sats if fee_sats is not None else int(np.ceil(self.feerate(h) * vsize(in_types, out_types)))
        if change_to:
            change = total_in - sum(v for _, v in outs) - fee
            if change >= DUST:
                outs.append((change_to, change))
            else:
                fee = total_in - sum(v for _, v in outs)
        else:
            fee = total_in - sum(v for _, v in outs)
        if fee < 0 or any(v < DUST for _, v in outs):
            raise ValueError("unbalanced injected tx")
        if shuffle:
            order = self.rng.permutation(len(outs))
            outs = [outs[i] for i in order]
        txid = txid_from(self.rng)
        created = [(txid, i, a, v, self.af_type(a)) for i, (a, v) in enumerate(outs)]
        for u in created:
            self.utxo_value[(txid, u[1])] = u[3]
        rec = dict(
            txid=txid, block_height=h, block_time=self.blocks.loc[h], t_intent=t,
            in_addr=[u[2] for u in inputs], in_sats=[u[3] for u in inputs], in_type=in_types,
            in_prev_txid=[u[0] for u in inputs], in_prev_vout=[u[1] for u in inputs],
            out_addr=[a for a, _ in outs], out_sats=[v for _, v in outs], out_type=[self.af_type(a) for a, _ in outs],
            out_vout=list(range(len(outs))), fee_sats=int(fee), dropped_opreturn=0,
            entity=initiator, scenario=scenario, origin="injected", **truth,
        )
        rec["vsize"] = vsize(rec["in_type"], rec["out_type"])
        rec["script_type"] = rec["out_type"][int(np.argmax(rec["out_sats"]))]
        self.txs.append(rec)
        return rec, created

    def bootstrap_tx(self, e: int, t: float, payee, scenario: str, spend: tuple | None = None,
                     fixed: tuple[str, int] | None = None, pool: list | None = None, **truth):
        """Licit cover transaction that copies the structure of a random REAL payment from the window: input/output
        counts, per-position script types, value proportions, which outputs are round (kept round), where the change
        sits and whether it reuses an input address. Values are rescaled/jittered so no real tx is duplicated.
        With `spend`, the template must have one input and that UTXO is spent (the entity's next payment).
        With `fixed=(address, sats)`, the template's first payment output becomes that exact payment (e.g. a ransom).
        Returns (tx record, change UTXO or None)."""
        rng = self.rng
        pool = pool or (self.templates_1in if spend is not None else self.templates)
        for _ in range(50):
            ia, iv, it, oa, ov, ot, fee, ttx = pool[int(rng.integers(0, len(pool)))]
            reuse = [j for j, a in enumerate(oa) if a in set(ia)]
            if reuse:
                ch = reuse[0]
            elif len(oa) == 1:
                ch = None
            else:
                tz = [len(str(v)) - len(str(v).rstrip("0")) for v in ov]
                ch = min(range(len(ov)), key=lambda j: (tz[j], -ov[j]))
            j0 = None
            if fixed is not None:
                cand = [j for j in range(len(oa)) if j != ch]
                if not cand or ch is None:
                    continue
                j0 = cand[0]
                f = fixed[1] / ov[j0]
                if not 0.05 < f < 20:
                    continue
            elif spend is not None:
                f = spend[3] / iv[0]
                if not 0.2 < f < 5:
                    continue
            else:
                f = float(np.exp(rng.normal(0, 0.08)))
            if spend is not None:
                ins = [spend]
            else:
                amap: dict[str, tuple] = {}
                ins = []
                for a, v, ty in zip(ia, iv, it):
                    if a not in amap:
                        amap[a] = self.addr(e, ty)
                    na = amap[a]
                    u = (txid_from(rng), int(rng.integers(0, 3)), na, max(int(round(v * f)), DUST), ty)
                    ins.append(u)
            total_in = sum(u[3] for u in ins)
            fee_new = int(fee)            # exact template fee: real fees cluster on exact values (fixed-rate bots)
            outs = []
            for j, (a, v, ty) in enumerate(zip(oa, ov, ot)):
                if j == ch:
                    outs.append(None)
                    continue
                if j == j0:
                    outs.append(fixed)
                    continue
                z = len(str(v)) - len(str(v).rstrip("0"))
                nv = max(10 ** z, int(round(v * f / 10 ** z)) * 10 ** z) if z >= 3 else int(round(v * f))
                outs.append((payee(ty), nv))
            paid = sum(o[1] for o in outs if o is not None)
            if ch is None:
                if len(outs) != 1:
                    continue
                outs = [(outs[0][0], total_in - fee_new)]
                if outs[0][1] < DUST:
                    continue
                chaddr = None
            else:
                cv = total_in - paid - fee_new
                if cv < DUST or any(o is not None and o[1] < DUST for o in outs):
                    continue
                chaddr = ins[0][2] if reuse else self.addr(e, ot[ch])
                outs[ch] = (chaddr, cv)
            rec, created = self.tx(t, e, ins, outs, scenario, shuffle=False, fee_sats=total_in - sum(v for _, v in outs),
                                   template_of=ttx, **truth)
            change = next((u for u in created if chaddr is not None and u[2] == chaddr), None)
            return rec, change
        return None, None

    def change_addr(self, e: int, inputs) -> str:
        ent = self.entities[e]
        if ent["entity_type"] in ("retail", "ransomware_victim", "ransomware_victim_real") and self.rng.random() < INJ["retail_addr_reuse"]:
            return inputs[0][2]                                     # address reuse: change back to the input address
        stype = inputs[0][4] if self.rng.random() > INJ["change_type_mismatch"] else None
        return self.addr(e, stype if stype in ("P2PKH", "P2SH", "P2WPKH", "P2WSH", "P2TR") else None)

    def deposit(self) -> str:
        """Cash-out destination: a real exchange deposit address from the window, or a fresh synthetic one."""
        if len(self.deposit_addrs) and self.rng.random() < INJ["cashout_to_real_exchange"]:
            return str(self.rng.choice(self.deposit_addrs))
        return self.af.new(str(self.rng.choice(self.type_vals, p=self.type_p)))

    def pattern(self, ptype: str, recs: list[dict], campaign: str | None, **params) -> str:
        pid = f"P{len(self.patterns):04d}"
        for i, r in enumerate(recs):
            r["pattern_id"], r["hop_index"] = pid, i
        wallets = sorted({a for r in recs for a in r["in_addr"] if a in self.addr_owner})
        self.patterns.append(dict(pattern_id=pid, type=ptype, campaign_id=campaign, txids=[r["txid"] for r in recs],
                                  wallets=wallets, params_json=json.dumps(params)))
        return pid

    def rand_t(self, lo: float = 0.0, hi: float = 1.0) -> float:
        """A time between the lo and hi quantiles of real payment activity (follows the real daily rhythm)."""
        n = len(self.act_t) - 1
        a, b = int(lo * n), max(int(hi * n), int(lo * n) + 1)
        return float(np.clip(self.act_t[int(self.rng.integers(a, b))] + self.rng.uniform(-300, 300), self.t0, self.t1))

    def rand_t_ext(self, p_pre: float = 0.4, pre_frac: float = 0.5) -> float:
        """Start time for a multi-step campaign or chain. With probability p_pre it began BEFORE the window (up to
        pre_frac of a window earlier), so the capture cuts into campaigns already underway - as a real capture does.
        Transactions falling outside the window are dropped at the end (see build)."""
        if self.rng.random() < p_pre:
            return float(self.t0 - self.rng.uniform(0, pre_frac * self.W))
        return self.rand_t(0.0, 1.0)

    def cnt(self, key: str) -> int:
        return max(1, int(round(INJ[key] * self.scale)))


# --- scenarios -----------------------------------------------------------------------------------------------------

def peel_chain(w: World, e: int, utxo: tuple, t: float, n: int, scenario: str, campaign, pay_to, gap_s=(120, 900),
               frac=(0.01, 0.10), fiat_round=False, amount_fn=None) -> tuple[list, tuple]:
    recs, cur = [], utxo
    for _ in range(n):
        if cur[3] < 20 * DUST:
            break
        if amount_fn is not None:
            amt = amount_fn()
        else:
            amt = w.fiat_round(cur[3] * w.rng.uniform(*frac) / 1e8 * w.price) if fiat_round else int(cur[3] * w.rng.uniform(*frac))
        amt = int(np.clip(amt, DUST, cur[3] // 2))
        nxt = w.addr(e, cur[4] if cur[4] != "P2A" else None)
        r, created = w.tx(t, e, [cur], [(pay_to(), amt)], scenario, change_to=nxt, campaign_id=campaign)
        recs.append(r)
        cur = next(u for u in created if u[2] == nxt) if any(u[2] == nxt for u in created) else None
        if cur is None:
            break
        t += w.rng.uniform(*gap_s)
    return recs, cur


def whirlpool_round(w: World, coord: int, premixes: list[tuple[int, tuple]], denom: int, t: float, campaign=None):
    """5 inputs from 5 different entities -> 5 equal outputs (denomination) to fresh P2WPKH addresses of their owners.
    Broadcast by the coordinator. Fee = sum(premix) - 5 * denom."""
    ins = [u for _, u in premixes]
    outs = [(w.addr(o, "P2WPKH"), denom) for o, _ in premixes]
    r, created = w.tx(t, coord, ins, outs, "coinjoin_whirlpool", shuffle=True, fee_sats=sum(u[3] for u in ins) - 5 * denom,
                      campaign_id=campaign)
    r["is_coinjoin"] = True
    return r, created


def tx0(w: World, e: int, utxo: tuple, denom: int, k: int, t: float, coord_fee_addr: str, scenario: str, campaign=None):
    """Whirlpool premix: split into k premix outputs (denom + mining-fee buffer), pay a pool fee, keep change."""
    buf = PREMIX_BUF
    outs = [(w.addr(e, "P2WPKH"), denom + buf) for _ in range(k)] + [(coord_fee_addr, max(DUST, int(denom * 0.05)))]
    r, created = w.tx(t, e, [utxo], outs, scenario, change_to=w.change_addr(e, [utxo]), campaign_id=campaign)
    prem = [u for u in created if u[3] == denom + buf and w.addr_owner.get(u[2]) == e]
    return r, prem[:k]


def run_scenarios(w: World, replay_src: pd.DataFrame, price_hist: pd.Series, rw_flows: pd.DataFrame,
                  rw_seeds: pd.DataFrame) -> None:
    rng = w.rng
    coord = w.entity("coinjoin_coordinator", False, role="service", path="own_node_tor")
    coord_fee = w.addr(coord)
    denoms = [int(d * 1e8) for d in INJ["whirlpool_pools_btc"]]
    # privacy users hold premix UTXOs ready for rounds
    privacy = [w.entity("privacy_user", False) for _ in range(w.cnt("privacy_users"))]
    premix_pool = {d: [] for d in denoms}
    for p in privacy:
        d = int(rng.choice(denoms))
        big = w.fund(p, d * 6 + w.usd(rng.uniform(50, 500)))
        _, prem = tx0(w, p, big, d, 4, w.rand_t(0.0, 0.3), coord_fee, "privacy_premix")
        premix_pool[d] += [(p, u) for u in prem]

    def pull_premix(d: int, n: int, exclude: set) -> list:
        out, keep = [], []
        rng.shuffle(premix_pool[d])
        for o, u in premix_pool[d]:
            if len(out) < n and o not in exclude and o not in {x for x, _ in out}:
                out.append((o, u))
            else:
                keep.append((o, u))
        premix_pool[d] = keep
        return out

    # licit-only CoinJoin rounds
    for _ in range(w.cnt("whirlpool_licit_rounds")):
        d = int(rng.choice(denoms))
        parts = pull_premix(d, 5, set())
        if len(parts) == 5:
            r, _ = whirlpool_round(w, coord, parts, d, w.rand_t(0.35, 0.9))
            w.pattern("coinjoin_whirlpool", [r], None, denom=d, illicit_participants=0)

    # merchants & retail (licit cover)
    merchants = [w.entity("merchant", False, role="service" if rng.random() < 0.3 else "user") for _ in range(w.cnt("merchants"))]
    merchant_addrs = {m: [w.addr(m) for _ in range(20)] for m in merchants}
    def payee(stype: str) -> str:
        if rng.random() < 0.3:
            return w.deposit()
        m = int(rng.choice(merchants))
        return str(rng.choice(merchant_addrs[m])) if rng.random() < 0.5 else w.addr(m, stype)

    # Retail cover = empirical shape bootstrap of real payments in the same window (see World.bootstrap_tx)
    for _ in range(w.cnt("retail_users")):
        u = w.entity("retail", False)
        t = w.rand_t()
        change = None
        for _k in range(int(rng.choice([1, 1, 1, 2, 3]))):
            rec, change = w.bootstrap_tx(u, t, payee, "retail_payment", spend=change)
            if change is None:
                break
            t += rng.exponential(1800 * min(w.scale, 4))

    # PayJoin (BIP-78): the receiver adds an input, so common-input ownership wrongly merges payer and merchant
    for _ in range(w.cnt("payjoins")):
        u = w.entity("retail", False)
        m = int(rng.choice(merchants))
        pay = w.fiat_round(float(np.exp(rng.normal(np.log(120), 1.0))))
        s_in = w.fund(u, int(pay * rng.uniform(1.2, 3.0)) + 5_000, "P2WPKH")
        m_in = w.fund(m, w.usd(float(np.exp(rng.normal(np.log(300), 1.0)))) + 5_000, "P2WPKH")
        r, _ = w.tx(w.rand_t(), u, [s_in, m_in], [(w.addr(m, "P2WPKH"), pay + m_in[3])], "payjoin_payment",
                    change_to=w.addr(u, "P2WPKH"))
        r["is_payjoin"] = True

    # exchange hot-wallet peeling withdrawals (hard negative for the peeling detector)
    for i in range(w.cnt("exchange_peel_chains")):
        ex = w.entity("exchange", False, role="service")
        hot = w.fund(ex, w.usd(float(rng.choice(w.svc_bal_usd))) + 5_000_000, "P2WPKH")   # [cal] real service balances
        n = int(rng.integers(*INJ["exchange_peel_len"]))
        recs, _ = peel_chain(w, ex, hot, w.rand_t_ext(), n, "exchange_peel_withdrawal", None,
                             pay_to=lambda: w.af.new(str(rng.choice(w.type_vals, p=w.type_p))),
                             gap_s=(60, 600), amount_fn=lambda: w.usd(float(rng.choice(w.svc_pay_usd))))   # [cal]
        w.pattern("exchange_peel_negative", recs, None, length=len(recs))

    # equal-amount batch payouts (hard negative for the CoinJoin detector)
    for i in range(w.cnt("batch_payouts")):
        svc = w.entity(str(rng.choice(["mining_pool", "payroll_processor"])), False, role="service")
        k = int(rng.integers(*INJ["batch_outputs"]))
        each = w.fiat_round(rng.uniform(20, 400))
        src = [w.fund(svc, int(each * k * rng.uniform(0.6, 0.8)) + 50_000, "P2WPKH") for _ in range(2)]
        outs = [(w.af.new(str(rng.choice(w.type_vals, p=w.type_p))), each) for _ in range(k)]
        r, _ = w.tx(w.rand_t(), svc, src, outs, "batch_payout", change_to=w.addr(svc, "P2WPKH"))
        w.pattern("batch_payout_negative", [r], None, outputs=k)

    # real-shaped exchange batch withdrawals (licit cover, bootstrapped from real 10-80-output batches)
    if w.templates_batch:
        for i in range(w.cnt("batch_payouts")):
            ex = w.entity("exchange", False, role="service")
            w.bootstrap_tx(ex, w.rand_t(), payee, "exchange_batch_withdrawal", pool=w.templates_batch)

    # ransomware campaigns (scripted, exact truth)
    for c in range(w.cnt("ransomware_campaigns")):
        cid = f"RW{c:02d}"
        op = w.entity("ransomware_operator", True, path=w.opsec(), campaign=cid)
        t_start = w.rand_t_ext()
        ransoms = []
        pay_recs = []
        for v in range(int(rng.integers(*INJ["victims"]))):
            vic = w.entity("ransomware_victim", False)
            amt = w.usd(float(rng.choice(w.ransom_usd)))                     # real Ransomwhere payment size
            recv = w.addr(op)
            tv = t_start + rng.uniform(0, 0.12 * w.W)
            r, _ = w.bootstrap_tx(vic, tv, payee, "ransom_payment", fixed=(recv, amt), campaign_id=cid, pays_illicit=True)
            if r is None:                                                    # no fitting template: plain payment
                src = w.fund(vic, int(amt * rng.uniform(1.05, 1.6)) + 20_000)
                r, _ = w.tx(tv, vic, [src], [(recv, amt)], "ransom_payment", change_to=w.change_addr(vic, [src]),
                            campaign_id=cid, pays_illicit=True)
            i = r["out_addr"].index(recv)
            ransoms.append((r["txid"], i, recv, r["out_sats"][i], r["out_type"][i]))
            pay_recs.append(r)
        t = max(r["t_intent"] for r in pay_recs) + rng.uniform(0.03, 0.15) * w.W     # dormancy
        agg = w.addr(op)
        r_agg, created = w.tx(t, op, ransoms, [], "ransomware_aggregation", change_to=agg, campaign_id=cid)
        pot = created[0]
        recs = [r_agg]
        if rng.random() < INJ["mixer_fraction"]:
            d = denoms[int(np.argmin([abs(pot[3] / 6 - d) for d in denoms]))]
            k = int(min(4, max(1, pot[3] // (d * 2))))
            if pot[3] > k * (d + d * 0.06) + 50_000:
                r0, prem = tx0(w, op, pot, d, k, t + 600, coord_fee, "ransomware_premix", campaign=cid)
                recs.append(r0)
                pot = next((u for u in [(r0["txid"], i, a, v, ty) for i, (a, v, ty) in
                                        enumerate(zip(r0["out_addr"], r0["out_sats"], r0["out_type"]))]
                            if w.addr_owner.get(u[2]) == op and u[3] != d + PREMIX_BUF), None)
                for pu in prem:
                    parts = [(op, pu)] + pull_premix(d, 4, {op})
                    if len(parts) == 5:
                        rr, created = whirlpool_round(w, coord, parts, d, t + rng.uniform(1200, 3600), campaign=cid)
                        w.pattern("coinjoin_whirlpool", [rr], cid, denom=d, illicit_participants=1)
                        post = [u for u in created if w.addr_owner.get(u[2]) == op]
                        for u in post:
                            rc, _ = w.tx(rr["t_intent"] + rng.uniform(1800, 4000), op, [u], [], "ransomware_cashout",
                                         change_to=w.deposit(), campaign_id=cid)
                            recs.append(rc)
        if pot is not None and pot[3] > 30 * DUST:
            n = int(rng.integers(*INJ["peel_len"]))
            chain, rest = peel_chain(w, op, pot, t + rng.uniform(900, 2400), n, "ransomware_peel", cid,
                                     pay_to=w.deposit, frac=INJ["peel_frac"])
            w.pattern("peeling_chain", chain, cid, length=len(chain))
            recs += chain
            if rest is not None and rest[3] > 2 * DUST:
                rc, _ = w.tx(chain[-1]["t_intent"] + rng.uniform(300, 1200) if chain else t + 3000, op, [rest], [],
                             "ransomware_cashout", change_to=w.deposit(), campaign_id=cid)
                recs.append(rc)
        w.pattern("ransomware_lifecycle", pay_recs + recs, cid, victims=len(pay_recs))
        if rng.random() < INJ["seed_coverage"]:
            for a in rng.choice(sorted({a for r in pay_recs for a in r["out_addr"] if w.addr_owner.get(a) == op}),
                                size=int(rng.integers(1, 3)), replace=False):
                w.seeds.append((str(a), cid))

    # smurfing launderers
    for c in range(w.cnt("launderers")):
        cid = f"LD{c:02d}"
        ld = w.entity("launderer", True, path=w.opsec(), campaign=cid)
        k = int(rng.integers(*INJ["smurf_split"]))
        thr = w.usd(INJ["smurf_threshold_usd"])
        each = [int(thr * rng.uniform(0.90, 0.99)) for _ in range(k)]
        src = w.fund(ld, sum(each) + 200_000)
        t = w.rand_t_ext()
        mules = [w.addr(ld) for _ in range(k)]
        r_split, created = w.tx(t, ld, [src], list(zip(mules, each)), "smurf_split", change_to=w.addr(ld), campaign_id=cid)
        hops = []
        mid = []
        for u in created:
            if u[2] in mules:
                nxt = w.addr(ld)
                rh, cr = w.tx(t + rng.uniform(600, max(3600, 0.08 * w.W)), ld, [u], [], "smurf_hop", change_to=nxt,
                              campaign_id=cid)
                hops.append(rh)
                mid += cr
        merges = []
        rng.shuffle(mid)
        for grp in np.array_split(np.arange(len(mid)), int(rng.integers(1, 4))):
            if len(grp):
                rm, _ = w.tx(max(h["t_intent"] for h in hops) + rng.uniform(600, max(3600, 0.05 * w.W)), ld,
                             [mid[i] for i in grp], [],
                             "smurf_merge", change_to=w.deposit(), campaign_id=cid)
                merges.append(rm)
        w.pattern("smurfing", [r_split] + hops + merges, cid, split=k)
        if rng.random() < INJ["seed_coverage"]:
            w.seeds.append((mules[0], cid))

    replay_elliptic(w, replay_src, price_hist)
    replay_ransomwhere(w, rw_flows, rw_seeds, price_hist)


def replay_flow(w: World, part: pd.DataFrame, cid: str, price_hist: pd.Series, role_of, scen_of, actor_type: str,
                actor_quality: str, pattern_type: str, **pparams) -> list[dict]:
    """Transplant one connected REAL flow into the window: every real address is re-keyed (synthetic, modern script
    type), amounts are re-priced at USD parity, fees re-rated to the block's real fee market, and the real relative
    timing is kept (compressed to <= max_replay_span_frac of the window). Ownership inside the flow follows the real
    co-spending; `role_of(addr)` says whether an input address belongs to a victim (licit) or an actor (illicit)."""
    rng = w.rng
    part = part.sort_values(["block_height", "txid"])
    then = float(price_hist.asof(part["block_time"].iloc[0].tz_localize(None).normalize()))
    scale = then / w.price                                                 # USD parity
    bt = network.epoch_s(part["block_time"])
    span = bt[-1] - bt[0]
    max_span = INJ["max_replay_span_frac"] * w.W
    squeeze = min(1.0, max_span / span) if span > 0 else 1.0
    t_start = w.rand_t_ext()
    uf2 = cluster.UnionFind()
    for addrs in part["in_addr"]:
        uf2.find(addrs[0])
        for a in addrs[1:]:
            uf2.union(addrs[0], a)
    ent_of: dict[str, int] = {}
    path = w.opsec()
    addr_map: dict[str, str] = {}
    counterparty_type = actor_type.replace("_actor", "_counterparty").replace("_operator", "_counterparty")

    def new_type(old: str) -> str:
        if old == "P2SH" and rng.random() < 0.3:
            return "P2SH"
        return str(rng.choice(w.type_vals, p=w.type_p))

    def owner(a: str) -> int:
        r = uf2.find(a)
        if r not in ent_of:
            if role_of(a) == "victim":
                ent_of[r] = w.entity("ransomware_victim_real", False, campaign=cid, quality="inherited_cioh")
            else:
                ent_of[r] = w.entity(actor_type, True, path=path, campaign=cid, quality=actor_quality)
        return ent_of[r]

    spent_internal = {(p, v) for ps, vs in zip(part["in_prev_txid"], part["in_prev_vout"]) for p, v in zip(ps, vs)}
    new_outpoint: dict[tuple[str, int], tuple] = {}
    recs = []
    for row, t_real in zip(part.itertuples(index=False), bt):
        e = owner(row.in_addr[0])
        ins = []
        for a, v, ty, p, pv in zip(row.in_addr, row.in_sats, row.in_type, row.in_prev_txid, row.in_prev_vout):
            if (p, pv) in new_outpoint:
                ins.append(new_outpoint[(p, pv)])
            else:
                if a not in addr_map:
                    addr_map[a] = w.addr(owner(a), new_type(ty))
                na = addr_map[a]
                ins.append((txid_from(rng), int(rng.integers(0, 3)), na, max(int(round(v * scale)), 5 * DUST), w.af_type(na)))
        outs, weights = [], []
        for a, v, ty, vout in zip(row.out_addr, row.out_sats, row.out_type, row.out_vout):
            internal = (row.txid, vout) in spent_internal
            if not internal and rng.random() < 0.35 and v * scale > 20 * DUST:
                na = w.deposit()                                        # cash-out leaves the flow
            else:
                if a not in addr_map:
                    # a receiver that never spends inside the flow: counterparty, illicit status unknown
                    ow = owner(a) if a in uf2.parent else w.entity(counterparty_type, None, campaign=cid, quality="unknown")
                    addr_map[a] = w.addr(ow, new_type(ty))
                na = addr_map[a]
            outs.append(na)
            weights.append(max(v, 1))
        total_in = sum(u[3] for u in ins)
        t_new = t_start + (t_real - bt[0]) * squeeze
        h = w.block_for(t_new)
        fee = int(np.ceil(w.feerate(h) * vsize([u[4] for u in ins], [w.af_type(a) for a in outs])))
        avail = total_in - fee
        if avail < DUST * len(outs):
            fee = max(total_in - DUST * len(outs) * 2, 1)
            avail = total_in - fee
        wsum = float(sum(weights))
        vals = [max(DUST, int(avail * wt / wsum)) for wt in weights]
        vals[int(np.argmax(vals))] += avail - sum(vals)
        if min(vals) < DUST:
            continue
        scen, extra = scen_of(row)
        r, created = w.tx(t_new, e, ins, list(zip(outs, vals)), scen, shuffle=False, fee_sats=total_in - sum(vals),
                          campaign_id=cid, replay_of=row.txid, **extra)
        for old_vout, u in zip(row.out_vout, created):
            new_outpoint[(row.txid, old_vout)] = u
        recs.append(r)
    if recs:
        w.pattern(pattern_type, recs, cid, size=len(recs), real_first_block=int(part["block_height"].iloc[0]), **pparams)
        if rng.random() < INJ["seed_coverage"]:
            cand = sorted({a for r in recs for a in r["out_addr"] + r["in_addr"]
                           if a in w.addr_owner and w.entities[w.addr_owner[a]]["is_illicit"] is True})
            if cand:
                w.seeds.append((str(rng.choice(cand)), cid))
    return recs


def _components(txids: pd.Series, prevs: pd.Series) -> pd.Series:
    pos = set(txids)
    uf = cluster.UnionFind()
    for t, ps in zip(txids, prevs):
        uf.find(t)
        for p in ps:
            if p in pos:
                uf.union(t, p)
    return pd.Series([uf.find(t) for t in txids], index=txids.index)


def replay_elliptic(w: World, src: pd.DataFrame, price_hist: pd.Series) -> None:
    """Real Elliptic illicit flow components (2016-17); every spender is an illicit actor."""
    ill = src[src["elliptic_class"] == "illicit"].reset_index(drop=True)
    comp = _components(ill["txid"], ill["in_prev_txid"])
    sizes = comp.value_counts()
    lo, hi = INJ["replay_size"]
    eligible = np.sort(sizes[(sizes >= lo) & (sizes <= hi)].index.to_numpy())
    chosen = w.rng.choice(eligible, size=min(w.cnt("replay_components"), len(eligible)), replace=False)
    for ci, root in enumerate(chosen):
        replay_flow(w, ill[comp.to_numpy() == root], f"EL{ci:03d}", price_hist, role_of=lambda a: "actor",
                    scen_of=lambda row: ("elliptic_replay", {}), actor_type="elliptic_replay_actor",
                    actor_quality="inherited_cioh", pattern_type="elliptic_replay_component")


def replay_ransomwhere(w: World, flows: pd.DataFrame, seeds: pd.DataFrame, price_hist: pd.Series) -> None:
    """Real ransomware money flows (2020-22) traced from Ransomwhere seeds: hop 0 = victim payments (licit initiator,
    pays_illicit), hop 1 = operator spends of ransom addresses, hop 2 = continued layering (kept only when mostly
    funded by hop 1 - otherwise it is a counterparty such as an exchange sweep)."""
    h1 = set(flows.loc[flows["hop"] == 1, "txid"])
    share = flows["in_prev_txid"].map(lambda ps: float(np.mean([p in h1 for p in ps])))
    keep = (flows["hop"] < 2) | ((share >= INJ["rw_hop2_min_share"]) & (flows["n_in"] <= INJ["rw_hop2_max_in"]))
    fl = flows[keep].reset_index(drop=True)
    comp = _components(fl["txid"], fl["in_prev_txid"])
    sizes = comp.value_counts()
    lo, hi = INJ["rw_replay_size"]
    ok = set(sizes[(sizes >= lo) & (sizes <= hi)].index)
    # a component must contain an operator spend (hop 1), otherwise there is no laundering to replay
    has_op = fl.assign(c=comp.to_numpy()).groupby("c")["hop"].apply(lambda h: bool((h == 1).any()))
    eligible = np.sort([c for c in ok if has_op.get(c, False)])
    chosen = w.rng.choice(eligible, size=min(w.cnt("rw_replay_components"), len(eligible)), replace=False)
    for ci, root in enumerate(chosen):
        part = fl[comp.to_numpy() == root]
        hop = dict(zip(part["txid"], part["hop"]))
        victim_in = {a for r in part[part["hop"] == 0].itertuples() for a in r.in_addr}
        actor_in = {a for r in part[part["hop"] > 0].itertuples() for a in r.in_addr}
        family = part["family"].dropna().mode()
        fam = str(family.iloc[0]) if len(family) else "unknown"

        def scen_of(row, hop=hop):
            h = hop[row.txid]
            if h == 0:
                return "rw_real_ransom_payment", {"pays_illicit": True}
            return ("rw_real_operator" if h == 1 else "rw_real_layering"), {}

        replay_flow(w, part, f"RR{ci:03d}", price_hist,
                    role_of=lambda a, v=victim_in, o=actor_in: "victim" if (a in v and a not in o) else "actor",
                    scen_of=scen_of, actor_type="ransomware_operator_real", actor_quality="traced_heuristic",
                    pattern_type="ransomware_real_replay", family=fam)


def build(space, seed: np.random.SeedSequence, out_obs, out_truth, coverage=None, formats=None, window=None) -> dict:
    s_inj, s_net = seed.spawn(2)
    rng = np.random.default_rng(s_inj)
    raw_path = C.RAW / "track_m" / (window or C.TRACK_M_WINDOW)
    raw = pd.read_parquet(raw_path)
    real, excluded = normalize(raw)
    real["data_carrier"] = real["dropped_opreturn"] > 0
    real["origin"] = "real"
    real["scenario"] = np.where(real["data_carrier"], "real_data_carrier", "real_background")

    # real ownership proxy + real exchange deposit addresses (inputs of multi-input sweeps by service clusters)
    ent_real, cj = cluster.cioh_clusters(real)
    real["entity"] = ent_real.to_numpy()
    real["coinjoin_rule"] = cj.to_numpy()
    svc = cluster.service_flags(real, ent_real)
    real["is_service"] = svc.to_numpy()
    sweeps = real[svc.to_numpy() & (real["n_in"] >= 3) & ~real["data_carrier"] & real["coinjoin_rule"].isna()]
    deposit_addrs = pd.unique(np.concatenate(sweeps["in_addr"].to_numpy())) if len(sweeps) else np.array([])
    from .addresses import VSIZE_OUT, validate_address
    deposit_addrs = np.array([a for a in deposit_addrs if validate_address(a) in VSIZE_OUT])

    price = pd.read_csv(C.RAW / "price" / "btc_usd_daily_bitstamp.csv", parse_dates=["date"]).set_index("date")["btc_usd_close"]
    day = real["block_time"].min().tz_localize(None).normalize()
    rw_seeds = pd.read_parquet(C.RAW / "ransomwhere_flows" / "seeds_used.parquet")
    rw_raw = pd.read_parquet(C.RAW / "ransomwhere_flows" / "ransomwhere_flow_txs.parquet")
    rw_led, _ = normalize(rw_raw)
    rw_led = rw_led.merge(rw_raw[["hash", "hop", "family"]].rename(columns={"hash": "txid"}), on="txid", how="left")
    w = World(rng, real, float(price.asof(day)), deposit_addrs, ransom_usd=rw_seeds["usd"].to_numpy())

    # Elliptic illicit flows for replay (real ledger from Track R raw)
    labels = elliptic_labels()
    illicit = set(labels.loc[labels["elliptic_class"] == "illicit", "txid"])
    r_raw = pd.read_parquet(C.RAW / "track_r" / "elliptic_real_txs.parquet")
    r_raw = r_raw[r_raw["hash"].isin(illicit)]                    # only illicit flows are replayed: saves memory
    r_led, _ = normalize(r_raw)
    r_led = r_led.merge(labels, on="txid", how="left")
    del r_raw
    run_scenarios(w, r_led, price, rw_led, rw_seeds)

    in_window = [r for r in w.txs if w.t0 <= r["t_intent"] <= w.t1]
    kept_ids = {r["txid"] for r in in_window}
    censored = len(w.txs) - len(in_window)
    for pt in w.patterns:
        pt["txids"] = [t for t in pt["txids"] if t in kept_ids]
    w.patterns = [pt for pt in w.patterns if pt["txids"]]
    seen_addrs = {a for r in in_window for a in r["in_addr"] + r["out_addr"]}
    w.seeds = [sd for sd in w.seeds if sd[0] in seen_addrs]
    w.txs = in_window
    inj = pd.DataFrame(w.txs)
    inj["n_in"] = inj["in_addr"].str.len()
    inj["n_out"] = inj["out_addr"].str.len()
    inj["feerate"] = inj["fee_sats"] / inj["vsize"]
    inj["block_time"] = pd.to_datetime(inj["block_time"], utc=True)
    ents = pd.DataFrame(list(w.entities.values()))
    inj["is_illicit"] = inj["entity"].map(ents.set_index("entity")["is_illicit"])

    cols = ["txid", "block_height", "block_time", "vsize", "fee_sats", "in_addr", "in_sats", "in_type", "in_prev_txid",
            "in_prev_vout", "out_addr", "out_sats", "out_type", "out_vout", "dropped_opreturn", "script_type", "n_in",
            "n_out", "feerate", "entity", "origin", "scenario"]
    ledger = pd.concat([real[cols], inj[cols]], ignore_index=True).sort_values(["block_height", "txid"]).reset_index(drop=True)

    real_ents = pd.DataFrame({"entity": np.unique(ent_real)})
    real_ents["role"] = np.where(real_ents["entity"].map(svc.groupby(ent_real).max()), "service", "user")
    real_ents["path"] = None
    all_ents = pd.concat([real_ents, ents[["entity", "role", "path"]]], ignore_index=True)
    net, net_truth, ip_truth, stats, ann = network.simulate(ledger, all_ents, space, s_net, coverage=coverage)
    obs = export.observable_table(ledger, net)
    files = export.write_all(obs, out_obs, formats=formats)
    files.update(export.write_announcements(ann, out_obs))

    # --- truth ----------------------------------------------------------------------------------------------------
    lab = ledger[["txid", "origin", "scenario", "entity", "block_height", "block_time", "n_in", "n_out", "feerate"]].copy()
    extra = inj.set_index("txid")
    for col in ["is_illicit", "campaign_id", "pattern_id", "hop_index", "pays_illicit", "is_coinjoin", "is_payjoin",
                "replay_of", "template_of"]:
        lab[col] = lab["txid"].map(extra[col]) if col in extra else None
    lab["coinjoin_rule"] = lab["txid"].map(real.set_index("txid")["coinjoin_rule"])
    lab["entity_truth_quality"] = np.where(lab["origin"] == "real", "cioh_proxy",
                                           lab["entity"].map(ents.set_index("entity")["truth_quality"]))
    # taint: hops from an illicit-initiated injected tx along injected spends
    taint = {t: 0 for t in lab.loc[lab["is_illicit"] == True, "txid"]}                    # noqa: E712
    by_prev = defaultdict(list)
    for t, prevs in zip(inj["txid"], inj["in_prev_txid"]):
        for p in sorted(set(prevs)):
            by_prev[p].append(t)
    frontier = list(taint)
    while frontier:
        nxt = []
        for t in frontier:
            for c in by_prev.get(t, []):
                if c not in taint:
                    taint[c] = taint[t] + 1
                    nxt.append(c)
        frontier = nxt
    lab["taint_hops"] = lab["txid"].map(taint)
    lab["input_outpoints"] = [[f"{t}:{v}" for t, v in zip(a, b)] for a, b in zip(ledger["in_prev_txid"], ledger["in_prev_vout"])]
    lab = lab.merge(net_truth, on="txid")
    lab["split"] = np.where(lab["campaign_id"].notna(),
                            lab["campaign_id"].map(lambda c: "test" if zlib.crc32(str(c).encode()) % 3 == 0 else "train"), "unlabelled")

    out_truth.mkdir(parents=True, exist_ok=True)
    lab.to_parquet(out_truth / "tx_labels.parquet", index=False)
    ent_out = pd.concat([ents.drop(columns=["pref_type"]),
                         real_ents.assign(entity_type="real_cioh_cluster", is_illicit=pd.NA, truth_quality="cioh_proxy")],
                        ignore_index=True)
    ent_out.to_parquet(out_truth / "entities.parquet", index=False)
    wallets = pd.DataFrame([(a, e) for a, e in w.addr_owner.items()], columns=["address", "entity"])
    wallets.to_parquet(out_truth / "wallets_injected.parquet", index=False)
    pd.DataFrame(w.patterns).to_parquet(out_truth / "patterns.parquet", index=False)
    pd.DataFrame(w.seeds, columns=["address", "campaign_id"]).to_parquet(out_truth / "seeds.parquet", index=False)
    ip_truth.to_parquet(out_truth / "ip_truth.parquet", index=False)
    excluded.to_parquet(out_truth / "excluded.parquet", index=False)
    pd.DataFrame({"address": deposit_addrs}).to_parquet(out_truth / "real_deposit_addresses_used_as_cashout.parquet", index=False)

    kinds = {"EL": "elliptic_replay", "RR": "ransomware_real_replay", "RW": "ransomware_scripted", "LD": "launderer_scripted"}
    campaigns = pd.Series(ents["campaign_id"].dropna().unique()).str[:2].map(kinds)
    stats.update({
        "track": "M", "source_window": raw_path.name, "btc_usd": w.price,
        "injected_censored_outside_window": int(censored),
        "kept": int(len(ledger)), "real": int((ledger["origin"] == "real").sum()),
        "injected": int((ledger["origin"] == "injected").sum()),
        "real_data_carrier_share": round(float(real["data_carrier"].mean()), 4),
        "scenario_counts": lab["scenario"].value_counts().to_dict(),
        "illicit_tx": int((lab["is_illicit"] == True).sum()),                                   # noqa: E712
        "illicit_share_all": round(float((lab["is_illicit"] == True).mean()), 5),               # noqa: E712
        "illicit_share_non_data_carrier": round(float((lab["is_illicit"] == True).sum() /
                                                      (lab["scenario"] != "real_data_carrier").sum()), 5),  # noqa: E712
        "injected_licit_tx": int(((lab["origin"] == "injected") & (lab["is_illicit"] == False)).sum()),   # noqa: E712
        "campaigns": campaigns.value_counts().to_dict(), "patterns": pd.DataFrame(w.patterns)["type"].value_counts().to_dict(),
        "seeds": len(w.seeds), "deposit_addresses_available": int(len(deposit_addrs)),
        "excluded_by_reason": excluded["reason"].value_counts().to_dict(),
        "files": files,
    })
    (out_truth / "stats_realized.json").write_text(json.dumps(stats, indent=2, default=str))
    return stats
