"""Ownership proxy for the real background: common-input-ownership (CIOH) clusters.

Used ONLY to keep the simulated network layer consistent (the same wallet cluster broadcasts from the same
node). It is not ground truth and is never used to score entity resolution on real data; exact entity truth
exists only for injected entities (Track M).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        p = self.parent.setdefault(x, x)
        root = x
        while p != root:
            root, p = p, self.parent.setdefault(p, p)
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def coinjoin_rule(out_sats: list[int], n_in: int) -> str | None:
    """Rule-based CoinJoin flag (silver label). Returns the rule id or None.

    whirlpool_5x5 : exactly 5 inputs and 5 equal outputs
    equal_outputs : >= 3 equal-valued outputs, >= as many inputs, and equal outputs are >= half of all outputs
    """
    n_out = len(out_sats)
    top_val, k = Counter(out_sats).most_common(1)[0]
    if n_in == 5 and n_out == 5 and k == 5:
        return "whirlpool_5x5"
    if k >= 3 and n_in >= k and 2 * k >= n_out and top_val > 0:
        return "equal_outputs"
    return None


def cioh_clusters(ledger: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Returns (initiator cluster id per tx, coinjoin rule per tx). CoinJoin-flagged txs are not unioned."""
    cj = pd.Series([coinjoin_rule(o, n) for o, n in zip(ledger["out_sats"], ledger["n_in"])], index=ledger.index)
    uf = UnionFind()
    for addrs, flag in zip(ledger["in_addr"], cj):
        if flag is None and len(addrs) > 1:
            first = addrs[0]
            for a in addrs[1:]:
                uf.union(first, a)
        else:
            for a in addrs:
                uf.find(a)
    roots = [uf.find(a[0]) for a in ledger["in_addr"]]
    codes, _ = pd.factorize(pd.Series(roots))
    return pd.Series(codes, index=ledger.index, name="entity_proxy"), cj.rename("coinjoin_rule")


def service_flags(ledger: pd.DataFrame, entity: pd.Series, min_txs: int = 20, batch_outputs: int = 20) -> pd.Series:
    """[assume] a cluster is a 'service' (exchange / pool / processor) if it initiates >= min_txs transactions
    in the dataset or ever sends a batch with >= batch_outputs outputs."""
    counts = entity.map(entity.value_counts())
    batch = ledger.groupby(entity)["n_out"].transform("max") >= batch_outputs
    return (counts >= min_txs) | batch
