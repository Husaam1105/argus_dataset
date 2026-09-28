"""Raw BigQuery rows -> canonical ledger table (one row per transaction, integer satoshis).

Scope rules (documented in the dataset card):
  * coinbase transactions are excluded (they have no input addresses);
  * zero-value non-address outputs (OP_RETURN) are dropped from the output arrays and counted in truth;
  * a transaction with a non-address input, or a *valued* non-address output (bare multisig, P2PK, nonstandard),
    is excluded from the observable layer and listed in truth with the reason. Fee stays consistent:
    fee = sum(inputs) - sum(address outputs) holds for every kept transaction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCRIPT_TYPES = {
    "pubkeyhash": "P2PKH",
    "scripthash": "P2SH",
    "witness_v0_keyhash": "P2WPKH",
    "witness_v0_scripthash": "P2WSH",
    "witness_v1_taproot": "P2TR",
    "anchor": "P2A",                 # pay-to-anchor (Bitcoin Core 28+), address bc1pfeessrawgf
}


def _addr(item) -> tuple[str | None, str | None]:
    st = SCRIPT_TYPES.get(item.get("type"))
    addrs = item.get("addresses")
    if st is None or addrs is None or len(addrs) != 1:
        return None, None
    return str(addrs[0]), st


def normalize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (ledger, excluded)."""
    keep, excluded = [], []
    for r in raw.itertuples(index=False):
        if r.is_coinbase:
            excluded.append((r.hash, "coinbase"))
            continue
        in_addr, in_sats, in_type, in_prev, in_vout = [], [], [], [], []
        reason = None
        for i in r.inputs:
            a, st = _addr(i)
            if a is None:
                reason = f"nonaddress_input:{i.get('type')}"
                break
            in_addr.append(a)
            in_sats.append(int(i["value"]))
            in_type.append(st)
            in_prev.append(i["spent_transaction_hash"])
            in_vout.append(int(i["spent_output_index"]))
        if reason:
            excluded.append((r.hash, reason))
            continue
        out_addr, out_sats, out_type, out_vout = [], [], [], []
        dropped_zero = 0
        for o in r.outputs:
            a, st = _addr(o)
            v = int(o["value"])
            if a is None:
                if v == 0:
                    dropped_zero += 1
                    continue
                reason = f"valued_nonaddress_output:{o.get('type')}"
                break
            out_addr.append(a)
            out_sats.append(v)
            out_type.append(st)
            out_vout.append(int(o["index"]))
        if reason or not out_addr:
            excluded.append((r.hash, reason or "no_address_outputs"))
            continue
        fee = sum(in_sats) - sum(out_sats)
        if fee < 0 or fee != int(r.fee):
            excluded.append((r.hash, "fee_mismatch"))
            continue
        keep.append(dict(
            txid=r.hash,
            block_height=int(r.block_number),
            block_time=r.block_timestamp,
            vsize=int(r.virtual_size),
            fee_sats=fee,
            in_addr=in_addr, in_sats=in_sats, in_type=in_type, in_prev_txid=in_prev, in_prev_vout=in_vout,
            out_addr=out_addr, out_sats=out_sats, out_type=out_type, out_vout=out_vout,
            dropped_opreturn=dropped_zero,
            script_type=out_type[int(np.argmax(out_sats))],
        ))
    ledger = pd.DataFrame(keep)
    ledger["block_time"] = pd.to_datetime(ledger["block_time"], utc=True)
    ledger["n_in"] = ledger["in_addr"].str.len()
    ledger["n_out"] = ledger["out_addr"].str.len()
    ledger["feerate"] = ledger["fee_sats"] / ledger["vsize"]
    ledger = ledger.sort_values(["block_height", "txid"]).reset_index(drop=True)
    return ledger, pd.DataFrame(excluded, columns=["txid", "reason"])
