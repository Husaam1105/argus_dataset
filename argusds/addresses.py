"""Bitcoin mainnet address encoding / validation (Base58Check, Bech32 BIP-173, Bech32m BIP-350).

Synthetic addresses are encoded from random payloads, so they are valid in format and checksum but correspond to
no known key. `validate_address` is the shared validator for the pipeline's ingestion step.
"""
from __future__ import annotations

import hashlib

import numpy as np

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
BECH32 = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
BECH32M_CONST = 0x2BC830A3


def _sha256d(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


def b58check_encode(version: int, payload: bytes) -> str:
    data = bytes([version]) + payload
    data += _sha256d(data)[:4]
    n = int.from_bytes(data, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = B58[r] + out
    return "1" * (len(data) - len(data.lstrip(b"\0"))) + out


def b58check_decode(s: str) -> tuple[int, bytes] | None:
    n = 0
    for c in s:
        i = B58.find(c)
        if i < 0:
            return None
        n = n * 58 + i
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    raw = b"\0" * (len(s) - len(s.lstrip("1"))) + raw
    if len(raw) < 5 or _sha256d(raw[:-4])[:4] != raw[-4:]:
        return None
    return raw[0], raw[1:-4]


def _polymod(values) -> int:
    gen = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ v
        for i in range(5):
            chk ^= gen[i] if ((b >> i) & 1) else 0
    return chk


def _hrp_expand(hrp: str):
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def _convertbits(data, frombits, tobits, pad=True):
    acc, bits, ret, maxv = 0, 0, [], (1 << tobits) - 1
    for v in data:
        acc = (acc << frombits) | v
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad and bits:
        ret.append((acc << (tobits - bits)) & maxv)
    elif not pad and (bits >= frombits or ((acc << (tobits - bits)) & maxv)):
        return None
    return ret


def segwit_encode(witver: int, program: bytes, hrp: str = "bc") -> str:
    data = [witver] + _convertbits(program, 8, 5)
    const = 1 if witver == 0 else BECH32M_CONST
    pm = _polymod(_hrp_expand(hrp) + data + [0] * 6) ^ const
    return hrp + "1" + "".join(BECH32[d] for d in data + [(pm >> 5 * (5 - i)) & 31 for i in range(6)])


def segwit_decode(addr: str, hrp: str = "bc") -> tuple[int, bytes] | None:
    if addr.lower() != addr and addr.upper() != addr:
        return None
    addr = addr.lower()
    pos = addr.rfind("1")
    if addr[:pos] != hrp or pos + 7 > len(addr) or len(addr) > 90:
        return None
    try:
        data = [BECH32.index(c) for c in addr[pos + 1:]]
    except ValueError:
        return None
    const = _polymod(_hrp_expand(hrp) + data)
    witver = data[0]
    if const != (1 if witver == 0 else BECH32M_CONST):
        return None
    prog = _convertbits(data[1:-6], 5, 8, False)
    if prog is None or not 2 <= len(prog) <= 40 or witver > 16:
        return None
    if witver == 0 and len(prog) not in (20, 32):
        return None
    return witver, bytes(prog)


def validate_address(addr: str) -> str | None:
    """Return the script type (P2PKH/P2SH/P2WPKH/P2WSH/P2TR/P2A/WITNESS_UNKNOWN) or None if invalid."""
    if addr.lower().startswith("bc1"):
        d = segwit_decode(addr)
        if d is None:
            return None
        v, p = d
        if v == 0:
            return "P2WPKH" if len(p) == 20 else "P2WSH"
        if v == 1 and len(p) == 32:
            return "P2TR"
        if v == 1 and p == bytes.fromhex("4e73"):
            return "P2A"
        return "WITNESS_UNKNOWN"
    d = b58check_decode(addr)
    if d is None or len(d[1]) != 20:
        return None
    return {0: "P2PKH", 5: "P2SH"}.get(d[0])


class AddressFactory:
    """Fresh synthetic addresses of a requested script type, never repeating."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.issued: set[str] = set()

    def new(self, script_type: str) -> str:
        while True:
            if script_type == "P2PKH":
                a = b58check_encode(0x00, self.rng.bytes(20))
            elif script_type == "P2SH":
                a = b58check_encode(0x05, self.rng.bytes(20))
            elif script_type == "P2WPKH":
                a = segwit_encode(0, self.rng.bytes(20))
            elif script_type == "P2WSH":
                a = segwit_encode(0, self.rng.bytes(32))
            elif script_type == "P2TR":
                a = segwit_encode(1, self.rng.bytes(32))
            else:
                raise ValueError(script_type)
            if a not in self.issued:
                self.issued.add(a)
                return a


def txid_from(rng: np.random.Generator) -> str:
    """A realistic-looking txid: sha256d of random bytes, displayed byte-reversed like real txids."""
    return _sha256d(rng.bytes(64))[::-1].hex()


# vbytes per input / output by script type [src: standard weight arithmetic; P2WSH input assumes 2-of-3 multisig]
VSIZE_IN = {"P2PKH": 148, "P2SH": 91, "P2WPKH": 68, "P2WSH": 104.5, "P2TR": 57.5}
VSIZE_OUT = {"P2PKH": 34, "P2SH": 32, "P2WPKH": 31, "P2WSH": 43, "P2TR": 43}


def vsize(in_types, out_types) -> int:
    return int(np.ceil(10.5 + sum(VSIZE_IN[t] for t in in_types) + sum(VSIZE_OUT[t] for t in out_types)))
