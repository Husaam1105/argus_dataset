"""Build a dataset release.

    python -m argusds.build --version 1.0.0 [--tracks r,m]

Both tracks share one shadow address space and one pair of GeoIP databases, so every IP in the release resolves
in observable/geoip/*.mmdb and no two nodes collide across tracks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time

import numpy as np

from . import config as C, geo


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--tracks", default="r,m")
    ap.add_argument("--seed", type=int, default=C.MASTER_SEED)
    ap.add_argument("--coverage", type=float, default=None, help="override sensor_peer_coverage (difficulty grid)")
    ap.add_argument("--formats", default=None, help="comma list; default all five")
    ap.add_argument("--window", default=None, help="Track M raw window file (default config.TRACK_M_WINDOW)")
    args = ap.parse_args()

    rel = C.RELEASES / f"argus-dataset-v{args.version}"
    if rel.exists():
        shutil.rmtree(rel)
    obs_dir, truth_dir = rel / "observable", rel / "truth"
    root = np.random.SeedSequence(args.seed)
    s_space, s_r, s_m = root.spawn(3)
    space = geo.ShadowSpace(geo.load_pool(), np.random.default_rng(s_space))
    stats = {}

    tracks = args.tracks.split(",")
    fmts = args.formats.split(",") if args.formats else None
    if "r" in tracks:
        from . import build_track_r
        t = time.time()
        stats["track_r"] = build_track_r.build(space, s_r, obs_dir / "track_r", truth_dir / "track_r", args.coverage, fmts)
        print(f"[track R] {stats['track_r']['kept']:,} txs in {time.time() - t:.0f}s")
        import gc
        gc.collect()
    if "m" in tracks:
        from . import build_track_m
        t = time.time()
        stats["track_m"] = build_track_m.build(space, s_m, obs_dir / "track_m", truth_dir / "track_m", args.coverage, fmts, args.window)
        print(f"[track M] {stats['track_m']['kept']:,} txs in {time.time() - t:.0f}s")

    prefixes = space.prefix_table()
    stats["geoip"] = geo.write_shadow_mmdb(prefixes, obs_dir / "geoip")
    prefixes.to_parquet(truth_dir / "shadow_prefix_map.parquet", index=False)

    manifest = {"version": args.version, "seed": args.seed, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "files": {str(p.relative_to(rel)).replace("\\", "/"): sha256(p) for p in sorted(rel.rglob("*")) if p.is_file()}}
    (rel / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (truth_dir / "build_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(f"[release] {rel}")


if __name__ == "__main__":
    main()
